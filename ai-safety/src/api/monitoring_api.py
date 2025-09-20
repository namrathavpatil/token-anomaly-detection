"""
FastAPI-based monitoring API for real-time token anomaly detection
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
import asyncio
import logging
from datetime import datetime, timedelta
import redis
from contextlib import asynccontextmanager

from ..models.anomaly_detector import AnomalyDetector, RealTimeMonitor
from .data_generator import generate_synthetic_traffic

logger = logging.getLogger(__name__)

# Global variables for the monitoring system
detector = None
monitor = None
redis_client = None
connected_websockets = []

class TokenRequest(BaseModel):
    """Schema for incoming token usage requests"""
    user_id: str
    session_id: str
    model: str
    endpoint: str
    input_tokens: int
    output_tokens: int
    prompt: str
    timestamp: Optional[str] = None
    request_duration_ms: Optional[int] = None
    tokens_per_second: Optional[float] = None
    max_new_tokens: Optional[int] = None
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0

class AnomalyResponse(BaseModel):
    """Response schema for anomaly detection"""
    is_anomaly: bool
    anomaly_score: float
    anomaly_reasons: Optional[List[str]] = None
    severity: Optional[str] = None
    recent_anomaly_rate: float
    alert_triggered: bool

class AlertInfo(BaseModel):
    """Alert information schema"""
    timestamp: str
    anomaly_rate: float
    alert_type: str
    message: str

class MetricsResponse(BaseModel):
    """System metrics response"""
    total_requests: int
    anomaly_count: int
    anomaly_rate: float
    recent_alerts: List[AlertInfo]
    system_status: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global detector, monitor, redis_client
    
    # Initialize Redis connection
    try:
        redis_client = redis.Redis(host='localhost', port=6379, decode_responses=True)
        redis_client.ping()
        logger.info("Connected to Redis")
    except Exception as e:
        logger.warning(f"Redis connection failed: {e}. Using in-memory storage.")
        redis_client = None
    
    # Initialize anomaly detector
    detector = AnomalyDetector(contamination=0.12)
    monitor = RealTimeMonitor(detector)
    
    # Load pre-trained model or train on synthetic data
    try:
        detector.load_model("models/token_anomaly_model.pkl")
        logger.info("Loaded pre-trained anomaly detection model")
    except FileNotFoundError:
        logger.info("No pre-trained model found. Training on synthetic data...")
        synthetic_data = generate_synthetic_traffic(1000, anomaly_rate=0.12)
        normal_data = [req for req in synthetic_data if not req.get('is_anomaly', False)]
        detector.train(normal_data)
        detector.save_model("models/token_anomaly_model.pkl")
        logger.info("Trained and saved anomaly detection model")
    
    yield
    
    # Cleanup
    if redis_client:
        redis_client.close()

app = FastAPI(
    title="Token Anomaly Detection API",
    description="Real-time monitoring system for LLM API token usage anomalies",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the monitoring dashboard"""
    with open("src/dashboard/dashboard.html", "r") as f:
        return HTMLResponse(content=f.read())

@app.post("/api/detect", response_model=AnomalyResponse)
async def detect_anomaly(request: TokenRequest, background_tasks: BackgroundTasks):
    """Detect anomalies in a single token request"""
    try:
        # Convert request to dict
        request_dict = request.dict()
        if not request_dict.get('timestamp'):
            request_dict['timestamp'] = datetime.now().isoformat()
        
        # Process through monitoring system
        result = monitor.process_request(request_dict)
        
        # Store in Redis for persistence
        if redis_client:
            background_tasks.add_task(
                _store_request, 
                request_dict, 
                result
            )
        
        # Broadcast to WebSocket clients
        await _broadcast_update(request_dict, result)
        
        # Prepare response
        anomaly_info = result.get('anomaly_reasons', [None])[0]
        response = AnomalyResponse(
            is_anomaly=result['anomalies'][0],
            anomaly_score=result['scores'][0],
            anomaly_reasons=anomaly_info['reasons'] if anomaly_info else None,
            severity=anomaly_info['severity'] if anomaly_info else None,
            recent_anomaly_rate=result['recent_anomaly_rate'],
            alert_triggered=result['alert_triggered']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error detecting anomaly: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/batch-detect")
async def detect_anomalies_batch(requests: List[TokenRequest], background_tasks: BackgroundTasks):
    """Detect anomalies in a batch of requests"""
    try:
        # Convert requests to dicts
        request_dicts = []
        for req in requests:
            req_dict = req.dict()
            if not req_dict.get('timestamp'):
                req_dict['timestamp'] = datetime.now().isoformat()
            request_dicts.append(req_dict)
        
        # Process batch
        result = detector.predict(request_dicts)
        
        # Store results
        if redis_client:
            for req_dict, anomaly in zip(request_dicts, result['anomalies']):
                background_tasks.add_task(_store_request, req_dict, {'anomalies': [anomaly]})
        
        return {
            'total_requests': len(requests),
            'anomaly_count': sum(result['anomalies']),
            'anomaly_rate': result['anomaly_rate'],
            'results': [
                {
                    'request_id': i,
                    'is_anomaly': result['anomalies'][i],
                    'score': result['scores'][i],
                    'reasons': result['anomaly_reasons'][i]['reasons'] if result['anomaly_reasons'][i] else None
                }
                for i in range(len(requests))
            ]
        }
        
    except Exception as e:
        logger.error(f"Error in batch detection: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metrics", response_model=MetricsResponse)
async def get_metrics():
    """Get current system metrics"""
    try:
        # Get recent alerts
        recent_alerts = monitor.get_recent_alerts(hours=24)
        alert_infos = [
            AlertInfo(
                timestamp=alert['timestamp'],
                anomaly_rate=alert['anomaly_rate'],
                alert_type=alert['alert_type'],
                message=alert['message']
            )
            for alert in recent_alerts
        ]
        
        # Calculate metrics
        total_requests = len(monitor.recent_requests)
        anomaly_count = sum(1 for req in monitor.recent_requests 
                           if detector.predict([req])['anomalies'][0])
        anomaly_rate = anomaly_count / max(total_requests, 1)
        
        return MetricsResponse(
            total_requests=total_requests,
            anomaly_count=anomaly_count,
            anomaly_rate=anomaly_rate,
            recent_alerts=alert_infos,
            system_status="healthy" if anomaly_rate < 0.15 else "warning"
        )
        
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/alerts")
async def get_alerts(hours: int = 24):
    """Get recent alerts"""
    try:
        alerts = monitor.get_recent_alerts(hours=hours)
        return {
            'alerts': alerts,
            'count': len(alerts),
            'hours': hours
        }
    except Exception as e:
        logger.error(f"Error getting alerts: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.websocket("/ws/monitor")
async def websocket_monitor(websocket: WebSocket):
    """WebSocket endpoint for real-time monitoring updates"""
    await websocket.accept()
    connected_websockets.append(websocket)
    
    try:
        # Send initial metrics
        metrics = await get_metrics()
        await websocket.send_text(json.dumps({
            'type': 'metrics',
            'data': metrics.dict()
        }))
        
        # Keep connection alive
        while True:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            await websocket.send_text(json.dumps({
                'type': 'heartbeat',
                'timestamp': datetime.now().isoformat()
            }))
            
    except WebSocketDisconnect:
        connected_websockets.remove(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        if websocket in connected_websockets:
            connected_websockets.remove(websocket)

async def _store_request(request: Dict, result: Dict):
    """Store request and result in Redis"""
    if not redis_client:
        return
        
    try:
        # Store request
        request_key = f"request:{datetime.now().timestamp()}"
        redis_client.hset(request_key, mapping={
            'user_id': request.get('user_id', 'unknown'),
            'session_id': request.get('session_id', 'unknown'),
            'model': request.get('model', 'unknown'),
            'input_tokens': request.get('input_tokens', 0),
            'output_tokens': request.get('output_tokens', 0),
            'timestamp': request.get('timestamp', ''),
            'is_anomaly': str(result.get('anomalies', [False])[0])
        })
        
        # Set expiration (7 days)
        redis_client.expire(request_key, 7 * 24 * 60 * 60)
        
        # Update counters
        redis_client.incr('total_requests')
        if result.get('anomalies', [False])[0]:
            redis_client.incr('anomaly_count')
            
    except Exception as e:
        logger.error(f"Error storing request: {e}")

async def _broadcast_update(request: Dict, result: Dict):
    """Broadcast update to all connected WebSocket clients"""
    if not connected_websockets:
        return
        
    update_data = {
        'type': 'new_request',
        'request': request,
        'result': result,
        'timestamp': datetime.now().isoformat()
    }
    
    # Send to all connected clients
    disconnected = []
    for websocket in connected_websockets:
        try:
            await websocket.send_text(json.dumps(update_data))
        except:
            disconnected.append(websocket)
    
    # Remove disconnected clients
    for websocket in disconnected:
        connected_websockets.remove(websocket)

@app.post("/api/retrain")
async def retrain_model(background_tasks: BackgroundTasks):
    """Retrain the anomaly detection model"""
    try:
        background_tasks.add_task(_retrain_model_task)
        return {"message": "Model retraining started"}
    except Exception as e:
        logger.error(f"Error starting retrain: {e}")
        raise HTTPException(status_code=500, detail=str(e))

async def _retrain_model_task():
    """Background task to retrain the model"""
    try:
        global detector, monitor
        
        # Generate new synthetic data
        synthetic_data = generate_synthetic_traffic(2000, anomaly_rate=0.12)
        normal_data = [req for req in synthetic_data if not req.get('is_anomaly', False)]
        
        # Train new model
        new_detector = AnomalyDetector(contamination=0.12)
        new_detector.train(normal_data)
        new_detector.save_model("models/token_anomaly_model.pkl")
        
        # Update global detector
        detector = new_detector
        monitor = RealTimeMonitor(detector)
        
        logger.info("Model retraining completed successfully")
        
    except Exception as e:
        logger.error(f"Error in model retraining: {e}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
