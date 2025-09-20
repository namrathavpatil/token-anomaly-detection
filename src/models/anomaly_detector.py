"""
Token Anomaly Detection Model using PyTorch and scikit-learn
Detects suspicious patterns in LLM API token usage
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional
import joblib
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class TokenUsageFeatures:
    """Feature engineering for token usage patterns"""
    
    def __init__(self):
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        
    def extract_features(self, requests: List[Dict]) -> np.ndarray:
        """Extract features from token usage requests"""
        features = []
        
        for req in requests:
            feature_vector = self._compute_request_features(req)
            features.append(feature_vector)
            
        return np.array(features)
    
    def _compute_request_features(self, req: Dict) -> List[float]:
        """Compute feature vector for a single request"""
        features = []
        
        # Basic token metrics
        input_tokens = req.get('input_tokens', 0)
        output_tokens = req.get('output_tokens', 0)
        total_tokens = input_tokens + output_tokens
        
        features.extend([
            input_tokens,
            output_tokens,
            total_tokens,
            output_tokens / max(input_tokens, 1),  # Token ratio
        ])
        
        # Temporal features
        timestamp = req.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
            
        hour = timestamp.hour
        day_of_week = timestamp.weekday()
        
        features.extend([
            hour,
            day_of_week,
            np.sin(2 * np.pi * hour / 24),  # Cyclical encoding
            np.cos(2 * np.pi * hour / 24),
        ])
        
        # Request pattern features
        user_id = req.get('user_id', 'unknown')
        session_id = req.get('session_id', 'unknown')
        
        features.extend([
            hash(user_id) % 1000,  # Hash to numeric
            hash(session_id) % 1000,
            req.get('request_duration_ms', 0),
            req.get('tokens_per_second', 0),
        ])
        
        # Content-based features
        prompt_length = len(req.get('prompt', ''))
        features.extend([
            prompt_length,
            req.get('max_new_tokens', 0),
            req.get('temperature', 0.7),
            req.get('top_p', 1.0),
        ])
        
        # API endpoint and model features
        model = req.get('model', 'unknown')
        endpoint = req.get('endpoint', 'unknown')
        
        features.extend([
            hash(model) % 100,
            hash(endpoint) % 100,
        ])
        
        return features

class AnomalyDetector:
    """Main anomaly detection system"""
    
    def __init__(self, contamination: float = 0.12):  # Expect ~12% anomalies
        self.contamination = contamination
        self.isolation_forest = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        self.dbscan = DBSCAN(eps=0.5, min_samples=5)
        self.feature_extractor = TokenUsageFeatures()
        self.is_trained = False
        self.anomaly_threshold = -0.1
        
    def train(self, normal_requests: List[Dict]) -> Dict:
        """Train the anomaly detection model on normal traffic"""
        logger.info(f"Training anomaly detector on {len(normal_requests)} requests")
        
        # Extract features
        X = self.feature_extractor.extract_features(normal_requests)
        
        # Fit scaler
        X_scaled = self.feature_extractor.scaler.fit_transform(X)
        
        # Train Isolation Forest
        self.isolation_forest.fit(X_scaled)
        
        # Calculate threshold based on training data
        scores = self.isolation_forest.decision_function(X_scaled)
        self.anomaly_threshold = np.percentile(scores, contamination * 100)
        
        self.is_trained = True
        
        return {
            'n_samples': len(normal_requests),
            'n_features': X.shape[1],
            'anomaly_threshold': self.anomaly_threshold,
            'contamination': self.contamination
        }
    
    def predict(self, requests: List[Dict]) -> Dict:
        """Predict anomalies in new requests"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
            
        # Extract features
        X = self.feature_extractor.extract_features(requests)
        X_scaled = self.feature_extractor.scaler.transform(X)
        
        # Get anomaly scores
        scores = self.isolation_forest.decision_function(X_scaled)
        predictions = self.isolation_forest.predict(X_scaled)
        
        # Classify as anomalous if score below threshold
        anomalies = scores < self.anomaly_threshold
        
        # Get anomaly reasons
        anomaly_reasons = []
        for i, is_anomaly in enumerate(anomalies):
            if is_anomaly:
                reason = self._analyze_anomaly(requests[i], scores[i])
                anomaly_reasons.append(reason)
            else:
                anomaly_reasons.append(None)
        
        return {
            'predictions': predictions,
            'scores': scores.tolist(),
            'anomalies': anomalies.tolist(),
            'anomaly_reasons': anomaly_reasons,
            'anomaly_rate': np.mean(anomalies)
        }
    
    def _analyze_anomaly(self, request: Dict, score: float) -> Dict:
        """Analyze why a request was flagged as anomalous"""
        reasons = []
        
        # Check token usage patterns
        input_tokens = request.get('input_tokens', 0)
        output_tokens = request.get('output_tokens', 0)
        
        if input_tokens > 50000:  # Very long input
            reasons.append(f"Extremely long input: {input_tokens} tokens")
        if output_tokens > 20000:  # Very long output
            reasons.append(f"Extremely long output: {output_tokens} tokens")
        if output_tokens / max(input_tokens, 1) > 10:  # High ratio
            reasons.append(f"High output/input ratio: {output_tokens/input_tokens:.2f}")
            
        # Check timing patterns
        timestamp = request.get('timestamp', datetime.now())
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
            
        hour = timestamp.hour
        if hour < 6 or hour > 23:  # Unusual hours
            reasons.append(f"Unusual time: {hour}:00")
            
        # Check request parameters
        if request.get('temperature', 0.7) > 1.5:
            reasons.append(f"High temperature: {request.get('temperature')}")
        if request.get('max_new_tokens', 0) > 10000:
            reasons.append(f"Very high max tokens: {request.get('max_new_tokens')}")
            
        return {
            'score': float(score),
            'reasons': reasons,
            'severity': 'high' if score < -0.5 else 'medium'
        }
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'isolation_forest': self.isolation_forest,
            'scaler': self.feature_extractor.scaler,
            'anomaly_threshold': self.anomaly_threshold,
            'contamination': self.contamination,
            'is_trained': self.is_trained
        }
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        model_data = joblib.load(filepath)
        self.isolation_forest = model_data['isolation_forest']
        self.feature_extractor.scaler = model_data['scaler']
        self.anomaly_threshold = model_data['anomaly_threshold']
        self.contamination = model_data['contamination']
        self.is_trained = model_data['is_trained']
        logger.info(f"Model loaded from {filepath}")

class RealTimeMonitor:
    """Real-time monitoring and alerting system"""
    
    def __init__(self, detector: AnomalyDetector):
        self.detector = detector
        self.alert_threshold = 0.15  # Alert if anomaly rate exceeds 15%
        self.recent_requests = []
        self.alert_history = []
        
    def process_request(self, request: Dict) -> Dict:
        """Process a single request and check for anomalies"""
        # Add timestamp if not present
        if 'timestamp' not in request:
            request['timestamp'] = datetime.now().isoformat()
            
        # Store request
        self.recent_requests.append(request)
        
        # Keep only last 1000 requests in memory
        if len(self.recent_requests) > 1000:
            self.recent_requests = self.recent_requests[-1000:]
            
        # Check for anomalies in recent batch
        result = self.detector.predict([request])
        
        # Check if we need to alert
        if len(self.recent_requests) >= 10:  # Minimum batch size for alerting
            recent_anomaly_rate = self._calculate_recent_anomaly_rate()
            if recent_anomaly_rate > self.alert_threshold:
                alert = self._create_alert(recent_anomaly_rate)
                self.alert_history.append(alert)
                
        result['recent_anomaly_rate'] = self._calculate_recent_anomaly_rate()
        result['alert_triggered'] = len(self.alert_history) > 0 and \
                                  self.alert_history[-1]['timestamp'] > datetime.now() - timedelta(minutes=5)
        
        return result
    
    def _calculate_recent_anomaly_rate(self) -> float:
        """Calculate anomaly rate for recent requests"""
        if len(self.recent_requests) < 10:
            return 0.0
            
        recent_batch = self.recent_requests[-50:]  # Last 50 requests
        result = self.detector.predict(recent_batch)
        return result['anomaly_rate']
    
    def _create_alert(self, anomaly_rate: float) -> Dict:
        """Create an alert for high anomaly rate"""
        return {
            'timestamp': datetime.now().isoformat(),
            'anomaly_rate': anomaly_rate,
            'alert_type': 'high_anomaly_rate',
            'message': f"Anomaly rate {anomaly_rate:.2%} exceeds threshold {self.alert_threshold:.2%}"
        }
    
    def get_recent_alerts(self, hours: int = 24) -> List[Dict]:
        """Get recent alerts"""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [
            alert for alert in self.alert_history 
            if datetime.fromisoformat(alert['timestamp']) > cutoff
        ]
