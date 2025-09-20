"""
Test script for the Token Anomaly Detection System
Validates the ~12% anomaly detection rate on synthetic traffic
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime
from typing import List, Dict
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.data_generator import generate_synthetic_traffic, generate_test_scenarios

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SystemTester:
    """Test the anomaly detection system with synthetic data"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = None
        
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def test_single_request(self, request_data: Dict) -> Dict:
        """Test a single request through the API"""
        try:
            async with self.session.post(
                f"{self.base_url}/api/detect",
                json=request_data
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"API error: {response.status} - {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Request failed: {e}")
            return None
    
    async def test_batch_requests(self, requests: List[Dict]) -> Dict:
        """Test a batch of requests through the API"""
        try:
            async with self.session.post(
                f"{self.base_url}/api/batch-detect",
                json=requests
            ) as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Batch API error: {response.status} - {await response.text()}")
                    return None
        except Exception as e:
            logger.error(f"Batch request failed: {e}")
            return None
    
    async def get_metrics(self) -> Dict:
        """Get system metrics"""
        try:
            async with self.session.get(f"{self.base_url}/api/metrics") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.error(f"Metrics API error: {response.status}")
                    return None
        except Exception as e:
            logger.error(f"Metrics request failed: {e}")
            return None
    
    async def test_anomaly_detection_rate(self, num_requests: int = 1000) -> Dict:
        """Test the anomaly detection rate on synthetic data"""
        logger.info(f"Generating {num_requests} synthetic requests with 12% anomaly rate...")
        
        # Generate synthetic data with known anomaly rate
        synthetic_data = generate_synthetic_traffic(num_requests, anomaly_rate=0.12)
        
        # Separate normal and anomalous requests
        actual_anomalies = [req for req in synthetic_data if req.get('is_anomaly', False)]
        actual_normal = [req for req in synthetic_data if not req.get('is_anomaly', False)]
        
        logger.info(f"Generated {len(actual_anomalies)} anomalous and {len(actual_normal)} normal requests")
        
        # Test batch detection
        batch_size = 50
        results = []
        
        for i in range(0, len(synthetic_data), batch_size):
            batch = synthetic_data[i:i + batch_size]
            
            # Convert to API format
            api_requests = []
            for req in batch:
                api_req = {
                    'user_id': req['user_id'],
                    'session_id': req['session_id'],
                    'model': req['model'],
                    'endpoint': req['endpoint'],
                    'input_tokens': req['input_tokens'],
                    'output_tokens': req['output_tokens'],
                    'prompt': req['prompt'],
                    'timestamp': req['timestamp'],
                    'request_duration_ms': req.get('request_duration_ms'),
                    'tokens_per_second': req.get('tokens_per_second'),
                    'max_new_tokens': req.get('max_new_tokens'),
                    'temperature': req.get('temperature'),
                    'top_p': req.get('top_p')
                }
                api_requests.append(api_req)
            
            # Test batch
            batch_result = await self.test_batch_requests(api_requests)
            if batch_result:
                results.extend(batch_result['results'])
            
            # Small delay to avoid overwhelming the API
            await asyncio.sleep(0.1)
        
        # Analyze results
        detected_anomalies = sum(1 for result in results if result['is_anomaly'])
        detected_normal = len(results) - detected_anomalies
        
        # Calculate metrics
        actual_anomaly_count = len(actual_anomalies)
        detected_anomaly_count = detected_anomalies
        false_positives = 0
        false_negatives = 0
        
        for i, result in enumerate(results):
            actual_is_anomaly = synthetic_data[i].get('is_anomaly', False)
            detected_is_anomaly = result['is_anomaly']
            
            if actual_is_anomaly and not detected_is_anomaly:
                false_negatives += 1
            elif not actual_is_anomaly and detected_is_anomaly:
                false_positives += 1
        
        precision = detected_anomaly_count / max(detected_anomaly_count, 1)
        recall = (actual_anomaly_count - false_negatives) / max(actual_anomaly_count, 1)
        f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
        
        test_results = {
            'total_requests': len(synthetic_data),
            'actual_anomalies': actual_anomaly_count,
            'actual_anomaly_rate': actual_anomaly_count / len(synthetic_data),
            'detected_anomalies': detected_anomaly_count,
            'detected_anomaly_rate': detected_anomaly_count / len(synthetic_data),
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'accuracy': (len(synthetic_data) - false_positives - false_negatives) / len(synthetic_data)
        }
        
        return test_results
    
    async def test_specific_scenarios(self) -> Dict:
        """Test specific anomaly scenarios"""
        logger.info("Testing specific anomaly scenarios...")
        
        scenarios = generate_test_scenarios()
        scenario_results = {}
        
        for scenario_name, requests in scenarios.items():
            logger.info(f"Testing scenario: {scenario_name}")
            
            # Convert to API format
            api_requests = []
            for req in requests:
                api_req = {
                    'user_id': req['user_id'],
                    'session_id': req['session_id'],
                    'model': req['model'],
                    'endpoint': req['endpoint'],
                    'input_tokens': req['input_tokens'],
                    'output_tokens': req['output_tokens'],
                    'prompt': req['prompt'],
                    'timestamp': req['timestamp'],
                    'request_duration_ms': req.get('request_duration_ms'),
                    'tokens_per_second': req.get('tokens_per_second'),
                    'max_new_tokens': req.get('max_new_tokens'),
                    'temperature': req.get('temperature'),
                    'top_p': req.get('top_p')
                }
                api_requests.append(api_req)
            
            # Test the scenario
            result = await self.test_batch_requests(api_requests)
            
            if result:
                anomaly_count = result['anomaly_count']
                anomaly_rate = result['anomaly_rate']
                
                scenario_results[scenario_name] = {
                    'total_requests': len(requests),
                    'detected_anomalies': anomaly_count,
                    'anomaly_rate': anomaly_rate,
                    'expected_anomalies': sum(1 for req in requests if req.get('is_anomaly', False))
                }
            
            await asyncio.sleep(0.1)
        
        return scenario_results
    
    async def test_real_time_monitoring(self, num_requests: int = 100) -> Dict:
        """Test real-time monitoring capabilities"""
        logger.info(f"Testing real-time monitoring with {num_requests} requests...")
        
        synthetic_data = generate_synthetic_traffic(num_requests, anomaly_rate=0.15)
        
        # Send requests one by one to simulate real-time traffic
        results = []
        for req in synthetic_data:
            api_req = {
                'user_id': req['user_id'],
                'session_id': req['session_id'],
                'model': req['model'],
                'endpoint': req['endpoint'],
                'input_tokens': req['input_tokens'],
                'output_tokens': req['output_tokens'],
                'prompt': req['prompt'],
                'timestamp': req['timestamp'],
                'request_duration_ms': req.get('request_duration_ms'),
                'tokens_per_second': req.get('tokens_per_second'),
                'max_new_tokens': req.get('max_new_tokens'),
                'temperature': req.get('temperature'),
                'top_p': req.get('top_p')
            }
            
            result = await self.test_single_request(api_req)
            if result:
                results.append(result)
            
            # Small delay between requests
            await asyncio.sleep(0.05)
        
        # Get final metrics
        final_metrics = await self.get_metrics()
        
        return {
            'requests_sent': len(results),
            'final_metrics': final_metrics,
            'average_anomaly_rate': sum(r['recent_anomaly_rate'] for r in results) / len(results) if results else 0
        }

async def run_comprehensive_test():
    """Run comprehensive tests of the anomaly detection system"""
    logger.info("Starting comprehensive system tests...")
    
    async with SystemTester() as tester:
        # Test 1: Anomaly detection rate validation
        logger.info("=" * 60)
        logger.info("TEST 1: Anomaly Detection Rate Validation")
        logger.info("=" * 60)
        
        detection_results = await tester.test_anomaly_detection_rate(1000)
        
        logger.info(f"Total requests: {detection_results['total_requests']}")
        logger.info(f"Actual anomalies: {detection_results['actual_anomalies']} ({detection_results['actual_anomaly_rate']:.2%})")
        logger.info(f"Detected anomalies: {detection_results['detected_anomalies']} ({detection_results['detected_anomaly_rate']:.2%})")
        logger.info(f"False positives: {detection_results['false_positives']}")
        logger.info(f"False negatives: {detection_results['false_negatives']}")
        logger.info(f"Precision: {detection_results['precision']:.3f}")
        logger.info(f"Recall: {detection_results['recall']:.3f}")
        logger.info(f"F1 Score: {detection_results['f1_score']:.3f}")
        logger.info(f"Accuracy: {detection_results['accuracy']:.3f}")
        
        # Test 2: Specific scenarios
        logger.info("\n" + "=" * 60)
        logger.info("TEST 2: Specific Anomaly Scenarios")
        logger.info("=" * 60)
        
        scenario_results = await tester.test_specific_scenarios()
        
        for scenario_name, results in scenario_results.items():
            logger.info(f"\n{scenario_name}:")
            logger.info(f"  Total requests: {results['total_requests']}")
            logger.info(f"  Expected anomalies: {results['expected_anomalies']}")
            logger.info(f"  Detected anomalies: {results['detected_anomalies']}")
            logger.info(f"  Detection rate: {results['anomaly_rate']:.2%}")
        
        # Test 3: Real-time monitoring
        logger.info("\n" + "=" * 60)
        logger.info("TEST 3: Real-time Monitoring")
        logger.info("=" * 60)
        
        real_time_results = await tester.test_real_time_monitoring(200)
        
        logger.info(f"Requests sent: {real_time_results['requests_sent']}")
        logger.info(f"Average anomaly rate: {real_time_results['average_anomaly_rate']:.2%}")
        
        if real_time_results['final_metrics']:
            metrics = real_time_results['final_metrics']
            logger.info(f"Final system metrics:")
            logger.info(f"  Total requests: {metrics['total_requests']}")
            logger.info(f"  Anomaly count: {metrics['anomaly_count']}")
            logger.info(f"  Anomaly rate: {metrics['anomaly_rate']:.2%}")
            logger.info(f"  System status: {metrics['system_status']}")
        
        # Test 4: System metrics
        logger.info("\n" + "=" * 60)
        logger.info("TEST 4: System Metrics")
        logger.info("=" * 60)
        
        metrics = await tester.get_metrics()
        if metrics:
            logger.info(f"System metrics retrieved successfully:")
            logger.info(f"  Total requests: {metrics['total_requests']}")
            logger.info(f"  Anomaly count: {metrics['anomaly_count']}")
            logger.info(f"  Anomaly rate: {metrics['anomaly_rate']:.2%}")
            logger.info(f"  Recent alerts: {len(metrics['recent_alerts'])}")
            logger.info(f"  System status: {metrics['system_status']}")
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST SUMMARY")
        logger.info("=" * 60)
        
        target_anomaly_rate = 0.12  # 12%
        detected_rate = detection_results['detected_anomaly_rate']
        rate_difference = abs(detected_rate - target_anomaly_rate)
        
        logger.info(f"Target anomaly rate: {target_anomaly_rate:.2%}")
        logger.info(f"Detected anomaly rate: {detected_rate:.2%}")
        logger.info(f"Rate difference: {rate_difference:.2%}")
        
        if rate_difference < 0.05:  # Within 5% of target
            logger.info("✅ SUCCESS: Anomaly detection rate is within acceptable range!")
        else:
            logger.warning("⚠️  WARNING: Anomaly detection rate differs significantly from target")
        
        if detection_results['f1_score'] > 0.7:
            logger.info("✅ SUCCESS: F1 score indicates good model performance!")
        else:
            logger.warning("⚠️  WARNING: F1 score suggests room for improvement")
        
        logger.info("\nAll tests completed!")

if __name__ == "__main__":
    asyncio.run(run_comprehensive_test())
