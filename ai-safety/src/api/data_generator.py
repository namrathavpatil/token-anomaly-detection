"""
Synthetic data generator for testing and training the anomaly detection system
"""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict
import json

class SyntheticTrafficGenerator:
    """Generate synthetic LLM API traffic with normal and anomalous patterns"""
    
    def __init__(self):
        self.models = [
            'gpt-4', 'gpt-3.5-turbo', 'claude-3-opus', 'claude-3-sonnet',
            'llama-2-70b', 'mistral-7b', 'codellama-34b'
        ]
        self.endpoints = [
            '/v1/chat/completions', '/v1/completions', '/v1/embeddings',
            '/v1/moderations', '/v1/fine-tunes'
        ]
        self.user_ids = [f"user_{i:04d}" for i in range(1000)]
        
    def generate_normal_request(self) -> Dict:
        """Generate a normal request with typical usage patterns"""
        timestamp = datetime.now() - timedelta(
            hours=random.randint(0, 24),
            minutes=random.randint(0, 59)
        )
        
        # Normal usage patterns
        input_tokens = random.randint(10, 2000)
        output_tokens = random.randint(1, min(input_tokens * 2, 4000))
        
        return {
            'user_id': random.choice(self.user_ids),
            'session_id': f"session_{random.randint(1000, 9999)}",
            'model': random.choice(self.models),
            'endpoint': random.choice(self.endpoints),
            'input_tokens': input_tokens,
            'output_tokens': output_tokens,
            'prompt': f"This is a normal prompt with {input_tokens} tokens",
            'timestamp': timestamp.isoformat(),
            'request_duration_ms': random.randint(100, 5000),
            'tokens_per_second': random.uniform(10, 100),
            'max_new_tokens': random.randint(100, 2000),
            'temperature': random.uniform(0.1, 1.0),
            'top_p': random.uniform(0.7, 1.0),
            'is_anomaly': False
        }
    
    def generate_anomalous_request(self) -> Dict:
        """Generate an anomalous request with suspicious patterns"""
        timestamp = datetime.now() - timedelta(
            hours=random.randint(0, 24),
            minutes=random.randint(0, 59)
        )
        
        anomaly_type = random.choice([
            'extreme_tokens', 'unusual_timing', 'high_ratio', 
            'suspicious_params', 'rapid_requests', 'long_prompts'
        ])
        
        base_request = self.generate_normal_request()
        base_request['is_anomaly'] = True
        base_request['anomaly_type'] = anomaly_type
        
        if anomaly_type == 'extreme_tokens':
            # Extremely high token usage
            base_request['input_tokens'] = random.randint(50000, 200000)
            base_request['output_tokens'] = random.randint(10000, 50000)
            
        elif anomaly_type == 'unusual_timing':
            # Requests at unusual hours
            unusual_hour = random.choice([1, 2, 3, 4, 5])  # 1-5 AM
            timestamp = timestamp.replace(hour=unusual_hour, minute=random.randint(0, 59))
            base_request['timestamp'] = timestamp.isoformat()
            
        elif anomaly_type == 'high_ratio':
            # Very high output to input ratio
            base_request['input_tokens'] = random.randint(100, 1000)
            base_request['output_tokens'] = random.randint(10000, 50000)
            
        elif anomaly_type == 'suspicious_params':
            # Suspicious parameter values
            base_request['temperature'] = random.uniform(1.5, 2.0)
            base_request['max_new_tokens'] = random.randint(10000, 50000)
            
        elif anomaly_type == 'rapid_requests':
            # Very fast requests (potential abuse)
            base_request['request_duration_ms'] = random.randint(1, 50)
            base_request['tokens_per_second'] = random.uniform(500, 2000)
            
        elif anomaly_type == 'long_prompts':
            # Extremely long prompts
            base_request['input_tokens'] = random.randint(30000, 100000)
            base_request['prompt'] = f"Very long prompt with {base_request['input_tokens']} tokens " * 100
            
        return base_request
    
    def generate_traffic_batch(self, count: int, anomaly_rate: float = 0.12) -> List[Dict]:
        """Generate a batch of traffic with specified anomaly rate"""
        requests = []
        anomaly_count = int(count * anomaly_rate)
        
        # Generate normal requests
        for _ in range(count - anomaly_count):
            requests.append(self.generate_normal_request())
            
        # Generate anomalous requests
        for _ in range(anomaly_count):
            requests.append(self.generate_anomalous_request())
            
        # Shuffle to mix normal and anomalous requests
        random.shuffle(requests)
        
        return requests

def generate_synthetic_traffic(count: int = 1000, anomaly_rate: float = 0.12) -> List[Dict]:
    """Generate synthetic traffic for testing and training"""
    generator = SyntheticTrafficGenerator()
    return generator.generate_traffic_batch(count, anomaly_rate)

def generate_user_sessions(num_sessions: int = 50) -> List[Dict]:
    """Generate realistic user sessions with multiple requests"""
    generator = SyntheticTrafficGenerator()
    sessions = []
    
    for session_id in range(num_sessions):
        user_id = random.choice(generator.user_ids)
        session_start = datetime.now() - timedelta(
            hours=random.randint(0, 48),
            minutes=random.randint(0, 59)
        )
        
        # Generate 1-10 requests per session
        num_requests = random.randint(1, 10)
        session_requests = []
        
        for i in range(num_requests):
            request = generator.generate_normal_request()
            request['user_id'] = user_id
            request['session_id'] = f"session_{session_id}"
            request['timestamp'] = (session_start + timedelta(minutes=i*5)).isoformat()
            session_requests.append(request)
        
        # Occasionally make entire session anomalous
        if random.random() < 0.05:  # 5% of sessions are anomalous
            for req in session_requests:
                req['is_anomaly'] = True
                req['anomaly_type'] = 'session_abuse'
        
        sessions.extend(session_requests)
    
    return sessions

def generate_test_scenarios() -> Dict[str, List[Dict]]:
    """Generate specific test scenarios for validation"""
    generator = SyntheticTrafficGenerator()
    scenarios = {}
    
    # Scenario 1: Normal baseline
    scenarios['normal_baseline'] = [
        generator.generate_normal_request() 
        for _ in range(100)
    ]
    
    # Scenario 2: High anomaly rate
    scenarios['high_anomaly'] = generator.generate_traffic_batch(100, anomaly_rate=0.3)
    
    # Scenario 3: Specific attack patterns
    scenarios['token_abuse'] = []
    for _ in range(20):
        req = generator.generate_anomalous_request()
        req['anomaly_type'] = 'extreme_tokens'
        req['input_tokens'] = random.randint(100000, 500000)
        scenarios['token_abuse'].append(req)
    
    # Scenario 4: Timing attacks
    scenarios['timing_attacks'] = []
    for _ in range(20):
        req = generator.generate_anomalous_request()
        req['anomaly_type'] = 'unusual_timing'
        req['timestamp'] = (datetime.now() - timedelta(
            hours=random.choice([1, 2, 3, 4, 5]),
            minutes=random.randint(0, 59)
        )).isoformat()
        scenarios['timing_attacks'].append(req)
    
    return scenarios

def save_synthetic_data(data: List[Dict], filename: str):
    """Save synthetic data to JSON file"""
    with open(filename, 'w') as f:
        json.dump(data, f, indent=2, default=str)

def load_synthetic_data(filename: str) -> List[Dict]:
    """Load synthetic data from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)

if __name__ == "__main__":
    # Generate test data
    print("Generating synthetic traffic data...")
    
    # Generate training data
    training_data = generate_synthetic_traffic(5000, anomaly_rate=0.12)
    save_synthetic_data(training_data, "data/training_data.json")
    print(f"Generated {len(training_data)} training samples")
    
    # Generate test scenarios
    test_scenarios = generate_test_scenarios()
    for scenario_name, data in test_scenarios.items():
        save_synthetic_data(data, f"data/test_{scenario_name}.json")
        print(f"Generated {len(data)} samples for scenario: {scenario_name}")
    
    # Generate user sessions
    session_data = generate_user_sessions(100)
    save_synthetic_data(session_data, "data/user_sessions.json")
    print(f"Generated {len(session_data)} requests across user sessions")
    
    print("Data generation complete!")
