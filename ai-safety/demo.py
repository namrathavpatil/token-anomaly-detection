"""
Demo script to showcase the Token Anomaly Detection System
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.data_generator import generate_synthetic_traffic
from src.models.anomaly_detector import AnomalyDetector, RealTimeMonitor

def demo_anomaly_detection():
    """Demonstrate the anomaly detection system with synthetic data"""
    print("🔍 Token Anomaly Detection System Demo")
    print("=" * 50)
    
    # Generate synthetic data
    print("\n📊 Generating synthetic traffic data...")
    synthetic_data = generate_synthetic_traffic(500, anomaly_rate=0.12)
    
    # Separate normal and anomalous requests
    normal_data = [req for req in synthetic_data if not req.get('is_anomaly', False)]
    anomalous_data = [req for req in synthetic_data if req.get('is_anomaly', False)]
    
    print(f"✅ Generated {len(synthetic_data)} total requests")
    print(f"   - Normal requests: {len(normal_data)}")
    print(f"   - Anomalous requests: {len(anomalous_data)}")
    print(f"   - Expected anomaly rate: {len(anomalous_data)/len(synthetic_data):.1%}")
    
    # Train the detector
    print("\n🤖 Training anomaly detection model...")
    detector = AnomalyDetector(contamination=0.12)
    detector.train(normal_data)
    print("✅ Model trained successfully")
    
    # Test detection
    print("\n🔎 Testing anomaly detection...")
    results = detector.predict(synthetic_data)
    
    # Analyze results
    detected_anomalies = sum(results['anomalies'])
    actual_anomalies = len(anomalous_data)
    
    print(f"📈 Detection Results:")
    print(f"   - Actual anomalies: {actual_anomalies}")
    print(f"   - Detected anomalies: {detected_anomalies}")
    print(f"   - Detection rate: {detected_anomalies/len(synthetic_data):.1%}")
    
    # Calculate accuracy metrics
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    for i, is_detected in enumerate(results['anomalies']):
        is_actual = synthetic_data[i].get('is_anomaly', False)
        
        if is_actual and is_detected:
            true_positives += 1
        elif not is_actual and is_detected:
            false_positives += 1
        elif is_actual and not is_detected:
            false_negatives += 1
    
    precision = true_positives / max(detected_anomalies, 1)
    recall = true_positives / max(actual_anomalies, 1)
    f1_score = 2 * (precision * recall) / max(precision + recall, 0.001)
    
    print(f"\n📊 Performance Metrics:")
    print(f"   - Precision: {precision:.3f}")
    print(f"   - Recall: {recall:.3f}")
    print(f"   - F1 Score: {f1_score:.3f}")
    
    # Show example anomalies
    print(f"\n🚨 Example Anomaly Detections:")
    anomaly_count = 0
    for i, (is_anomaly, reasons) in enumerate(zip(results['anomalies'], results['anomaly_reasons'])):
        if is_anomaly and anomaly_count < 3:
            req = synthetic_data[i]
            print(f"\n   Request {i+1}:")
            print(f"   - User: {req['user_id']}")
            print(f"   - Model: {req['model']}")
            print(f"   - Tokens: {req['input_tokens']} → {req['output_tokens']}")
            print(f"   - Score: {results['scores'][i]:.3f}")
            if reasons:
                print(f"   - Reasons: {', '.join(reasons['reasons'])}")
            anomaly_count += 1
    
    # Demo real-time monitoring
    print(f"\n⚡ Real-time Monitoring Demo:")
    monitor = RealTimeMonitor(detector)
    
    # Process a few requests
    for i in range(5):
        req = synthetic_data[i]
        result = monitor.process_request(req)
        
        status = "🚨 ANOMALY" if result['anomalies'][0] else "✅ Normal"
        print(f"   Request {i+1}: {status} (Score: {result['scores'][0]:.3f})")
    
    print(f"\n🎯 Demo completed successfully!")
    print(f"   Target anomaly rate: 12%")
    print(f"   Achieved detection rate: {detected_anomalies/len(synthetic_data):.1%}")
    
    if f1_score > 0.7:
        print("✅ System performance meets expectations!")
    else:
        print("⚠️  System performance could be improved")

if __name__ == "__main__":
    demo_anomaly_detection()
