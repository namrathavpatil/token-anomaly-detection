"""
Main entry point for the Token Anomaly Detection System
"""

import uvicorn
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.api.monitoring_api import app

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/anomaly_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def create_directories():
    """Create necessary directories"""
    directories = ['logs', 'models', 'data']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)

def main():
    """Main application entry point"""
    logger.info("Starting Token Anomaly Detection System...")
    
    # Create necessary directories
    create_directories()
    
    # Start the FastAPI server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
