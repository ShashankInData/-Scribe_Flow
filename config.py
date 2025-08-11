import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory
BASE_DIR = Path(__file__).parent.parent

# Configuration
class Config:
    # OpenAI
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    # App settings
    DEBUG = os.getenv("DEBUG", "False").lower() == "true"
    MAX_FILE_SIZE = os.getenv("MAX_FILE_SIZE", "100MB")
    
    # File paths
    UPLOAD_DIR = BASE_DIR / "data" / "uploads"
    OUTPUT_DIR = BASE_DIR / "data" / "outputs"
    TEMP_DIR = BASE_DIR / "data" / "temp"
    
    # Create directories if they don't exist
    @classmethod
    def create_directories(cls):
        for directory in [cls.UPLOAD_DIR, cls.OUTPUT_DIR, cls.TEMP_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def validate_config(cls):
        """Validate required configuration"""
        if not cls.OPENAI_API_KEY:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. "
                "Please create a .env file with your OpenAI API key."
            )
        return True

# Initialize directories
Config.create_directories() 