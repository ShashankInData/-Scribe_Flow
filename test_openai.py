from openai import OpenAI, __version__
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

print("=== OpenAI SDK Test ===")
print(f"openai sdk version: {__version__}")
print(f"OPENAI_API_KEY set: {'Yes' if os.getenv('OPENAI_API_KEY') else 'No'}")
print(f"OPENAI_BASE_URL: {os.getenv('OPENAI_BASE_URL', 'Not set')}")

# Test client connection
try:
    client = OpenAI()
    models = client.models.list()
    print(f"✅ Connection successful! Found {len(models.data)} models")
    
    # Test transcription endpoint
    print("✅ OpenAI client initialized successfully")
    
except Exception as e:
    print(f"❌ Error: {e}") 