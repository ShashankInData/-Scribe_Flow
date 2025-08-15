import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

token = os.getenv("HUGGINGFACE_TOKEN")
assert token, "HUGGINGFACE_TOKEN not set"

print("✅ Token loaded:", token[:10] + "..." + token[-4:])
