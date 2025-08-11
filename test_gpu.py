import os
import torch
from pyannote.audio import Pipeline

print("CUDA available:", torch.cuda.is_available())

# Load diarization pipeline from Hugging Face
pipe = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=os.getenv("HUGGINGFACE_TOKEN")
)

# Move to GPU if available
if torch.cuda.is_available():
    pipe.to(torch.device("cuda"))
    print("Pipeline device set to: cuda")
else:
    print("Pipeline device set to: cpu")