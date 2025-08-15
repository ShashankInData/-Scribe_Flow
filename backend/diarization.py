import os
from pathlib import Path
from typing import List, Dict, Optional

import torch

# Load .env explicitly (helps when Streamlit launches from a different CWD)
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=Path(__file__).resolve().parents[1] / ".env")
except Exception:
    pass

try:
    from pyannote.audio import Pipeline
except Exception as e:
    Pipeline = None  # handled in is_available()


_PIPELINE = None


def _get_token() -> Optional[str]:
    """
    Try multiple env names and Streamlit secrets for the HF token.
    """
    token = (
        os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HF_ACCESS_TOKEN")
    )
    # Streamlit secrets fallback (if available)
    try:
        import streamlit as st  # type: ignore
        token = token or st.secrets.get("HUGGINGFACE_TOKEN", None)
    except Exception:
        pass
    return token


def _get_pipeline():
    """
    Lazily initialize the pyannote diarization pipeline.
    """
    global _PIPELINE
    if _PIPELINE is not None:
        return _PIPELINE

    if Pipeline is None:
        raise RuntimeError("pyannote.audio is not installed. pip install 'pyannote.audio>=3.1'")

    token = _get_token()
    if not token:
        raise RuntimeError(
            "Missing HUGGINGFACE_TOKEN. Put it in .env (same folder as app.py) or Streamlit secrets."
        )

    pipe = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=token,
    )

    if torch.cuda.is_available():
        try:
            pipe.to(torch.device("cuda"))
        except Exception:
            # Fall back to CPU if device move fails
            pass

    _PIPELINE = pipe
    return _PIPELINE


def diarize(wav_path: str) -> List[Dict]:
    """
    Run diarization on a WAV path. Returns list of turns:
    [{ 'start': float, 'end': float, 'speaker': 'SPEAKER_00' }, ...]
    """
    pipeline = _get_pipeline()
    diar = pipeline(wav_path)

    turns: List[Dict] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })

    turns.sort(key=lambda x: x["start"])
    return turns


def is_available() -> bool:
    """
    Quick availability check. Prints reason on failure.
    """
    try:
        _ = _get_pipeline()
        return True
    except Exception as e:
        print(f"[diarization] not available: {e}")
        return False
