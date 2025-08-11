import os
from typing import List, Dict

import torch
from pyannote.audio import Pipeline

# Pick device from env or auto-detect
DEVICE = os.getenv("DIAR_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")

_PIPELINE: Pipeline | None = None


def _get_pipeline() -> Pipeline:
    global _PIPELINE
    if _PIPELINE is None:
        token = os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            raise RuntimeError("Missing HUGGINGFACE_TOKEN. Add it to your .env or environment.")
        pipe = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=token,
        )
        pipe.to(torch.device(DEVICE))
        print(f"[diarization] Loaded pyannote on device: {DEVICE}")
        _PIPELINE = pipe
    return _PIPELINE


def diarize(wav_path: str) -> List[Dict]:
    """Return a list of speech turns: [{start, end, speaker}] sorted by start."""
    pipeline = _get_pipeline()
    diar = pipeline(wav_path)
    turns: List[Dict] = []
    for turn, _, speaker in diar.itertracks(yield_label=True):
        turns.append({
            "start": float(turn.start),
            "end": float(turn.end),
            "speaker": str(speaker),
        })
    turns.sort(key=lambda x: x["start"])  # chronological
    return turns


def merge_short_adjacent(turns: List[Dict], min_dur: float = 0.6, max_gap: float = 0.4) -> List[Dict]:
    """
    Merge very short turns and collapse tiny gaps between same-speaker turns.
    - min_dur: ensure each kept turn is at least this long (seconds)
    - max_gap: if consecutive turns by same speaker are separated by <= this gap, merge them
    """
    if not turns:
        return turns
    out: List[Dict] = []
    cur = turns[0].copy()
    for t in turns[1:]:
        same_spk = t["speaker"] == cur["speaker"]
        gap = t["start"] - cur["end"]
        if same_spk and 0 <= gap <= max_gap:
            cur["end"] = t["end"]
            continue
        if (cur["end"] - cur["start"]) < min_dur:
            cur["end"] = max(cur["end"], min(t["start"], cur["start"] + min_dur))
        out.append(cur)
        cur = t.copy()
    if (cur["end"] - cur["start"]) < min_dur and out:
        cur["start"] = min(cur["start"], out[-1]["end"] - min_dur)
    out.append(cur)
    return out


def is_available() -> bool:
    try:
        _ = _get_pipeline()
        return True
    except Exception:
        return False