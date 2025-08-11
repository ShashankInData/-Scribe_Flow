import io
import os
import re
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .utils import extract_audio, get_audio_duration, slice_audio

try:
    from .diarization import diarize, merge_short_adjacent, is_available as diarization_available
except Exception:
    def diarization_available() -> bool: return False
    def diarize(_): return []
    def merge_short_adjacent(t, *_, **__): return t


class ASRProcessor:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        self.client = OpenAI(api_key=api_key)

    def transcribe(self, file_path: str, enable_diarization: bool = False) -> dict:
        """
        Transcribe audio/video file. If enable_diarization, run pyannote and label speakers.
        Uses chunked transcription for long files.
        """
        temp_files: List[str] = []
        try:
            # 1) Ensure 16k mono WAV
            audio_path = self._prepare_audio(file_path, temp_files)
            duration = float(get_audio_duration(audio_path))
            print(f"[asr] audio={audio_path} dur={duration:.2f}s")

            # 2) Optional diarization first (on full wav)
            speaker_turns: List[Dict[str, Any]] = []
            diar_meta: Dict[str, Any] = {"requested": enable_diarization, "turns": 0, "speakers": []}
            if enable_diarization and diarization_available():
                try:
                    speaker_turns = diarize(audio_path)
                    speaker_turns = merge_short_adjacent(speaker_turns, min_dur=0.6, max_gap=0.4)
                    diar_meta["turns"] = len(speaker_turns)
                    diar_meta["speakers"] = sorted({t["speaker"] for t in speaker_turns})
                    print(f"[asr] diarization: {diar_meta['turns']} turns {diar_meta['speakers']}")
                except Exception as e:
                    print(f"[asr] diarization failed: {e}")
                    speaker_turns = []
            elif enable_diarization:
                print("[asr] diarization requested but unavailable (missing token/package)")

            # 3) Chunk the audio (20–60s windows) and transcribe in limited parallel
            slices = list(slice_audio(audio_path, window_s=30.0, overlap_s=0.3))
            max_workers = int(os.getenv("ASR_CONCURRENCY", "3"))

            # submit with index to restore order
            ordered_results: List[Dict[str, Any]] = [None] * len(slices)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futs = {}
                for idx, (start_s, end_s, wav_bytes) in enumerate(slices):
                    futs[pool.submit(self._transcribe_bytes, wav_bytes)] = (idx, start_s, end_s)
                for fut in as_completed(futs):
                    idx, start_s, end_s = futs[fut]
                    try:
                        text = fut.result()
                    except Exception as e:
                        print(f"[asr] chunk {start_s:.1f}-{end_s:.1f}s failed: {e}")
                        text = ""
                    ordered_results[idx] = {"start": start_s, "end": end_s, "text": text}

            # 4) Merge in chronological order
            merged_segments = [seg for seg in ordered_results if seg]
            full_text = " ".join(seg["text"] for seg in merged_segments).strip()

            # 5) If diarized, assign speakers by max overlap
            if speaker_turns:
                labeled = []
                for seg in merged_segments:
                    labeled.append({**seg, "speaker": self._assign_speaker(seg, speaker_turns)})
                merged_segments = labeled

            # 6) Fallback single-shot if nothing produced
            if not full_text:
                print("[asr] chunks empty; fallback single-shot")
                with open(audio_path, "rb") as f:
                    text = self.client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=f,
                        response_format="text",
                    )
                full_text = text.strip() if isinstance(text, str) else str(text)
                merged_segments = self._create_sentence_segments(full_text, duration)

            return {
                "text": full_text,
                "segments": merged_segments if merged_segments else self._create_sentence_segments(full_text, duration),
                "meta": {"diarization": diar_meta},
            }

        finally:
            # cleanup temp files we created here
            for p in temp_files:
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass

    # ---- helpers ----
    def _assign_speaker(self, seg: Dict[str, Any], turns: List[Dict[str, Any]]) -> str:
        best = None
        best_overlap = 0.0
        for t in turns:
            overlap = max(0.0, min(seg["end"], t["end"]) - max(seg["start"], t["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best = t
        return best["speaker"] if best else "SPEAKER_00"

    def _transcribe_bytes(self, wav_bytes: bytes) -> str:
        bio = io.BytesIO(wav_bytes)
        bio.name = "chunk.wav"
        out = self.client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=bio,
            response_format="text",
        )
        return out.strip() if isinstance(out, str) else str(out)

    def _prepare_audio(self, file_path: str, temp_files: List[str]) -> str:
        # If already WAV, keep, else extract via ffmpeg
        if file_path.lower().endswith(".wav"):
            return file_path
        wav_path = extract_audio(file_path)
        temp_files.append(wav_path)
        return wav_path

    def _create_sentence_segments(self, text: str, duration: float) -> List[Dict[str, Any]]:
        sentences = self._split_sentences(text)
        if not sentences:
            return [{"start": 0.0, "end": duration, "text": text}]
        segs = []
        tps = duration / max(1, len(sentences))
        for i, s in enumerate(sentences):
            start = i * tps
            end = (i + 1) * tps
            segs.append({"start": start, "end": end, "text": s.strip()})
        segs[-1]["end"] = duration
        return segs

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p for p in parts if p]