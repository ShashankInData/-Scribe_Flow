import os
import re
import io
import tempfile
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from .utils import extract_audio, get_audio_duration, slice_audio

# Diarization imports with safe fallback
try:
    from .diarization import diarize, is_available as diarization_available
except Exception:
    def diarization_available() -> bool:
        return False
    def diarize(_):
        return []


class ASRProcessor:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY in your environment/.env.")
        self.client = OpenAI(api_key=api_key)
        self.temp_files: List[str] = []

    def transcribe(self, file_path: str, enable_diarization: bool = False) -> dict:
        """
        Transcribe audio/video file. If enable_diarization, run pyannote and map speakers to segments.
        Uses chunked transcription for long files.
        """
        try:
            audio_path = self._prepare_audio(file_path)
            duration = get_audio_duration(audio_path)
            print(f"[asr] audio={audio_path} dur={duration:.2f}s")

            # Optional diarization first (on full wav)
            speaker_turns: List[Dict[str, Any]] = []
            diar_meta: Dict[str, Any] = {"requested": enable_diarization, "turns": 0, "speakers": []}
            if enable_diarization:
                if diarization_available():
                    try:
                        speaker_turns = diarize(audio_path)
                        diar_meta["turns"] = len(speaker_turns)
                        diar_meta["speakers"] = sorted(list({t["speaker"] for t in speaker_turns}))
                        print(f"[asr] diarization turns={diar_meta['turns']} speakers={diar_meta['speakers']}")
                    except Exception as e:
                        print(f"[asr] diarization failed: {e}")
                        speaker_turns = []
                else:
                    print("[asr] diarization requested but unavailable (missing token/package)")

            # Prepare 30s overlapping slices
            slices = list(slice_audio(audio_path, window_s=30.0, overlap_s=0.3))

            aggregated_text_parts: List[str] = []
            merged_segments: List[Dict[str, Any]] = []

            max_workers = int(os.getenv("ASR_CONCURRENCY", "3"))
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {
                    pool.submit(self._transcribe_bytes, wav_bytes): (start_s, end_s)
                    for start_s, end_s, wav_bytes in slices
                }
                for fut in as_completed(futures):
                    start_s, end_s = futures[fut]
                    try:
                        text = fut.result()
                        aggregated_text_parts.append(text)
                        merged_segments.append({"start": start_s, "end": end_s, "text": text})
                    except Exception as e:
                        print(f"[asr] chunk {start_s:.1f}-{end_s:.1f}s failed: {e}")

            merged_segments.sort(key=lambda s: s["start"])
            full_text = " ".join(t for t in aggregated_text_parts).strip()

            # Map segments to speaker turns (max overlap)
            if speaker_turns and merged_segments:
                labeled = []
                for seg in merged_segments:
                    labeled.append({**seg, "speaker": self._assign_speaker(seg, speaker_turns)})
                merged_segments = labeled

            # Fallback single-shot if chunk pass produced nothing
            if not full_text:
                with open(audio_path, "rb") as audio_file:
                    print("[asr] single-shot fallback → OpenAI")
                    response = self.client.audio.transcriptions.create(
                        model="gpt-4o-mini-transcribe",
                        file=audio_file,
                        response_format="text",
                    )
                full_text = str(response).strip()
                merged_segments = self._create_sentence_segments(full_text, duration)

            return {
                "text": full_text,
                "segments": merged_segments if merged_segments else self._create_sentence_segments(full_text, duration),
                "meta": {"diarization": diar_meta},
            }

        except Exception as e:
            import traceback
            traceback.print_exc()
            raise Exception(f"Transcription failed: {e}") from e
        finally:
            self._cleanup_temp_files()

    def _transcribe_bytes(self, wav_bytes: bytes) -> str:
        bio = io.BytesIO(wav_bytes)
        bio.name = "chunk.wav"
        out = self.client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=bio,
            response_format="text",
        )
        return str(out).strip()

    def _assign_speaker(self, seg: Dict[str, Any], turns: List[Dict[str, Any]]) -> str:
        best = None
        best_overlap = 0.0
        for t in turns:
            overlap = max(0.0, min(seg["end"], t["end"]) - max(seg["start"], t["start"]))
            if overlap > best_overlap:
                best_overlap = overlap
                best = t
        return best["speaker"] if best else "SPEAKER_00"

    def _prepare_audio(self, file_path: str) -> str:
        if file_path.lower().endswith(".wav"):
            return file_path
        wav_path = extract_audio(file_path)
        self.temp_files.append(wav_path)
        return wav_path

    def _create_sentence_segments(self, text: str, duration: float) -> List[Dict[str, Any]]:
        sentences = self._split_sentences(text)
        if not sentences:
            return [{"start": 0.0, "end": duration, "text": text}]
        segs: List[Dict[str, Any]] = []
        tps = max(duration / max(1, len(sentences)), 0.01)
        for i, s in enumerate(sentences):
            segs.append({"start": float(i * tps), "end": float(min(duration, (i + 1) * tps)), "text": s.strip()})
        if segs:
            segs[-1]["end"] = float(duration)
        return segs

    def _split_sentences(self, text: str) -> List[str]:
        parts = re.split(r"(?<=[.!?])\s+", text.strip())
        return [p.strip() for p in parts if p.strip()]

    def _cleanup_temp_files(self):
        for p in self.temp_files:
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception:
                pass
        self.temp_files.clear()
