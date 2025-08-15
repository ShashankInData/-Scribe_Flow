import os
import io
import tempfile
import subprocess
from pathlib import Path
from typing import Iterator, Tuple, Optional

import yt_dlp
import ffmpeg
from pydub import AudioSegment


def setup_directories() -> None:
    for directory in ["data/uploads", "data/outputs"]:
        Path(directory).mkdir(parents=True, exist_ok=True)


def check_ffmpeg() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False


def get_audio_duration(path: str) -> float:
    if not check_ffmpeg():
        return 60.0
    try:
        probe = ffmpeg.probe(path)
        dur: Optional[float] = None
        for s in probe.get("streams", []):
            if "duration" in s:
                dur = float(s["duration"])
                break
        if dur is None:
            fmt = probe.get("format", {})
            if "duration" in fmt:
                dur = float(fmt["duration"])
        return float(dur) if dur is not None else 60.0
    except Exception:
        return 60.0


def extract_audio(input_path: str, sr: int = 16000, mono: bool = True) -> str:
    if not check_ffmpeg():
        raise RuntimeError("FFmpeg not found. Please install FFmpeg to process media files.")

    input_path = str(Path(input_path))
    tmp_wav = Path(tempfile.gettempdir()) / f"audio_{next(tempfile._get_candidate_names())}.wav"

    try:
        (
            ffmpeg
            .input(input_path)
            .output(str(tmp_wav), ac=1 if mono else 2, ar=sr, vn=None, f="wav")
            .global_args("-y", "-hide_banner", "-loglevel", "error")
            .run(capture_stdout=True, capture_stderr=True)
        )
        if not tmp_wav.exists() or tmp_wav.stat().st_size == 0:
            raise RuntimeError("FFmpeg produced empty output.")
        return str(tmp_wav)
    except Exception:
        # pydub fallback
        audio = AudioSegment.from_file(input_path)
        if mono:
            audio = audio.set_channels(1)
        audio = audio.set_frame_rate(sr)
        audio.export(str(tmp_wav), format="wav")
        if not tmp_wav.exists() or tmp_wav.stat().st_size == 0:
            raise RuntimeError("pydub produced empty output.")
        return str(tmp_wav)


def download_youtube_video(url: str, output_dir: str = "data/uploads") -> str:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "noplaylist": True,
        "quiet": True,
        "retries": 10,
        "ignoreerrors": True,
        "restrictfilenames": True,
        "outtmpl": str(out_dir / "%(id)s.%(ext)s"),  # safe ID-only filename
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        if not info:
            raise RuntimeError("yt-dlp failed to download the media.")
        vid = info.get("id")
        ext = info.get("ext", "m4a")
        candidate = out_dir / f"{vid}.{ext}"
        if candidate.exists():
            return str(candidate)
        matches = list(out_dir.glob(f"{vid}.*"))
        if not matches:
            raise RuntimeError("Downloaded file not found after yt-dlp run.")
        return str(matches[0])


def is_valid_file(file_path: str) -> bool:
    return Path(file_path).suffix.lower() in {
        ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv",
        ".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg", ".opus",
    }


def get_file_size(file_path: str) -> str:
    size_bytes = os.path.getsize(file_path)
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} TB"


def slice_audio(wav_path: str, window_s: float = 30.0, overlap_s: float = 0.3) -> Iterator[Tuple[float, float, bytes]]:
    audio = AudioSegment.from_wav(wav_path)
    total_ms = len(audio)
    start_ms = 0
    window_ms = int(window_s * 1000)
    overlap_ms = int(overlap_s * 1000)

    while start_ms < total_ms:
        end_ms = min(start_ms + window_ms, total_ms)
        clip = audio[start_ms:end_ms]
        buf = io.BytesIO()
        buf.name = "slice.wav"
        clip.set_frame_rate(16000).set_channels(1).export(buf, format="wav")
        buf.seek(0)
        yield (start_ms / 1000.0, end_ms / 1000.0, buf.read())
        if end_ms == total_ms:
            break
        start_ms = end_ms - overlap_ms
