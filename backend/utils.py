import yt_dlp
import ffmpeg
import os
from pathlib import Path
import tempfile
import shutil
import subprocess
from typing import Iterator, Tuple
from pydub import AudioSegment
import io


def setup_directories():
    """
    Create necessary directories if they don't exist
    """
    directories = [
        "data/uploads",
        "data/outputs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def check_ffmpeg():
    """Check if FFmpeg is available"""
    try:
        result = subprocess.run(['ffmpeg', '-version'], 
                              capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return False

def get_audio_duration(audio_path: str) -> float:
    """
    Get audio duration in seconds using ffmpeg
    """
    if not check_ffmpeg():
        print("Warning: FFmpeg not found. Using fallback duration.")
        return 60.0  # Default fallback duration
    
    try:
        probe = ffmpeg.probe(audio_path)
        duration = float(probe['streams'][0]['duration'])
        return duration
    except Exception as e:
        print(f"Warning: Could not get audio duration: {e}")
        return 60.0  # Default fallback duration

def extract_audio(video_path: str, output_path: str = None) -> str:
    """
    Extract audio from video using ffmpeg
    """
    if not check_ffmpeg():
        raise Exception("FFmpeg not found. Please install FFmpeg to process video files.")
    
    try:
        if output_path is None:
            # Create temp file
            temp_dir = tempfile.gettempdir()
            output_path = os.path.join(temp_dir, f"audio_{os.path.basename(video_path)}.wav")
        
        print(f"Extracting audio from: {video_path}")
        print(f"Output path: {output_path}")
        
        # Extract audio
        stream = ffmpeg.input(video_path)
        stream = ffmpeg.output(stream, output_path, acodec='pcm_s16le', ac=1, ar='16000')
        
        print("Running FFmpeg...")
        ffmpeg.run(stream, overwrite_output=True, quiet=True)
        
        # Verify output file exists and has content
        if os.path.exists(output_path):
            file_size = os.path.getsize(output_path)
            print(f"Audio extracted successfully: {output_path} ({file_size} bytes)")
            return output_path
        else:
            raise Exception("FFmpeg output file not found")
        
    except Exception as e:
        print(f"FFmpeg error: {str(e)}")
        raise Exception(f"Audio extraction failed: {str(e)}")

def download_youtube_video(url: str, output_path: str = None) -> str:
    """
    Download YouTube video using yt-dlp
    """
    try:
        if output_path is None:
            output_path = "data/uploads"
        
        # Ensure directory exists
        Path(output_path).mkdir(parents=True, exist_ok=True)
        
        ydl_opts = {
            'outtmpl': f'{output_path}/%(title)s.%(ext)s',
            'format': 'bestaudio[ext=m4a]/bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'm4a',
                'preferredquality': '192',
            }],
            'quiet': True,
        }
        
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            filename = ydl.prepare_filename(info)
            
        return filename
    except Exception as e:
        raise Exception(f"YouTube download failed: {str(e)}")

def is_valid_file(file_path: str) -> bool:
    """
    Check if file is a valid audio/video file
    """
    valid_extensions = {
        '.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv',  # Video
        '.mp3', '.wav', '.m4a', '.flac', '.aac', '.ogg'  # Audio
    }
    
    file_ext = Path(file_path).suffix.lower()
    return file_ext in valid_extensions

def get_file_size(file_path: str) -> str:
    """
    Get human-readable file size
    """
    size_bytes = os.path.getsize(file_path)
    
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    
    return f"{size_bytes:.1f} TB"

def cleanup_temp_files(temp_files: list):
    """
    Clean up temporary files
    """
    for temp_file in temp_files:
        try:
            if os.path.exists(temp_file):
                os.remove(temp_file)
        except Exception as e:
            print(f"Error cleaning up {temp_file}: {e}")

def get_supported_formats() -> dict:
    """
    Get supported file formats
    """
    return {
        "Video": [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv"],
        "Audio": [".mp3", ".wav", ".m4a", ".flac", ".aac", ".ogg"]
    }


def slice_audio(wav_path: str, window_s: float = 30.0, overlap_s: float = 0.3) -> Iterator[Tuple[float, float, bytes]]:
    """Yield (start_s, end_s, wav_bytes) for overlapped slices of a wav file.
    Ensures 16k mono output for each slice.
    """
    audio = AudioSegment.from_wav(wav_path)
    total_ms = len(audio)
    start_ms = 0
    window_ms = int(window_s * 1000)
    overlap_ms = int(overlap_s * 1000)

    while start_ms < total_ms:
        end_ms = min(start_ms + window_ms, total_ms)
        clip = audio[start_ms:end_ms]
        # Export to bytes in proper format
        buf = io.BytesIO()
        buf.name = "slice.wav"
        clip.set_frame_rate(16000).set_channels(1).export(buf, format="wav")
        buf.seek(0)
        yield (start_ms / 1000.0, end_ms / 1000.0, buf.read())
        if end_ms == total_ms:
            break
        start_ms = end_ms - overlap_ms 