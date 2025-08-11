# 🎙 ScribeFlow — Speech to Structured Text

ScribeFlow is a **Streamlit** app that transcribes audio/video, can optionally separate speakers (diarization), and exports results to **SRT, VTT, DOCX, and PDF**.  
It also offers an **AI-generated summary** of the transcript.

---

## ✨ Features

- 🎵 **Multi-format upload**: MP3, WAV, M4A, MP4, MOV, AVI, FLAC, OGG
- 📺 **YouTube URL support** (via `yt-dlp`)
- 🤖 **OpenAI Whisper transcription** (`gpt-4o-mini-transcribe`)
- 🗣️ **Optional speaker diarization** (`pyannote.audio 3.1+`)
- 🏷️ **Speaker rename UI** when multiple speakers are detected
- 📂 **Exports**: SRT, WebVTT, DOCX, PDF
- 📜 **AI summary** (OpenAI GPT)

---

## 📂 Project Structure

Scribe_Flow/
├─ app.py
├─ backend/
│ ├─ asr.py # Transcription + chunking + optional diarization mapping
│ ├─ diarization.py # pyannote.audio 3.1 pipeline wrapper (GPU-aware)
│ ├─ exports.py # SRT/VTT/DOCX/PDF exporters (DOCX/PDF return bytes)
│ ├─ ai_tools.py # AI summary (and optional extensions)
│ └─ utils.py # ffmpeg, yt-dlp, file helpers, duration
├─ data/
│ ├─ uploads/
│ └─ outputs/
├─ .env.example
├─ requirements.txt
└─ README.md

---

## 📋 Requirements

- Python **3.11+**
- **FFmpeg** installed and on PATH
- **OpenAI API key**
- (Optional) Hugging Face access token for diarization

---

## 🚀 Setup

```bash
# Clone the repository
git clone https://github.com/ShashankInData/-Scribe_Flow.git
cd -Scribe_Flow

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

⚙️ Configuration
Create your .env file
# Windows (PowerShell)
copy .env.example .env

# macOS/Linux
cp .env.example .env

Edit .env and set your API keys:
OPENAI_API_KEY=your_openai_key_here
# Optional for diarization:
HUGGINGFACE_TOKEN=your_hf_token_here
# Optional tuning:
ASR_CONCURRENCY=3

▶️ Run the App
streamlit run app.py

In the sidebar:

Paste your OpenAI API key

Toggle "Enable speaker diarization" if your audio has multiple speakers
(A rename UI will appear if ≥2 speakers are detected)


📖 Usage
Upload a local media file or paste a YouTube URL

Click Transcribe

View results:

Transcript (full text)

Timed Segments (with optional speaker rename)

Export in SRT, VTT, DOCX, or PDF

Use the AI Summary feature for a concise recap

📌 Notes on Large Files
Very large uploads are memory-heavy in Streamlit

Prefer YouTube URL for long content

For local large files, consider pre-downloading with yt-dlp into data/uploads/

⚡ GPU & Diarization
Diarization (pyannote.audio 3.1) runs on CPU by default

Switches to GPU if CUDA is available

Install PyTorch with CUDA for acceleration

🛠 Troubleshooting
DOCX/PDF unreadable → Ensure backend/exports.py returns bytes (✅ fixed in this version)

FFmpeg not found → Install FFmpeg and ensure it’s on PATH (ffmpeg -version should work)

OpenAI 404/model errors → Verify model gpt-4o-mini-transcribe and API key

Diarization not working → Check HUGGINGFACE_TOKEN and model terms acceptance on Hugging Face

🗺 Roadmap
AI: Action items & highlights

Chapters for long media

Quotes extraction / post writer / quiz generator

Resumable uploads for very large files

📜 License
MIT License
