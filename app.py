from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
import streamlit as st

from backend.asr import ASRProcessor
from backend.exports import ExportManager
from backend.ai_tools import AITools
from backend.utils import setup_directories


# ------------------------------
# Streamlit page configuration
# ------------------------------
st.set_page_config(
    page_title="ScribeFlow — Speech to Structured Text",
    layout="wide",
)


def _ensure_state():
    if "transcription_result" not in st.session_state:
        st.session_state.transcription_result = None
    if "show_results" not in st.session_state:
        st.session_state.show_results = False
    if "speaker_map" not in st.session_state:
        st.session_state.speaker_map = {}


def main():
    st.title("ScribeFlow — Speech to Structured Text")
    st.write("Upload audio/video for accurate transcription, optional speaker separation, and AI-powered analysis.")

    # Init folders and backends
    setup_directories()
    _ensure_state()

    asr_processor = ASRProcessor()
    export_manager = ExportManager()
    ai_tools = AITools()

    # ------------------------------
    # Sidebar: configuration
    # ------------------------------
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            help="Used only in this session."
        )
        enable_diar = st.checkbox(
            "Enable speaker diarization",
            value=False,
            help="Identifies who spoke when. Enable for meetings, interviews, or multi-speaker audio."
        )

        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.success("API Key configured")
        else:
            st.warning("Please enter your OpenAI API Key")
            st.stop()

    # ------------------------------
    # Inputs
    # ------------------------------
    st.header("Input")

    youtube_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    uploaded_file = st.file_uploader(
        "Upload audio/video file",
        type=["mp3", "mp4", "wav", "m4a", "avi", "mov", "flac", "aac", "ogg"],
        help="Supported formats: MP3, MP4, WAV, M4A, AVI, MOV, FLAC, AAC, OGG"
    )

    file_path = None
    if youtube_url or uploaded_file is not None:
        if youtube_url:
            from backend.utils import download_youtube_video
            try:
                with st.spinner("Downloading YouTube video..."):
                    file_path = download_youtube_video(youtube_url)
                st.success(f"Downloaded: {os.path.basename(file_path)}")
            except Exception as e:
                st.error(f"YouTube download failed: {e}")
                st.stop()
        else:
            upload_dir = Path("data/uploads")
            upload_dir.mkdir(parents=True, exist_ok=True)
            file_path = str(upload_dir / uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"Uploaded: {uploaded_file.name}")

        # ------------------------------
        # Transcribe
        # ------------------------------
        if st.button("Transcribe", type="primary"):
            try:
                with st.spinner("Transcribing..."):
                    result = asr_processor.transcribe(file_path, enable_diarization=enable_diar)

                st.session_state.transcription_result = result
                st.session_state.show_results = True

                if enable_diar:
                    speakers_meta = (
                        result.get("meta", {})
                        .get("diarization", {})
                        .get("speakers", [])
                    )
                    if speakers_meta:
                        st.info("Detected speakers: " + ", ".join(speakers_meta))

                st.success("Transcription complete")
            except Exception as e:
                st.error(f"Transcription failed: {e}")

    # ------------------------------
    # Results
    # ------------------------------
    if st.session_state.show_results and st.session_state.transcription_result:
        result = st.session_state.transcription_result

        st.header("Transcription Results")
        st.subheader("Full Transcript")
        st.text_area("Transcript", result.get("text", ""), height=200)

        # Timed Segments & Speaker Rename
        segments = result.get("segments", [])
        if segments:
            st.subheader("Timed Segments")

            speakers = sorted({seg.get("speaker") for seg in segments if seg.get("speaker")})
            if enable_diar and len(speakers) > 1:
                st.markdown("Rename speakers:")
                if not st.session_state.speaker_map:
                    st.session_state.speaker_map = {s: s for s in speakers}
                cols = st.columns(min(4, len(speakers)))
                for idx, spk in enumerate(speakers):
                    with cols[idx % len(cols)]:
                        new_label = st.text_input(
                            f"{spk}",
                            value=st.session_state.speaker_map.get(spk, spk),
                            key=f"spk_{spk}"
                        )
                        st.session_state.speaker_map[spk] = new_label
                for seg in segments:
                    spk = seg.get("speaker")
                    if spk in st.session_state.speaker_map:
                        seg["speaker"] = st.session_state.speaker_map[spk]

            for seg in segments:
                start_time = f"{seg['start']:.1f}s"
                end_time = f"{seg['end']:.1f}s"
                label = f"{seg.get('speaker')}: " if seg.get("speaker") else ""
                st.write(f"[{start_time} - {end_time}] {label}{seg.get('text','')}")

        # Export Options
        st.header("Export Options")
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            try:
                srt_content = export_manager.to_srt(result)
                st.download_button("Download SRT", srt_content, file_name="transcript.srt", mime="text/plain")
            except Exception as e:
                st.error(f"SRT export failed: {e}")

        with col2:
            try:
                vtt_content = export_manager.to_vtt(result)
                st.download_button("Download VTT", vtt_content, file_name="transcript.vtt", mime="text/plain")
            except Exception as e:
                st.error(f"VTT export failed: {e}")

        with col3:
            try:
                docx_bytes = export_manager.to_docx(result)
                st.download_button(
                    "Download DOCX", docx_bytes, file_name="transcript.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            except Exception as e:
                st.error(f"DOCX export failed: {e}")

        with col4:
            try:
                pdf_bytes = export_manager.to_pdf(result)
                st.download_button("Download PDF", pdf_bytes, file_name="transcript.pdf", mime="application/pdf")
            except Exception as e:
                st.error(f"PDF export failed: {e}")

        # AI Tools
        st.header("AI-Powered Analysis")
        if st.button("Generate Summary"):
            try:
                with st.spinner("Generating summary..."):
                    summary = ai_tools.generate_summary(result.get("text", ""))
                st.text_area("Summary", summary, height=150)
            except Exception as e:
                st.error(f"Summary generation failed: {e}")


if __name__ == "__main__":
    main()
