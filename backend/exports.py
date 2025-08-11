import srt
from datetime import timedelta
from docx import Document
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
import io

class ExportManager:
    def __init__(self):
        self.styles = getSampleStyleSheet()
    
    def to_srt(self, transcription_data, filename: str = "transcript") -> str:
        """Convert transcription with segments to SRT subtitle format."""
        try:
            if isinstance(transcription_data, dict) and "segments" in transcription_data:
                segments = transcription_data["segments"]
                subs = []
                for i, segment in enumerate(segments):
                    speaker = segment.get("speaker", "")
                    text = segment.get("text", "").strip()
                    if speaker:
                        text = f"{speaker}: {text}"
                    sub = srt.Subtitle(
                        index=i + 1,
                        start=timedelta(seconds=segment["start"]),
                        end=timedelta(seconds=segment["end"]),
                        content=text
                    )
                    subs.append(sub)
                return srt.compose(subs)
            else:
                return self._text_to_srt_fallback(transcription_data)
        except Exception as e:
            return f"Error creating SRT: {str(e)}"
    
    def to_vtt(self, transcription_data, filename: str = "transcript") -> str:
        """Convert transcription with segments to WebVTT format."""
        try:
            if isinstance(transcription_data, dict) and "segments" in transcription_data:
                segments = transcription_data["segments"]
                vtt_content = "WEBVTT\n\n"
                for segment in segments:
                    start_str = f"{int(segment['start']//60):02d}:{segment['start']%60:05.2f}"
                    end_str = f"{int(segment['end']//60):02d}:{segment['end']%60:05.2f}"
                    speaker = segment.get("speaker", "")
                    text = segment.get("text", "").strip()
                    if speaker:
                        text = f"{speaker}: {text}"
                    vtt_content += f"{start_str} --> {end_str}\n{text}\n\n"
                return vtt_content
            else:
                return self._text_to_vtt_fallback(transcription_data)
        except Exception as e:
            return f"Error creating VTT: {str(e)}"
    
    def to_docx(self, transcription_data, filename: str = "transcript") -> bytes:
        """Convert transcription data to DOCX bytes."""
        try:
            if isinstance(transcription_data, dict) and "segments" in transcription_data:
                content_parts = []
                for seg in transcription_data["segments"]:
                    speaker = seg.get("speaker", "")
                    text_content = seg.get("text", "").strip()
                    if speaker:
                        content_parts.append(f"{speaker}: {text_content}")
                    else:
                        content_parts.append(text_content)
                content = "\n\n".join(content_parts)
            else:
                content = str(transcription_data)
            
            doc = Document()
            doc.add_heading('Transcription', 0)
            doc.add_paragraph(content)
            
            buffer = io.BytesIO()
            doc.save(buffer)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            return f"Error creating DOCX: {str(e)}".encode()
    
    def to_pdf(self, transcription_data, filename: str = "transcript") -> bytes:
        """Convert transcription data to PDF bytes."""
        try:
            if isinstance(transcription_data, dict) and "segments" in transcription_data:
                content_parts = []
                for seg in transcription_data["segments"]:
                    speaker = seg.get("speaker", "")
                    text_content = seg.get("text", "").strip()
                    if speaker:
                        content_parts.append(f"{speaker}: {text_content}")
                    else:
                        content_parts.append(text_content)
                content = "\n\n".join(content_parts)
            else:
                content = str(transcription_data)
            
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer, pagesize=letter)
            story = []
            story.append(Paragraph("Transcription", self.styles['Title']))
            story.append(Spacer(1, 12))
            story.append(Paragraph(content, self.styles['Normal']))
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue()
        except Exception as e:
            return f"Error creating PDF: {str(e)}".encode()
    
    def _text_to_srt_fallback(self, text: str) -> str:
        """Fallback SRT creation for plain text."""
        sentences = text.split('. ')
        subs = []
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                start_time = timedelta(seconds=i * 3)
                end_time = timedelta(seconds=(i + 1) * 3)
                sub = srt.Subtitle(
                    index=i + 1,
                    start=start_time,
                    end=end_time,
                    content=sentence.strip()
                )
                subs.append(sub)
        return srt.compose(subs)
    
    def _text_to_vtt_fallback(self, text: str) -> str:
        """Fallback VTT creation for plain text."""
        sentences = text.split('. ')
        vtt_content = "WEBVTT\n\n"
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                start_time = i * 3
                end_time = (i + 1) * 3
                start_str = f"{start_time//60:02d}:{start_time%60:02d}.000"
                end_str = f"{end_time//60:02d}:{end_time%60:02d}.000"
                vtt_content += f"{start_str} --> {end_str}\n{sentence.strip()}\n\n"
        return vtt_content
