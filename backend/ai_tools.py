import openai
import os

class AITools:
    def __init__(self):
        # Guard: if someone set OPENAI_BASE_URL by mistake, ignore it for OpenAI platform
        if os.getenv("OPENAI_BASE_URL"):
            del os.environ["OPENAI_BASE_URL"]
        
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
        
        # Clean client initialization - no base_url, no proxy settings
        self.client = openai.OpenAI(api_key=api_key)
    
    def generate_summary(self, text: str) -> str:
        """
        Generate a concise summary of the transcription
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that creates concise summaries."},
                    {"role": "user", "content": f"Please provide a brief summary of this transcription:\n\n{text[:2000]}..."}
                ],
                max_tokens=200
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Summary generation failed: {str(e)}"
    
    def generate_minutes(self, text: str) -> str:
        """
        Generate meeting minutes from transcription
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional meeting minutes writer."},
                    {"role": "user", "content": f"Please create structured meeting minutes from this transcription:\n\n{text[:2000]}..."}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Minutes generation failed: {str(e)}"
    
    def generate_blog(self, text: str) -> str:
        """
        Generate a blog post from transcription
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative blog writer."},
                    {"role": "user", "content": f"Please create an engaging blog post based on this transcription:\n\n{text[:2000]}..."}
                ],
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Blog generation failed: {str(e)}"
    
    def generate_quiz(self, text: str) -> str:
        """
        Generate quiz questions from transcription
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a creative blog writer."},
                    {"role": "user", "content": f"Please create an engaging blog post based on this transcription:\n\n{text[:2000]}..."}
                ],
                max_tokens=400
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Blog generation failed: {str(e)}"
    
    def generate_email(self, text: str, email_type: str = "summary") -> str:
        """
        Generate email content from transcription
        """
        try:
            prompt = f"Create a {email_type} email based on this transcription"
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a professional email writer."},
                    {"role": "user", "content": f"{prompt}:\n\n{text[:2000]}..."}
                ],
                max_tokens=300
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Email generation failed: {str(e)}"
