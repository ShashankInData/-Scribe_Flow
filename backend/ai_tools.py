import os
from typing import Optional
from openai import OpenAI
from openai._exceptions import APIConnectionError, RateLimitError, APIStatusError


class AITools:
    def __init__(self):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")

        # Allow swapping the model via env without code changes
        # Defaults to gpt-4o-mini which is fast & inexpensive
        self.model = os.getenv("AI_TOOLS_MODEL", "gpt-4o-mini")

        # Make the client a bit more robust
        # - timeout (seconds): avoid hanging forever
        # - max_retries: simple resilience to transient errors
        self.client = OpenAI(
            api_key=api_key,
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
            timeout=float(os.getenv("OPENAI_TIMEOUT", "30")),
        )

        # Defensive: if a stray base URL is set in the environment, unset it
        # (only do this if you are not intentionally using a proxy/gateway)
        if os.getenv("OPENAI_BASE_URL"):
            del os.environ["OPENAI_BASE_URL"]

    def _safe_text(self, text: str, limit: int = 6000) -> str:
        # Basic guard so we don't throw on None/short content
        text = (text or "").strip()
        if not text:
            return "No content provided."
        # Simple character clamp to keep prompts small and predictable
        return text[:limit]

    def _chat(self, system: str, user: str, max_tokens: int = 400, temperature: float = 0.3) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            return resp.choices[0].message.content
        except RateLimitError as e:
            return f"Summary generation failed: rate limit — {e}"
        except APIConnectionError as e:
            return f"Summary generation failed: network — {e}"
        except APIStatusError as e:
            return f"Summary generation failed: API {e.status_code} — {e.message}"
        except Exception as e:
            return f"Summary generation failed: {e}"

    def generate_summary(self, text: str) -> str:
        content = self._safe_text(text)
        return self._chat(
            "You create concise, factual summaries.",
            f"Summarize the following content in ~5–7 bullet points:\n\n{content}",
            max_tokens=300,
            temperature=0.2,
        )

    def generate_quiz(self, text: str) -> str:
        content = self._safe_text(text)
        try:
            return self._chat(
                "You write short quizzes with answers.",
                (
                    "Create 5 short Q&A pairs from this content. "
                    "Format exactly as:\n"
                    "Q1: ...\nA1: ...\n"
                    "Q2: ...\nA2: ...\n"
                    "Q3: ...\nA3: ...\n"
                    "Q4: ...\nA4: ...\n"
                    "Q5: ...\nA5: ...\n\n"
                    f"Content:\n{content}"
                ),
                max_tokens=400,
                temperature=0.5,
            )
        except Exception as e:
            return f"Quiz generation failed: {e}"

    def generate_email(self, text: str, email_type: Optional[str] = "summary") -> str:
        content = self._safe_text(text)
        try:
            style_hint = "summary" if (email_type or "summary") == "summary" else "general"
            return self._chat(
                "You write professional emails.",
                (
                    f"Draft a short, professional email ({style_hint} style) summarizing the key points below. "
                    "Keep it clear and actionable, with a subject line.\n\n"
                    f"{content}"
                ),
                max_tokens=300,
                temperature=0.4,
            )
        except Exception as e:
            return f"Email generation failed: {e}"
