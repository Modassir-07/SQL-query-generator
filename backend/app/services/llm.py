import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()

# Google's Gemini API exposes an OpenAI-compatible endpoint, so we can keep using
# the `openai` SDK/chat.completions interface everywhere else in the codebase and
# just point it at Gemini instead of OpenAI.
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"

# Change the model in one place: set GEMINI_MODEL in your .env file.
# Defaults to gemini-3.1-flash-lite for a much higher free-tier quota
# (500 requests/day vs. gemini-3.5-flash's 20/day).
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")

_client = None

def get_llm_client() -> AsyncOpenAI:
    """Returns a configured AsyncOpenAI client pointed at the Gemini API."""
    global _client
    if _client is None:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("Warning: GEMINI_API_KEY not found in environment variables.")
        _client = AsyncOpenAI(api_key=api_key, base_url=GEMINI_BASE_URL)
    return _client


# Backwards-compatible alias in case anything still imports the old name.
get_openai_client = get_llm_client
