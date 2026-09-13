"""Audiothing config and env management."""

import os
from pathlib import Path


def load_dotenv(dotenv_path: Path = Path(".env")) -> None:
    """Load .env key-values into os.environ."""
    if dotenv_path.is_file():
        with dotenv_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k and k not in os.environ:
                    os.environ[k] = v


load_dotenv()

DEFAULT_SERVER = os.getenv("RIVA_SERVER", "grpc.nvcf.nvidia.com:443")
DEFAULT_FUNCTION_ID = os.getenv(
    "NVCF_FUNCTION_ID", "ddacc747-1269-4fab-bfd9-8f593dead106"
)
DEFAULT_API_KEY = os.getenv("NVCF_API_KEY", os.getenv("NVIDIA_API_KEY", ""))
DEFAULT_VOICE = os.getenv("RIVA_VOICE", "Chatterbox-Multilingual.en-US.Male")
DEFAULT_LANG = os.getenv("RIVA_LANGUAGE_CODE", "en-US")
DEFAULT_SAMPLE_RATE = 22050
DEFAULT_SENTENCE_PAUSE = 0.25
DEFAULT_PARAGRAPH_PAUSE = 0.50
DEFAULT_MAX_CHARS = 220
DEFAULT_REQUEST_DELAY = 0.5
