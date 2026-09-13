"""Modular toolkit for Text-to-Speech synthesis."""

from audiothing.chunker import prepare_speech_units
from audiothing.config import (
    DEFAULT_API_KEY,
    DEFAULT_FUNCTION_ID,
    DEFAULT_LANG,
    DEFAULT_MAX_CHARS,
    DEFAULT_PARAGRAPH_PAUSE,
    DEFAULT_REQUEST_DELAY,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_SENTENCE_PAUSE,
    DEFAULT_SERVER,
    DEFAULT_VOICE,
    load_dotenv,
)
from audiothing.sanitizer import sanitize_text
from audiothing.initialisms import (
    pronounce_initialisms,
    pronounce_initialisms_phonetic,
)
from audiothing.storage import get_next_audio_filename, get_next_comparison_filenames
from audiothing.synthesizer import synthesize_chunk_with_retry, synthesize_speech

__all__ = [
    "load_dotenv",
    "get_next_audio_filename",
    "get_next_comparison_filenames",
    "sanitize_text",
    "pronounce_initialisms",
    "pronounce_initialisms_phonetic",
    "prepare_speech_units",
    "synthesize_chunk_with_retry",
    "synthesize_speech",
    "DEFAULT_SERVER",
    "DEFAULT_FUNCTION_ID",
    "DEFAULT_API_KEY",
    "DEFAULT_VOICE",
    "DEFAULT_LANG",
    "DEFAULT_MAX_CHARS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_SENTENCE_PAUSE",
    "DEFAULT_PARAGRAPH_PAUSE",
    "DEFAULT_REQUEST_DELAY",
]
