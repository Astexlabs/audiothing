"""
Text segmentation and chunking for Triton/Riva model input limits.
"""

import re

from audiothing.config import DEFAULT_MAX_CHARS


def prepare_speech_units(
    text: str, max_chars: int = DEFAULT_MAX_CHARS
) -> list[tuple[str, bool]]:
    """
    Splits input text into manageable chunks respecting sentence and clause boundaries.
    The Triton Chatterbox model has a hard output limit of approximately 500
    speech tokens. A conservative character limit helps avoid truncation even
    though characters and generated speech tokens are not equivalent.

    Returns a list of (chunk_text, is_paragraph_end) tuples so appropriate pause durations
    can be inserted between sentences vs paragraphs.
    """
    raw_paragraphs = [p.strip() for p in text.splitlines() if p.strip()]
    units: list[tuple[str, bool]] = []

    for p in raw_paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", p)
        p_chunks: list[str] = []
        cur = ""
        for s in sentences:
            s = s.strip()
            if not s:
                continue

            if len(s) > max_chars:
                if cur:
                    p_chunks.append(cur)
                    cur = ""
                # Split large sentence on clause punctuation (,;:)
                subparts = re.split(r"(?<=[,;:])\s+", s)
                for part in subparts:
                    part = part.strip()
                    if not part:
                        continue
                    if len(part) > max_chars:
                        for word in part.split():
                            if cur and len(cur) + len(word) + 1 > max_chars:
                                p_chunks.append(cur)
                                cur = word
                            else:
                                cur = f"{cur} {word}" if cur else word
                    else:
                        if cur and len(cur) + len(part) + 1 > max_chars:
                            p_chunks.append(cur)
                            cur = part
                        else:
                            cur = f"{cur} {part}" if cur else part
                if cur:
                    p_chunks.append(cur)
                    cur = ""
            else:
                if cur and len(cur) + len(s) + 1 > max_chars:
                    p_chunks.append(cur)
                    cur = s
                else:
                    cur = f"{cur} {s}" if cur else s
        if cur:
            p_chunks.append(cur)

        # Mark all chunks in this paragraph
        for i, chunk in enumerate(p_chunks):
            chunk = chunk.strip().strip(",;:").strip()
            if not chunk:
                continue
            is_end = i == len(p_chunks) - 1
            units.append((chunk, is_end))

    return units
