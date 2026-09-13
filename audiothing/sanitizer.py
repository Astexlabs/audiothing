"""
Text sanitization and normalization for speech synthesis.
"""

import re


def sanitize_text(text: str) -> str:
    """
    Sanitizes input text for speech synthesis so TTS models (like Chatterbox)
    do not literally vocalize punctuation, code formatting, or markdown syntax.
    - Strips double quotes (", “”), backticks (`), and single quotes around words
    - Preserves English contractions (don't, it's, let's, we've)
    - Replaces hyphens, em-dashes, and underscores with natural pauses or spaces
    - Expands numeric ranges (e.g. 350-450 -> 350 to 450) and symbols (& -> and)
    - Strips markdown formatting (headers, bullet points, asterisks, brackets)
    """
    # 1. Normalize line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # 2. Markdown headers (e.g. "# Title", "### Section")
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)

    # 3. Markdown horizontal rules
    text = re.sub(r"^[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)

    # 4. Markdown links [text](url) -> text and images ![alt](url) -> alt
    text = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    # 5. Fenced code blocks and inline code backticks
    text = re.sub(r"```[\w-]*\n([\s\S]*?)```", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    text = text.replace("`", "").replace("´", "")

    # 6. HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 7. Markdown bullet lists at start of lines (*, -, +)
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)

    # 8. Numbered lists: convert "1)" to "1." to avoid reading "one right parenthesis"
    text = re.sub(r"^\s*(\d+)\)\s+", r"\1. ", text, flags=re.MULTILINE)
    text = re.sub(r"(?<=\n)(\d+)\)\s+", r"\1. ", text)

    # 9. Asterisks (markdown bold/italics: ***, **, *)
    text = re.sub(r"\*+", "", text)

    # 10. Tildes (strikethrough or approximation)
    text = re.sub(r"~\s*(\d+)", r"about \1", text)
    text = text.replace("~", "")

    # 11. Slashes: replace word/word with "word, word" (e.g. hardware/software -> hardware, software)
    text = re.sub(r"(?<=[a-zA-Z0-9])\s*/\s*(?=[a-zA-Z0-9])", ", ", text)
    text = text.replace("/", " ").replace("\\", " ")

    # 12. Underscores (snake_case, dunder, or markdown)
    text = re.sub(r"_+", " ", text)

    # 13. Numeric ranges: 350-450 -> 350 to 450
    text = re.sub(r"(\d+)\s*[-–—]\s*(\d+)", r"\1 to \2", text)

    # 14. Dashes, hyphens, and em/en dashes
    # Em-dashes, en-dashes, and multi-hyphens -> natural pause (comma)
    text = re.sub(r"\s*[-–—]{2,}\s*", ", ", text)
    text = re.sub(r"\s+[-–—]\s+", ", ", text)
    text = re.sub(r"[–—]", ", ", text)

    # Hyphenated compound words: non-trivial -> non trivial, anti-virus -> anti virus
    text = re.sub(r"(?<=[a-zA-Z0-9])-(?=[a-zA-Z0-9])", " ", text)
    # Remaining hyphens
    text = text.replace("-", " ")

    # 15. Quotes: double quotes (straight and curly) -> remove completely
    text = re.sub(r"[\"“”«»„‟]", "", text)

    # 16. Single quotes and apostrophes:
    # Preserve apostrophes inside contractions / possessives: don't, it's, let's, O'Connor
    text = re.sub(r"(?<=[a-zA-Z0-9])[’\x27](?=[a-zA-Z])", "§APOS§", text)
    # Remove all quotation single quotes / curly quotes
    text = re.sub(r"[\x27‘’]", "", text)
    # Restore internal apostrophes
    text = text.replace("§APOS§", "'")

    # 17. Parentheses, brackets, and braces
    # Convert parenthetical thoughts to natural comma-delimited clauses
    text = re.sub(r"\(([^)]+)\)", r", \1, ", text)
    text = re.sub(r"[()\[\]{}]", " ", text)

    # 18. Common symbols
    text = re.sub(r"#(\d+)", r"number \1", text)
    text = text.replace("#", " ")
    text = re.sub(r"\s*&\s*", " and ", text)
    text = re.sub(r"(\d+)\s*%", r"\1 percent", text)
    text = text.replace("%", " percent ")
    text = re.sub(r"\s*\+\s*", " plus ", text)
    text = re.sub(r"\s*=\s*", " equals ", text)
    text = re.sub(r"[@^|<>]+", " ", text)

    # 19. Clean up whitespace and punctuation spacing
    text = re.sub(r"\s+([,.:;?!])", r"\1", text)
    text = re.sub(r",(\s*,)+", ",", text)
    text = re.sub(r",(\s*\.)+", ".", text)
    text = re.sub(r"\.(\s*,)+", ".", text)
    text = re.sub(r"^\s*,\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"[ \t]+", " ", text)
    text = "\n".join(line.strip() for line in text.splitlines())
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return text
