"""Pronounce short uppercase initialisms as letter names."""

from __future__ import annotations

import re
from collections.abc import Callable

_COMMON_UPPERCASE_WORDS = frozenset(
    {
        "ALL",
        "AND",
        "ANY",
        "ARE",
        "AS",
        "AT",
        "BE",
        "BUT",
        "BY",
        "CAN",
        "DO",
        "FOR",
        "GET",
        "GO",
        "HAD",
        "HAS",
        "HE",
        "HER",
        "HIM",
        "HOW",
        "IF",
        "IN",
        "IS",
        "IT",
        "LET",
        "LOW",
        "MAY",
        "ME",
        "MY",
        "NEW",
        "NO",
        "NOT",
        "NOW",
        "OF",
        "OK",
        "OLD",
        "ON",
        "ONE",
        "OR",
        "OUR",
        "OUT",
        "PER",
        "PUT",
        "RED",
        "RUN",
        "SAY",
        "SEE",
        "SHE",
        "SO",
        "THE",
        "TO",
        "TOO",
        "TWO",
        "UP",
        "US",
        "WAY",
        "WE",
        "WHO",
        "WHY",
        "YES",
        "YOU",
    }
)

_KNOWN_INITIALISMS = frozenset(
    {
        "AI",
        "API",
        "AWS",
        "CD",
        "CEO",
        "CI",
        "CLI",
        "CPU",
        "CSV",
        "CTO",
        "DB",
        "DNS",
        "GPU",
        "GPS",
        "HR",
        "ID",
        "IP",
        "IT",
        "JWT",
        "ML",
        "OS",
        "PDF",
        "PM",
        "PR",
        "QA",
        "SDK",
        "SQL",
        "SSH",
        "TCP",
        "UDP",
        "UI",
        "UN",
        "URL",
        "USA",
        "USB",
        "UX",
        "XML",
    }
)

_LETTER_NAMES = {
    "A": "ay",
    "B": "bee",
    "C": "see",
    "D": "dee",
    "E": "ee",
    "F": "eff",
    "G": "gee",
    "H": "aitch",
    "I": "eye",
    "J": "jay",
    "K": "kay",
    "L": "el",
    "M": "em",
    "N": "en",
    "O": "oh",
    "P": "pee",
    "Q": "cue",
    "R": "ar",
    "S": "ess",
    "T": "tee",
    "U": "you",
    "V": "vee",
    "W": "double you",
    "X": "ex",
    "Y": "why",
    "Z": "zee",
}

# IPA-style spellings are plain Unicode text, without slash delimiters. Slashes
# can be spoken or sanitized as punctuation instead of acting as hints.
_PHONETIC_NAMES = {
    "A": "eɪ",
    "B": "biː",
    "C": "siː",
    "D": "diː",
    "E": "iː",
    "F": "ɛf",
    "G": "dʒiː",
    "H": "eɪtʃ",
    "I": "aɪ",
    "J": "dʒeɪ",
    "K": "keɪ",
    "L": "ɛl",
    "M": "ɛm",
    "N": "ɛn",
    "O": "oʊ",
    "P": "piː",
    "Q": "kjuː",
    "R": "ɑr",
    "S": "ɛs",
    "T": "tiː",
    "U": "juː",
    "V": "viː",
    "W": "ˈdʌbəl juː",
    "X": "ɛks",
    "Y": "waɪ",
    "Z": "ziː",
}

_DOTTED_INITIALISM = re.compile(r"(?<![A-Za-z.'])((?:[A-Z]\.){2,3})(?![A-Za-z.'])")
_PLAIN_INITIALISM = re.compile(r"(?<![A-Za-z'])(([A-Z]){2,3})(?![A-Za-z'])")


def _render_letter_names(letters: str) -> str:
    """Return space-separated letter names without pause punctuation."""
    return " ".join(_LETTER_NAMES[letter] for letter in letters)


def _render_phonetic_names(letters: str) -> str:
    """Return space-separated IPA-style letter spellings."""
    return " ".join(_PHONETIC_NAMES[letter] for letter in letters)


def _normalize_initialisms(
    text: str, renderer: Callable[[str], str]
) -> str:
    """Apply the shared candidate classifier with a selected renderer."""
    def replace_dotted(match: re.Match[str]) -> str:
        raw = match.group(1)
        return renderer(raw.replace(".", ""))

    def replace_plain(match: re.Match[str]) -> str:
        letters = match.group(1)
        if letters in _COMMON_UPPERCASE_WORDS and letters not in _KNOWN_INITIALISMS:
            return letters
        return renderer(letters)

    text = _DOTTED_INITIALISM.sub(replace_dotted, text)
    return _PLAIN_INITIALISM.sub(replace_plain, text)


def pronounce_initialisms(text: str) -> str:
    """Expand 2-3 letter initialisms into letter names."""
    return _normalize_initialisms(text, _render_letter_names)


def pronounce_initialisms_phonetic(text: str) -> str:
    """Expand 2-3 letter initialisms using IPA-style Unicode spellings."""
    return _normalize_initialisms(text, _render_phonetic_names)
