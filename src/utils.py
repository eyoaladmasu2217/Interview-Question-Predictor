"""
utils.py – Shared text preprocessing helpers for the Interview Predictor.
"""
import re
import string


def clean_text(text: str) -> str:
    """Lowercase, strip punctuation, and collapse whitespace."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[' + re.escape(string.punctuation) + r']', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def validate_question(text: str, min_words: int = 3) -> tuple[bool, str]:
    """
    Validate that the question meets minimum quality requirements.

    Returns:
        (is_valid: bool, error_message: str)
    """
    if not text or not text.strip():
        return False, "Question cannot be empty."
    word_count = len(text.strip().split())
    if word_count < min_words:
        return False, f"Question is too short. Please enter at least {min_words} words."
    if len(text.strip()) > 2000:
        return False, "Question is too long. Please limit to 2000 characters."
    return True, ""


def truncate_text(text: str, max_chars: int = 120) -> str:
    """Truncate text for display purposes, appending ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + '…'
