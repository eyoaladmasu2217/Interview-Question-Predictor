"""
utils.py – Shared text preprocessing helpers for the Interview Predictor.
"""
import re
import string
import math
from collections import Counter

# Common English stop words (lightweight, no NLTK dependency)
_STOP_WORDS = {
    'a', 'an', 'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
    'of', 'with', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have',
    'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
    'may', 'might', 'shall', 'can', 'need', 'dare', 'ought', 'used', 'it',
    'its', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'we',
    'they', 'me', 'him', 'her', 'us', 'them', 'my', 'your', 'his', 'our',
    'what', 'how', 'why', 'when', 'where', 'which', 'who', 'whom', 'not',
    'if', 'then', 'else', 'so', 'yet', 'both', 'either', 'neither', 'just',
    'about', 'between', 'into', 'through', 'during', 'before', 'after',
    'up', 'down', 'out', 'off', 'over', 'under', 'again', 'there', 'here',
}


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
    words = text.strip().split()
    word_count = len(words)
    if word_count < min_words:
        return False, f"Question is too short. Please enter at least {min_words} words."
    if len(text.strip()) > 2000:
        return False, "Question is too long. Please limit to 2000 characters."
    # Spam / repetition guard: flag if single word >50% of all words
    if word_count >= 6:
        freq = Counter(w.lower() for w in words)
        most_common_word, most_common_count = freq.most_common(1)[0]
        if most_common_count / word_count > 0.5 and most_common_word not in _STOP_WORDS:
            return False, "Question appears to contain repeated words. Please enter a meaningful question."
    return True, ""


def truncate_text(text: str, max_chars: int = 120) -> str:
    """Truncate text for display purposes, appending ellipsis if needed."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + '…'


def extract_keywords(text: str, top_n: int = 6) -> list[str]:
    """
    Extract the top N meaningful keywords from the given text using a
    simple TF-like term scoring (frequent non-stop-words, length-weighted).

    Returns:
        A list of keyword strings, ranked by relevance score.
    """
    cleaned = clean_text(text)
    tokens = [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 2]
    if not tokens:
        return []
    freq = Counter(tokens)
    # Score = tf * log(1 + word_length) to prefer meaningful long terms
    scored = {word: count * math.log(1 + len(word)) for word, count in freq.items()}
    ranked = sorted(scored, key=lambda w: scored[w], reverse=True)
    return ranked[:top_n]


def detect_question_type(text: str) -> str:
    """
    Heuristically classify the style/type of an interview question.

    Types:
        - Conceptual   : "What is …", "Explain …", "Define …"
        - Implementation: "Write …", "Implement …", "Code …"
        - Design        : "Design …", "Architect …", "How would you build …"
        - Comparison    : "Compare …", "Difference between …", "vs"
        - Debugging     : "Why does …", "Find the bug …", "Fix …"
        - General       : everything else
    """
    lower = text.lower()
    if re.search(r'\b(write|implement|code|program|develop|create a function|create an algorithm)\b', lower):
        return 'Implementation'
    if re.search(r'\b(design|architect|how would you build|build a system|system design)\b', lower):
        return 'Design'
    if re.search(r'\b(compare|difference between|vs\.?|versus|contrast|distinguish)\b', lower):
        return 'Comparison'
    if re.search(r'\b(why does|find the bug|debug|fix|error|exception|mistake)\b', lower):
        return 'Debugging'
    if re.search(r'\b(what is|what are|explain|define|describe|tell me about)\b', lower):
        return 'Conceptual'
    return 'General'
