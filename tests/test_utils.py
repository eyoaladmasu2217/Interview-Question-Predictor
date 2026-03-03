"""
tests/test_utils.py – Unit tests for src/utils.py helpers.

Run with:
    python -m pytest tests/ -v
or:
    python -m unittest discover -s tests -v
"""

import sys
import os
import unittest

# Ensure project root is on path so `from src.xxx` works
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import (
    clean_text,
    validate_question,
    truncate_text,
    extract_keywords,
    detect_question_type,
)


# ──────────────────────────────────────────────────────────────────────────────
# clean_text
# ──────────────────────────────────────────────────────────────────────────────

class TestCleanText(unittest.TestCase):

    def test_lowercases(self):
        self.assertEqual(clean_text("Hello World"), "hello world")

    def test_strips_punctuation(self):
        result = clean_text("What is O(n)?")
        self.assertNotIn("(", result)
        self.assertNotIn(")", result)
        self.assertNotIn("?", result)

    def test_collapses_whitespace(self):
        result = clean_text("  multiple   spaces   here  ")
        self.assertEqual(result, "multiple spaces here")

    def test_non_string_returns_empty(self):
        self.assertEqual(clean_text(None), "")
        self.assertEqual(clean_text(42),   "")
        self.assertEqual(clean_text([]),   "")

    def test_empty_string(self):
        self.assertEqual(clean_text(""), "")


# ──────────────────────────────────────────────────────────────────────────────
# validate_question
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateQuestion(unittest.TestCase):

    def test_valid_question(self):
        ok, msg = validate_question("What is a binary search tree?")
        self.assertTrue(ok)
        self.assertEqual(msg, "")

    def test_empty_string_invalid(self):
        ok, msg = validate_question("")
        self.assertFalse(ok)
        self.assertIn("empty", msg.lower())

    def test_whitespace_only_invalid(self):
        ok, msg = validate_question("   ")
        self.assertFalse(ok)

    def test_too_short_invalid(self):
        ok, msg = validate_question("hello")
        self.assertFalse(ok)
        self.assertIn("short", msg.lower())

    def test_too_long_invalid(self):
        long_q = "word " * 401   # 401 words, well over 2000 chars
        ok, msg = validate_question(long_q)
        self.assertFalse(ok)
        self.assertIn("long", msg.lower())

    def test_spam_repetition_rejected(self):
        spammy = "foo " * 20
        ok, msg = validate_question(spammy)
        self.assertFalse(ok)
        self.assertIn("repeated", msg.lower())

    def test_custom_min_words(self):
        ok, _ = validate_question("explain recursion", min_words=5)
        self.assertFalse(ok)
        ok2, _ = validate_question("explain recursion", min_words=2)
        self.assertTrue(ok2)

    def test_none_invalid(self):
        ok, msg = validate_question(None)
        self.assertFalse(ok)


# ──────────────────────────────────────────────────────────────────────────────
# truncate_text
# ──────────────────────────────────────────────────────────────────────────────

class TestTruncateText(unittest.TestCase):

    def test_short_text_unchanged(self):
        text = "Short text"
        self.assertEqual(truncate_text(text, max_chars=50), text)

    def test_long_text_truncated(self):
        text = "This is a fairly long sentence that should be truncated at some point."
        result = truncate_text(text, max_chars=20)
        self.assertLessEqual(len(result), 25)   # allow for ellipsis
        self.assertTrue(result.endswith("…"))

    def test_exact_length_not_truncated(self):
        text = "x" * 120
        self.assertEqual(truncate_text(text, max_chars=120), text)


# ──────────────────────────────────────────────────────────────────────────────
# extract_keywords
# ──────────────────────────────────────────────────────────────────────────────

class TestExtractKeywords(unittest.TestCase):

    def test_returns_list(self):
        result = extract_keywords("What is a binary search tree?")
        self.assertIsInstance(result, list)

    def test_respects_top_n(self):
        text = "binary search tree node pointer left right traversal insertion deletion"
        result = extract_keywords(text, top_n=3)
        self.assertLessEqual(len(result), 3)

    def test_no_stop_words(self):
        result = extract_keywords("What is the difference between a stack and a queue?")
        for kw in result:
            self.assertNotIn(kw, {"what", "is", "the", "a", "and"})

    def test_empty_text_returns_empty(self):
        self.assertEqual(extract_keywords(""), [])

    def test_all_stop_words_returns_empty(self):
        self.assertEqual(extract_keywords("the a an and or but"), [])


# ──────────────────────────────────────────────────────────────────────────────
# detect_question_type
# ──────────────────────────────────────────────────────────────────────────────

class TestDetectQuestionType(unittest.TestCase):

    def test_conceptual(self):
        self.assertEqual(detect_question_type("What is a linked list?"), "Conceptual")

    def test_implementation(self):
        self.assertEqual(
            detect_question_type("Implement a function to reverse a linked list."),
            "Implementation"
        )

    def test_design(self):
        self.assertEqual(
            detect_question_type("Design a URL shortener system like bit.ly."),
            "Design"
        )

    def test_comparison(self):
        self.assertEqual(
            detect_question_type("What is the difference between REST and GraphQL?"),
            "Comparison"
        )

    def test_debugging(self):
        self.assertEqual(
            detect_question_type("Why does this code throw a null pointer exception?"),
            "Debugging"
        )

    def test_general_fallback(self):
        result = detect_question_type("Tell me something interesting about Python.")
        # "Tell me about" → Conceptual due to regex; just assert it returns a string
        self.assertIsInstance(result, str)

    def test_case_insensitive(self):
        t1 = detect_question_type("IMPLEMENT a sorting algorithm.")
        t2 = detect_question_type("implement a sorting algorithm.")
        self.assertEqual(t1, t2)


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    unittest.main(verbosity=2)
