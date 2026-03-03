"""
data_validator.py – Dataset quality checker for the Interview Predictor.

Scans the training CSV for common data quality issues and prints/returns a
structured report.  Can be run directly or imported as a module.

Usage:
    python -m src.data_validator
    python -m src.data_validator --data "data/Software Questions.csv" --fix
"""

import argparse
import os
import re

import pandas as pd

from src.logger import get_logger

logger = get_logger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS   = {"Question", "Category", "Difficulty"}
ALLOWED_DIFFICULTY = {"Easy", "Medium", "Hard"}
MIN_QUESTION_WORDS = 3
MAX_QUESTION_CHARS = 2000


# ──────────────────────────────────────────────────────────────────────────────
# Validation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _check_missing_columns(df: pd.DataFrame) -> list[str]:
    missing = REQUIRED_COLUMNS - set(df.columns)
    return sorted(missing)


def _check_nulls(df: pd.DataFrame) -> dict[str, int]:
    """Return {column: null_count} for columns that have at least one null."""
    null_counts = df[list(REQUIRED_COLUMNS & set(df.columns))].isnull().sum()
    return {col: int(cnt) for col, cnt in null_counts.items() if cnt > 0}


def _check_difficulty_values(df: pd.DataFrame) -> list[int]:
    """Return row indices where Difficulty is not in ALLOWED_DIFFICULTY."""
    if "Difficulty" not in df.columns:
        return []
    bad_mask = ~df["Difficulty"].isin(ALLOWED_DIFFICULTY)
    return df.index[bad_mask].tolist()


def _check_question_length(df: pd.DataFrame) -> dict[str, list[int]]:
    """Return indices of questions that are too short or too long."""
    if "Question" not in df.columns:
        return {"too_short": [], "too_long": []}
    q = df["Question"].fillna("")
    word_counts = q.str.split().str.len().fillna(0)
    char_counts  = q.str.len()
    return {
        "too_short": df.index[word_counts < MIN_QUESTION_WORDS].tolist(),
        "too_long":  df.index[char_counts  > MAX_QUESTION_CHARS].tolist(),
    }


def _check_duplicates(df: pd.DataFrame) -> int:
    """Return number of exact duplicate Question rows."""
    if "Question" not in df.columns:
        return 0
    return int(df["Question"].duplicated().sum())


def _check_probability_range(df: pd.DataFrame) -> list[int]:
    """Return indices where Probability is outside [0, 1]."""
    if "Probability" not in df.columns:
        return []
    p = pd.to_numeric(df["Probability"], errors="coerce")
    bad = df.index[(p < 0) | (p > 1) | p.isna()].tolist()
    return bad


# ──────────────────────────────────────────────────────────────────────────────
# Fix helpers  (--fix mode)
# ──────────────────────────────────────────────────────────────────────────────

def _apply_fixes(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Apply safe automatic fixes and return (fixed_df, list_of_actions).

    Fixes applied:
    - Strip leading/trailing whitespace from text columns
    - Normalise Difficulty capitalisation (e.g. 'easy' → 'Easy')
    - Drop exact Question duplicates (keep first)
    - Drop rows with null Question, Category, or Difficulty
    """
    actions: list[str] = []
    original_len = len(df)

    # Strip whitespace
    for col in ["Question", "Category", "Difficulty"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    # Normalise difficulty capitalisation
    if "Difficulty" in df.columns:
        df["Difficulty"] = df["Difficulty"].str.capitalize()
        actions.append("Normalised Difficulty column capitalisation.")

    # Drop nulls in required columns
    before = len(df)
    df = df.dropna(subset=list(REQUIRED_COLUMNS & set(df.columns)))
    dropped_nulls = before - len(df)
    if dropped_nulls:
        actions.append(f"Dropped {dropped_nulls} row(s) with null values in required columns.")

    # Drop duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["Question"], keep="first")
    dropped_dups = before - len(df)
    if dropped_dups:
        actions.append(f"Dropped {dropped_dups} duplicate question row(s).")

    total_removed = original_len - len(df)
    if total_removed:
        actions.append(f"Dataset reduced from {original_len} → {len(df)} rows.")
    else:
        actions.append("No rows removed — dataset is clean.")

    return df, actions


# ──────────────────────────────────────────────────────────────────────────────
# Main validation entry point
# ──────────────────────────────────────────────────────────────────────────────

def validate_dataset(data_path: str = "data/Software Questions.csv",
                     fix: bool = False,
                     fixed_output: str | None = None) -> dict:
    """
    Run all data quality checks on *data_path*.

    Parameters
    ----------
    data_path     : Path to the CSV file.
    fix           : If True, apply automatic fixes.
    fixed_output  : Path to write the (optionally fixed) clean CSV.

    Returns
    -------
    dict with keys: issues, summary, fixed_df (if fix=True)
    """
    logger.info(f"Loading dataset from: {data_path}")
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="ISO-8859-1")
    except FileNotFoundError:
        logger.error(f"File not found: {data_path}")
        return {"error": f"File not found: {data_path}"}

    logger.info(f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns.")

    issues: dict = {}

    # 1. Missing columns
    missing_cols = _check_missing_columns(df)
    if missing_cols:
        issues["missing_columns"] = missing_cols
        logger.warning(f"Missing required columns: {missing_cols}")

    # 2. Null values
    null_info = _check_nulls(df)
    if null_info:
        issues["null_values"] = null_info
        logger.warning(f"Null values found: {null_info}")

    # 3. Invalid difficulty labels
    bad_diff = _check_difficulty_values(df)
    if bad_diff:
        issues["invalid_difficulty_rows"] = bad_diff[:20]  # cap for display
        logger.warning(f"{len(bad_diff)} row(s) have invalid Difficulty values.")

    # 4. Question length violations
    length_issues = _check_question_length(df)
    if length_issues["too_short"]:
        issues["too_short_questions"] = len(length_issues["too_short"])
        logger.warning(f"{len(length_issues['too_short'])} question(s) are too short (< {MIN_QUESTION_WORDS} words).")
    if length_issues["too_long"]:
        issues["too_long_questions"] = len(length_issues["too_long"])
        logger.warning(f"{len(length_issues['too_long'])} question(s) exceed {MAX_QUESTION_CHARS} chars.")

    # 5. Duplicates
    dup_count = _check_duplicates(df)
    if dup_count:
        issues["duplicate_questions"] = dup_count
        logger.warning(f"{dup_count} duplicate question(s) found.")

    # 6. Probability out of range
    bad_prob = _check_probability_range(df)
    if bad_prob:
        issues["out_of_range_probability"] = len(bad_prob)
        logger.warning(f"{len(bad_prob)} Probability value(s) outside [0, 1].")

    summary = {
        "total_rows":    len(df),
        "total_columns": len(df.columns),
        "issue_count":   len(issues),
        "status":        "⚠️  Issues found" if issues else "✅  All checks passed",
    }

    result = {"issues": issues, "summary": summary}

    # Apply fixes if requested
    if fix:
        fixed_df, actions = _apply_fixes(df.copy())
        result["fixed_df"]     = fixed_df
        result["fix_actions"]  = actions
        logger.info("Fix actions applied:")
        for a in actions:
            logger.info(f"  • {a}")

        out_path = fixed_output or data_path.replace(".csv", "_cleaned.csv")
        fixed_df.to_csv(out_path, index=False, encoding="utf-8")
        logger.info(f"Cleaned dataset saved to: {out_path}")
        result["fixed_output"] = out_path

    return result


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate (and optionally fix) the Interview Predictor dataset."
    )
    parser.add_argument(
        "--data", default="data/Software Questions.csv",
        help="Path to the CSV dataset."
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="Apply automatic fixes and save a cleaned copy of the dataset."
    )
    parser.add_argument(
        "--output", default=None,
        help="Path to save the cleaned CSV (used with --fix). "
             "Defaults to <original_name>_cleaned.csv."
    )
    args = parser.parse_args()

    result = validate_dataset(data_path=args.data, fix=args.fix, fixed_output=args.output)

    if "error" in result:
        print(f"\n❌ Error: {result['error']}")
        return

    summary = result["summary"]
    print(f"\n{'═' * 50}")
    print(f"  Dataset Validation Report")
    print(f"{'═' * 50}")
    print(f"  Rows:    {summary['total_rows']}")
    print(f"  Columns: {summary['total_columns']}")
    print(f"  Status:  {summary['status']}")

    if result["issues"]:
        print(f"\n  Issues detected ({summary['issue_count']}):")
        for key, val in result["issues"].items():
            print(f"    • {key}: {val}")
    else:
        print("\n  No issues found. Dataset looks healthy!")

    if args.fix and "fix_actions" in result:
        print(f"\n  Fix actions applied:")
        for a in result["fix_actions"]:
            print(f"    • {a}")
        print(f"\n  Cleaned file saved to: {result.get('fixed_output', '–')}")

    print(f"{'═' * 50}\n")


if __name__ == "__main__":
    main()
