"""
evaluator.py – Standalone model evaluation script for the Interview Predictor.

Loads the saved models and the dataset, then prints a detailed classification
report for both category and difficulty models, plus regression stats for the
probability model.  Optionally saves the report to a text file.

Usage:
    python -m src.evaluator
    python -m src.evaluator --output my_report.txt
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_models(model_dir: str = "models") -> dict:
    """Load all persisted model artefacts from *model_dir*."""
    paths = {
        "cat_model":    "category_model.pkl",
        "diff_model":   "difficulty_model.pkl",
        "prob_model":   "probability_model.pkl",
        "nn_vectorizer":"nn_vectorizer.pkl",
        "nn_model":     "nn_model.pkl",
        "df":           "df.pkl",
    }
    artefacts = {}
    for key, fname in paths.items():
        full = os.path.join(model_dir, fname)
        if not os.path.exists(full):
            print(f"[WARNING] Artefact not found: {full}")
            artefacts[key] = None
        else:
            artefacts[key] = joblib.load(full)
    return artefacts


def _load_data(data_path: str = "data/Software Questions.csv") -> pd.DataFrame:
    """Load the dataset, adding a synthetic Probability column if absent."""
    try:
        df = pd.read_csv(data_path, encoding="utf-8")
    except UnicodeDecodeError:
        df = pd.read_csv(data_path, encoding="ISO-8859-1")
    if "Probability" not in df.columns:
        np.random.seed(42)
        df["Probability"] = np.random.uniform(0.1, 0.95, size=len(df))
    return df


def _section(title: str, width: int = 60) -> str:
    bar = "─" * width
    return f"\n{bar}\n  {title}\n{bar}"


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation logic
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(model_dir: str = "models",
             data_path: str = "data/Software Questions.csv",
             output_path: str | None = None) -> str:
    """
    Run full evaluation and return a formatted report string.

    Parameters
    ----------
    model_dir   : directory where .pkl model files are stored
    data_path   : path to the CSV dataset
    output_path : optional file path to write the report

    Returns
    -------
    str – the full report text
    """
    print("Loading models …")
    artefacts = _load_models(model_dir)

    print(f"Loading dataset from {data_path} …")
    df = _load_data(data_path)

    X = df["Question"]
    y_cat  = df["Category"]
    y_diff = df["Difficulty"]
    y_prob = df["Probability"]

    _, X_test, _, y_cat_test, _, y_diff_test, _, y_prob_test = train_test_split(
        X, y_cat, y_diff, y_prob, test_size=0.2, random_state=42
    )

    lines: list[str] = []

    # ── Model info ────────────────────────────────────────────────────────────
    lines.append(_section("Model Information"))
    for label, key in [("Category Model",  "cat_model"),
                       ("Difficulty Model", "diff_model"),
                       ("Probability Model","prob_model")]:
        m = artefacts.get(key)
        if m is None:
            lines.append(f"  {label}: NOT LOADED")
        else:
            step = m.named_steps.get("clf") or m.named_steps.get("reg")
            lines.append(f"  {label}: {type(step).__name__}")
    lines.append(f"  Dataset rows: {len(df)}")

    # ── Category classification ───────────────────────────────────────────────
    lines.append(_section("1. Category Classification"))
    cat_model = artefacts.get("cat_model")
    if cat_model:
        y_pred_cat = cat_model.predict(X_test)
        acc = (y_pred_cat == y_cat_test).mean()
        lines.append(f"  Accuracy: {acc:.4f}")
        lines.append("")
        lines.append(classification_report(y_cat_test, y_pred_cat))
    else:
        lines.append("  [SKIPPED – model not loaded]")

    # ── Difficulty classification ─────────────────────────────────────────────
    lines.append(_section("2. Difficulty Classification"))
    diff_model = artefacts.get("diff_model")
    if diff_model:
        y_pred_diff = diff_model.predict(X_test)
        acc = (y_pred_diff == y_diff_test).mean()
        lines.append(f"  Accuracy: {acc:.4f}")
        lines.append("")
        lines.append(classification_report(y_diff_test, y_pred_diff))
    else:
        lines.append("  [SKIPPED – model not loaded]")

    # ── Probability regression ────────────────────────────────────────────────
    lines.append(_section("3. Probability Regression"))
    prob_model = artefacts.get("prob_model")
    if prob_model:
        y_pred_prob = prob_model.predict(X_test)
        mse  = mean_squared_error(y_prob_test, y_pred_prob)
        mae  = mean_absolute_error(y_prob_test, y_pred_prob)
        r2   = r2_score(y_prob_test, y_pred_prob)
        rmse = mse ** 0.5
        lines.append(f"  MSE  : {mse:.6f}")
        lines.append(f"  RMSE : {rmse:.6f}")
        lines.append(f"  MAE  : {mae:.6f}")
        lines.append(f"  R²   : {r2:.4f}")
    else:
        lines.append("  [SKIPPED – model not loaded]")

    # ── JSON metrics (if available) ───────────────────────────────────────────
    metrics_path = "model_metrics.json"
    if os.path.exists(metrics_path):
        lines.append(_section("4. Saved Training Metrics (model_metrics.json)"))
        with open(metrics_path) as jf:
            mdata = json.load(jf)
        lines.append(json.dumps(mdata, indent=4))

    report = "\n".join(lines)

    if output_path:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"Report saved to {output_path}")

    return report


# ──────────────────────────────────────────────────────────────────────────────
# CLI entry point
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the Interview Predictor's trained models."
    )
    parser.add_argument(
        "--model-dir", default="models",
        help="Directory containing .pkl model files (default: models/)"
    )
    parser.add_argument(
        "--data", default="data/Software Questions.csv",
        help="Path to the CSV dataset (default: data/Software Questions.csv)"
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save the report as a text file."
    )
    args = parser.parse_args()

    report = evaluate(
        model_dir=args.model_dir,
        data_path=args.data,
        output_path=args.output,
    )
    print(report)


if __name__ == "__main__":
    main()
