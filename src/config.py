"""
config.py – Central application configuration for the Interview Predictor.

All hard-coded paths, model hyper-parameters, and UI thresholds live here.
Import from this module instead of scattering magic strings/numbers across files.

Usage:
    from src.config import DATA_PATH, MODEL_DIR, PROB_THRESHOLDS
"""

from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

# Project root (two levels up from this file: src/config.py → src/ → project/)
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR   = PROJECT_ROOT / "data"
MODEL_DIR  = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT  # comparison_report.txt etc. live in root for now

DATA_PATH          = DATA_DIR / "Software Questions.csv"
COMPARISON_REPORT  = REPORTS_DIR / "comparison_report.txt"
METRICS_JSON       = REPORTS_DIR / "model_metrics.json"
APP_LOG            = PROJECT_ROOT / "app.log"

# ──────────────────────────────────────────────────────────────────────────────
# Model artefact filenames
# ──────────────────────────────────────────────────────────────────────────────

MODEL_FILES = {
    "category_model":    MODEL_DIR / "category_model.pkl",
    "difficulty_model":  MODEL_DIR / "difficulty_model.pkl",
    "probability_model": MODEL_DIR / "probability_model.pkl",
    "nn_vectorizer":     MODEL_DIR / "nn_vectorizer.pkl",
    "nn_model":          MODEL_DIR / "nn_model.pkl",
    "dataframe":         MODEL_DIR / "df.pkl",
}

# ──────────────────────────────────────────────────────────────────────────────
# Training hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

TRAIN_TEST_SPLIT_SIZE = 0.2
RANDOM_STATE          = 42

# TF-IDF settings (shared across all pipelines)
TFIDF_STOP_WORDS = "english"

# Random Forest
RF_N_ESTIMATORS  = 100

# Logistic Regression
LR_MAX_ITER      = 1000

# Nearest Neighbours (for related question retrieval)
NN_N_NEIGHBORS   = 20
NN_METRIC        = "cosine"

# ──────────────────────────────────────────────────────────────────────────────
# Probability display thresholds
# ──────────────────────────────────────────────────────────────────────────────

PROB_THRESHOLDS = {
    "high":   0.70,   # ≥ 70 %  → "High Likelihood"
    "medium": 0.40,   # ≥ 40 %  → "Moderate"
    # < 40 %  → "Low Likelihood"
}

PROB_COLORS = {
    "high":   "#22c55e",   # green
    "medium": "#f59e0b",   # amber
    "low":    "#ef4444",   # red
}

# ──────────────────────────────────────────────────────────────────────────────
# Validation thresholds
# ──────────────────────────────────────────────────────────────────────────────

QUESTION_MIN_WORDS = 3
QUESTION_MAX_CHARS = 2000

# ──────────────────────────────────────────────────────────────────────────────
# UI settings
# ──────────────────────────────────────────────────────────────────────────────

APP_TITLE       = "Interview Predictor and Recommendation"
APP_SUBTITLE    = "AI-powered analytics for software engineering interview preparation."
HISTORY_MAX_LEN = 5    # Number of recent predictions shown in sidebar
RELATED_TOP_K   = 20   # Number of related questions to retrieve from NN model
KEYWORD_TOP_N   = 7    # Number of keywords to extract and display
