# Interview Question Predictor

> An AI-powered tool that predicts the **Category**, **Difficulty**, and **Appearance Probability** of software engineering interview questions — with a rich Streamlit dashboard, batch analysis, and automatic model selection.

---

## ✨ Features

| Feature | Description |
|---|---|
| **Category Prediction** | Classifies questions into topics: Arrays, System Design, General Programming, etc. |
| **Difficulty Prediction** | Estimates Easy / Medium / Hard with a confidence breakdown |
| **Appearance Probability** | Calculates the likelihood of a question appearing in a real interview |
| **Question Type Detection** | Auto-detects Conceptual / Implementation / Design / Comparison / Debugging |
| **Keyword Highlights** | Extracts the top key concepts detected in your question |
| **Single Analyser** | Analyse one question at a time with rich visual output |
| **Batch Analyser** | Upload a CSV or paste multiple questions to get bulk predictions |
| **Export Options** | Download single or batch predictions as a tidy CSV or JSON file |
| **Model Auto-Selection** | Trains Random Forest, SVM, and Logistic Regression — picks the best automatically |
| **Model Evaluation** | Standalone `src/evaluator.py` script for detailed metrics reporting |
| **Prediction History** | Sidebar tracks your last 5 predictions |
| **JSON Metrics** | `model_metrics.json` stores training accuracy for every model |
| **Interactive Retraining** | Click "Retrain System" in the sidebar to refresh all models |

---

## 🏗️ Architecture

```
Interview Predictor/
├── app.py                    # Streamlit UI (Single + Batch tabs)
├── requirements.txt          # Python dependencies
├── comparison_report.txt     # Human-readable training comparison
├── model_metrics.json        # Machine-readable accuracy/MSE metrics
├── data/
│   └── Software Questions.csv  # Training dataset
├── models/                   # Persisted .pkl model artefacts
│   ├── category_model.pkl
│   ├── difficulty_model.pkl
│   ├── probability_model.pkl
│   ├── nn_vectorizer.pkl
│   ├── nn_model.pkl
│   └── df.pkl
└── src/
    ├── train.py     # Model training + auto-selection + metrics saving
    ├── predict.py   # Predictor class (single + batch + model info)
    ├── utils.py     # clean_text, validate_question, extract_keywords, detect_question_type
    └── evaluator.py # Standalone evaluation script with CLI interface
```

### ML Pipeline

```
Raw Question Text
      │
      ▼
 TF-IDF Vectoriser (stop-word removal)
      │
      ├──► Category Classifier   (RF vs SVM vs LR → best selected)
      ├──► Difficulty Classifier (RF vs SVM vs LR → best selected)
      ├──► Probability Regressor (Random Forest Regressor)
      └──► Nearest Neighbours   (cosine-similarity for related questions)
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/eyoaladmasu2217/Interview-Question-Predictor.git
cd Interview-Question-Predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the models

Models are not committed to Git (they are in `.gitignore`). Train them once before launching the app:

```bash
python -m src.train
```

This will:
- Compare Random Forest, SVM, and Logistic Regression for both category and difficulty
- Automatically save the best-performing model for each task
- Write `comparison_report.txt` and `model_metrics.json`

---

## ▶️ Usage

### Run the web application

```bash
streamlit run app.py
```

Open your browser to `http://localhost:8501`.

### Single Analyser tab

1. Type or paste a technical interview question.
2. Click **Generate Analysis**.
3. View Category (with confidence bars), Difficulty (with confidence bars), Appearance Probability, Question Type badge, and Key Concepts.
4. Optionally **Export Result as CSV** or **Export Result as JSON**.
5. Browse similar questions pulled from the training corpus.

### Batch Analyser tab

1. Upload a CSV file with a `Question` column, **or** paste questions one per line.
2. Click **▶ Run Batch Analysis**.
3. View summary stats (top category, average probability, most common difficulty).
4. Browse the full results table.
5. Click **Download Full Results as CSV** or **JSON**.

---

## 🧪 Model Evaluation

Run the standalone evaluator for a detailed report on the held-out test set:

```bash
python -m src.evaluator
```

Save the report to a file:

```bash
python -m src.evaluator --output eval_report.txt
```

Custom paths:

```bash
python -m src.evaluator --model-dir models --data "data/Software Questions.csv" --output report.txt
```

The evaluator prints:
- Model types in use
- Classification report (precision / recall / F1) for category and difficulty
- MSE / RMSE / MAE / R² for probability regression
- Training metrics from `model_metrics.json` (if available)

---

## 📊 Dataset

The model is trained on `data/Software Questions.csv`. The dataset should contain at minimum:

| Column | Description |
|---|---|
| `Question` | The interview question text |
| `Category` | Topic label (e.g., Arrays, System Design) |
| `Difficulty` | Easy / Medium / Hard |
| `Probability` | (Optional) Float 0–1 interview appearance score |

If `Probability` is absent it is generated synthetically at training time.

You can extend the dataset with your own questions and click **Retrain System** in the sidebar.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Frontend | Streamlit |
| ML Models | Scikit-learn (RandomForest, LinearSVC, LogisticRegression, NearestNeighbors) |
| Feature Engineering | TF-IDF Vectoriser |
| Data | Pandas, NumPy |
| Serialisation | Joblib |
| Language | Python 3.10+ |

---

## 📝 License

MIT
