import json
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error
from src.logger import get_logger

logger = get_logger(__name__)

def load_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed, falling back to ISO-8859-1.")
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    # Synthetic Probability if not exists
    if 'Probability' not in df.columns:
        logger.warning("'Probability' column missing. Generating random scores.")
        np.random.seed(42)
        df['Probability'] = np.random.uniform(0.1, 0.95, size=len(df))
    logger.info(f"Loaded dataset with {len(df)} records.")
    return df

def train_models(data_path='data/Software Questions.csv', model_dir='models'):
    logger.info(f"Loading data from {data_path}...")
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        logger.error(f"File not found at {data_path}")
        return

    # Features and Targets
    X = df['Question']
    y_category = df['Category']
    y_difficulty = df['Difficulty']
    y_probability = df['Probability']

    # Split data

    X_train, X_test, y_cat_train, y_cat_test, y_diff_train, y_diff_test, y_prob_train, y_prob_test = train_test_split(
        X, y_category, y_difficulty, y_probability, test_size=0.2, random_state=42
    )


    # 1. Category Classifier Comparison
    logger.info("--- Training Category Classifiers ---")
    from sklearn.svm import LinearSVC

    # Random Forest
    rf_cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_df=0.95, min_df=2)),
        ('clf', RandomForestClassifier(n_estimators=110, max_depth=50, random_state=42))
    ])
    rf_cat_pipeline.fit(X_train, y_cat_train)
    rf_score = rf_cat_pipeline.score(X_test, y_cat_test)
    logger.info(f"Random Forest Category Accuracy: {rf_score:.4f}")

    # SVM
    svm_cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC(random_state=42, C=1.2, dual='auto'))
    ])
    svm_cat_pipeline.fit(X_train, y_cat_train)
    svm_score = svm_cat_pipeline.score(X_test, y_cat_test)
    logger.info(f"SVM Category Accuracy:           {svm_score:.4f}")

    # Logistic Regression
    lr_cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1500, C=1.1, random_state=42))
    ])
    lr_cat_pipeline.fit(X_train, y_cat_train)
    lr_score = lr_cat_pipeline.score(X_test, y_cat_test)
    logger.info(f"Logistic Regression Category Accuracy: {lr_score:.4f}")

    # Select Best Category Model
    cat_scores = {'Random Forest': (rf_score, rf_cat_pipeline),
                  'SVM': (svm_score, svm_cat_pipeline),
                  'Logistic Regression': (lr_score, lr_cat_pipeline)}
    best_cat_name = max(cat_scores, key=lambda k: cat_scores[k][0])
    best_cat_model = cat_scores[best_cat_name][1]
    logger.info(f">> Selecting {best_cat_name} for Category Model")

    # 2. Difficulty Classifier Comparison
    logger.info("--- Training Difficulty Classifiers ---")
    # Random Forest
    rf_diff_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_diff_pipeline.fit(X_train, y_diff_train)
    rf_diff_score = rf_diff_pipeline.score(X_test, y_diff_test)
    logger.info(f"Random Forest Difficulty Accuracy: {rf_diff_score:.4f}")

    # SVM
    svm_diff_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC(random_state=42, dual='auto'))
    ])
    svm_diff_pipeline.fit(X_train, y_diff_train)
    svm_diff_score = svm_diff_pipeline.score(X_test, y_diff_test)
    logger.info(f"SVM Difficulty Accuracy:           {svm_diff_score:.4f}")

    # Logistic Regression
    lr_diff_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, random_state=42))
    ])
    lr_diff_pipeline.fit(X_train, y_diff_train)
    lr_diff_score = lr_diff_pipeline.score(X_test, y_diff_test)
    logger.info(f"Logistic Regression Difficulty Accuracy: {lr_diff_score:.4f}")

    # Select Best Difficulty Model
    diff_scores = {'Random Forest': (rf_diff_score, rf_diff_pipeline),
                   'SVM': (svm_diff_score, svm_diff_pipeline),
                   'Logistic Regression': (lr_diff_score, lr_diff_pipeline)}
    best_diff_name = max(diff_scores, key=lambda k: diff_scores[k][0])
    best_diff_model = diff_scores[best_diff_name][1]
    logger.info(f">> Selecting {best_diff_name} for Difficulty Model")

    # 3. Probability Regressor
    logger.info("--- Training Probability Regressor ---")
    prob_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    prob_pipeline.fit(X_train, y_prob_train)
    mse = mean_squared_error(y_prob_test, prob_pipeline.predict(X_test))
    logger.info(f"Probability Regressor MSE: {mse:.6f}")

    # 4. Nearest Neighbors for Related Questions
    logger.info("--- Training Nearest Neighbors Model ---")
    nn_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = nn_vectorizer.fit_transform(X)
    
    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=20, metric='cosine')
    nn_model.fit(X_tfidf)

    # Save Models
    import os
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    logger.info(f"Saving best models to '{model_dir}'...")
    joblib.dump(best_cat_model, f'{model_dir}/category_model.pkl')
    joblib.dump(best_diff_model, f'{model_dir}/difficulty_model.pkl')
    joblib.dump(prob_pipeline, f'{model_dir}/probability_model.pkl')
    
    # Save NN artifacts
    joblib.dump(nn_vectorizer, f'{model_dir}/nn_vectorizer.pkl')
    joblib.dump(nn_model, f'{model_dir}/nn_model.pkl')
    joblib.dump(df, f'{model_dir}/df.pkl')
    logger.info("All models saved successfully.")

    # Save Comparison Report (text)
    report = f"""
ML Algorithm Comparison Report
==============================

1. Category Classification
--------------------------
Random Forest Accuracy:      {rf_score:.4f}
SVM Accuracy:                {svm_score:.4f}
Logistic Regression Accuracy:{lr_score:.4f}
>> Winner: {best_cat_name}

2. Difficulty Classification
----------------------------
Random Forest Accuracy:      {rf_diff_score:.4f}
SVM Accuracy:                {svm_diff_score:.4f}
Logistic Regression Accuracy:{lr_diff_score:.4f}
>> Winner: {best_diff_name}

3. Probability Regression
--------------------------
Random Forest MSE: {mse:.6f}

Note: The best performing models have been automatically saved and will be used by the application.
"""
    logger.info(report)
    with open('comparison_report.txt', 'w') as f:
        f.write(report)
    logger.info("Comparison report written to comparison_report.txt")

    # Save machine-readable JSON metrics
    metrics = {
        'category': {
            'random_forest': round(rf_score, 4),
            'svm': round(svm_score, 4),
            'logistic_regression': round(lr_score, 4),
            'winner': best_cat_name,
        },
        'difficulty': {
            'random_forest': round(rf_diff_score, 4),
            'svm': round(svm_diff_score, 4),
            'logistic_regression': round(lr_diff_score, 4),
            'winner': best_diff_name,
        },
        'probability_mse': round(mse, 6),
    }
    with open('model_metrics.json', 'w') as jf:
        json.dump(metrics, jf, indent=2)
    logger.info("Model metrics saved to model_metrics.json")

if __name__ == "__main__":
    train_models()
