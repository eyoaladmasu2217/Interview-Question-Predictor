import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, mean_squared_error

def load_data(filepath):
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='ISO-8859-1')
    # Synthetic Probability if not exists
    if 'Probability' not in df.columns:
        # Fallback if update failed, but ideally we use the CSV values
        print("Warning: 'Probability' column missing. Generating random scores.")
        np.random.seed(42)
        df['Probability'] = np.random.uniform(0.1, 0.95, size=len(df))
    return df

def train_models(data_path='data/Software Questions.csv', model_dir='models'):
    print(f"Loading data from {data_path}...")
    try:
        df = load_data(data_path)
    except FileNotFoundError:
        print(f"Error: File not found at {data_path}")
        return

    # Features and Targets
    X = df['Question']
    y_category = df['Category']
    y_difficulty = df['Difficulty']
    y_probability = df['Probability']

    # Split data
    # We use the same split for all for simplicity, or we could split separately. 
    # Since inputs are the same, same split is fine.
    X_train, X_test, y_cat_train, y_cat_test, y_diff_train, y_diff_test, y_prob_train, y_prob_test = train_test_split(
        X, y_category, y_difficulty, y_probability, test_size=0.2, random_state=42
    )

    # Pipelines
    # 1. Category Classifier
    print("Training Category Classifier...")
    cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    cat_pipeline.fit(X_train, y_cat_train)
    print("Category Classifier Score:", cat_pipeline.score(X_test, y_cat_test))

    # 2. Difficulty Classifier
    print("Training Difficulty Classifier...")
    diff_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    diff_pipeline.fit(X_train, y_diff_train)
    print("Difficulty Classifier Score:", diff_pipeline.score(X_test, y_diff_test))

    # 3. Probability Regressor
    print("Training Probability Regressor...")
    prob_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    prob_pipeline.fit(X_train, y_prob_train)
    print("Probability Regressor MSE:", mean_squared_error(y_prob_test, prob_pipeline.predict(X_test)))

    # Save Models
    import os
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print(f"Saving models to {model_dir}...")
    joblib.dump(cat_pipeline, f'{model_dir}/category_model.pkl')
    joblib.dump(diff_pipeline, f'{model_dir}/difficulty_model.pkl')
    joblib.dump(prob_pipeline, f'{model_dir}/probability_model.pkl')
    print("Done.")

if __name__ == "__main__":
    train_models()
