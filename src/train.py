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

    X_train, X_test, y_cat_train, y_cat_test, y_diff_train, y_diff_test, y_prob_train, y_prob_test = train_test_split(
        X, y_category, y_difficulty, y_probability, test_size=0.2, random_state=42
    )


    # 1. Category Classifier Comparison
    print("\n--- Training Category Classifiers ---")
    # Random Forest
    rf_cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_cat_pipeline.fit(X_train, y_cat_train)
    rf_score = rf_cat_pipeline.score(X_test, y_cat_test)
    print(f"Random Forest Category Accuracy: {rf_score:.4f}")

    # SVM
    from sklearn.svm import LinearSVC
    svm_cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC(random_state=42, dual='auto'))
    ])
    svm_cat_pipeline.fit(X_train, y_cat_train)
    svm_score = svm_cat_pipeline.score(X_test, y_cat_test)
    print(f"SVM Category Accuracy:           {svm_score:.4f}")

    # Select Best Category Model
    if svm_score > rf_score:
        print(">> Selecting SVM for Category Model")
        best_cat_model = svm_cat_pipeline
    else:
        print(">> Selecting Random Forest for Category Model")
        best_cat_model = rf_cat_pipeline

    # 2. Difficulty Classifier Comparison
    print("\n--- Training Difficulty Classifiers ---")
    # Random Forest
    rf_diff_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
    ])
    rf_diff_pipeline.fit(X_train, y_diff_train)
    rf_diff_score = rf_diff_pipeline.score(X_test, y_diff_test)
    print(f"Random Forest Difficulty Accuracy: {rf_diff_score:.4f}")

    # SVM
    svm_diff_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearSVC(random_state=42, dual='auto'))
    ])
    svm_diff_pipeline.fit(X_train, y_diff_train)
    svm_diff_score = svm_diff_pipeline.score(X_test, y_diff_test)
    print(f"SVM Difficulty Accuracy:           {svm_diff_score:.4f}")

    # Select Best Difficulty Model
    if svm_diff_score > rf_diff_score:
        print(">> Selecting SVM for Difficulty Model")
        best_diff_model = svm_diff_pipeline
    else:
        print(">> Selecting Random Forest for Difficulty Model")
        best_diff_model = rf_diff_pipeline

    # 3. Probability Regressor (Keeping RF for now as requested comparison was mainly for algos)
    print("\n--- Training Probability Regressor ---")
    prob_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    prob_pipeline.fit(X_train, y_prob_train)
    print("Probability Regressor MSE:", mean_squared_error(y_prob_test, prob_pipeline.predict(X_test)))

    # 4. Nearest Neighbors for Related Questions
    print("\n--- Training Nearest Neighbors Model ---")
    # vectorizer for this to ensure we can transform input separately
    nn_vectorizer = TfidfVectorizer(stop_words='english')
    X_tfidf = nn_vectorizer.fit_transform(X)
    
    from sklearn.neighbors import NearestNeighbors
    nn_model = NearestNeighbors(n_neighbors=20, metric='cosine') # Default to 20 neighbors as requested
    nn_model.fit(X_tfidf)

    # Save Models
    import os
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    print(f"\nSaving best models to {model_dir}...")
    joblib.dump(best_cat_model, f'{model_dir}/category_model.pkl')
    joblib.dump(best_diff_model, f'{model_dir}/difficulty_model.pkl')
    joblib.dump(prob_pipeline, f'{model_dir}/probability_model.pkl')
    
    # Save NN artifacts
    joblib.dump(nn_vectorizer, f'{model_dir}/nn_vectorizer.pkl')
    joblib.dump(nn_model, f'{model_dir}/nn_model.pkl')
    joblib.dump(df, f'{model_dir}/df.pkl') # Save data to retrieve questions
    print("Done.")

    # Save Comparison Report
    report = f"""
ML Algorithm Comparison Report
==============================

1. Category Classification
--------------------------
Random Forest Accuracy: {rf_score:.4f}
SVM Accuracy:           {svm_score:.4f}
>> Winner: {'SVM' if svm_score > rf_score else 'Random Forest'}

2. Difficulty Classification
----------------------------
Random Forest Accuracy: {rf_diff_score:.4f}
SVM Accuracy:           {svm_diff_score:.4f}
>> Winner: {'SVM' if svm_diff_score > rf_diff_score else 'Random Forest'}

Note: The best performing models have been automatically saved and will be used by the application.
"""
    print(report)
    with open('comparison_report.txt', 'w') as f:
        f.write(report)

if __name__ == "__main__":
    train_models()
