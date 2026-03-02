import joblib
import os
import numpy as np

class Predictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.nn_vectorizer = None
        self.nn_model = None
        self.df = None
        self.load_models()

    def load_models(self):
        try:
            self.cat_model = joblib.load(os.path.join(self.model_dir, 'category_model.pkl'))
            self.diff_model = joblib.load(os.path.join(self.model_dir, 'difficulty_model.pkl'))
            self.prob_model = joblib.load(os.path.join(self.model_dir, 'probability_model.pkl'))
            
            # Load NN artifacts
            self.nn_vectorizer = joblib.load(os.path.join(self.model_dir, 'nn_vectorizer.pkl'))
            self.nn_model = joblib.load(os.path.join(self.model_dir, 'nn_model.pkl'))
            self.df = joblib.load(os.path.join(self.model_dir, 'df.pkl'))
            
        except FileNotFoundError:
            print("Models not found. Please train the models first.")

    def predict(self, question_text):
        if not self.cat_model or not self.diff_model or not self.prob_model:
            return None

        X = [question_text]

        category = self.cat_model.predict(X)[0]
        difficulty = self.diff_model.predict(X)[0]
        probability = self.prob_model.predict(X)[0]

        # --- Confidence scores for category ---
        cat_confidence = {}
        clf = self.cat_model.named_steps['clf']
        tfidf = self.cat_model.named_steps['tfidf']
        X_vec = tfidf.transform(X)
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_vec)[0]
            classes = clf.classes_
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X_vec)[0]
            # Softmax normalisation
            e_scores = np.exp(scores - np.max(scores))
            probs = e_scores / e_scores.sum()
            classes = clf.classes_
        else:
            probs, classes = [], []

        if len(probs):
            top_indices = np.argsort(probs)[::-1][:3]
            cat_confidence = {
                classes[i]: round(float(probs[i]) * 100, 1)
                for i in top_indices
            }

        return {
            'Category': category,
            'Difficulty': difficulty,
            'Probability': probability,
            'CategoryConfidence': cat_confidence,
        }

    def get_related_questions(self, question_text, top_k=20):
        if not self.nn_model or not self.nn_vectorizer or self.df is None:
            return []
        
        # Vectorize input
        X_vec = self.nn_vectorizer.transform([question_text])
        
        # Find neighbors
        distances, indices = self.nn_model.kneighbors(X_vec, n_neighbors=top_k)
        
        # Retrieve questions
        related_questions = []
        for idx in indices[0]:
            related_questions.append(self.df.iloc[idx]['Question'])
            
        return related_questions
