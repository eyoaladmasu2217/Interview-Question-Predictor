import joblib
import os
import numpy as np


class Predictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.cat_model = None
        self.diff_model = None
        self.prob_model = None
        self.nn_vectorizer = None
        self.nn_model = None
        self.df = None
        self.scaler = None
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

    # ------------------------------------------------------------------
    # Internal helper: compute confidence scores for any pipeline
    # ------------------------------------------------------------------
    @staticmethod
    def _confidence_scores(pipeline, X, top_k=3):
        """Return a dict {label: pct} for the top_k most probable classes."""
        clf = pipeline.named_steps['clf']
        tfidf = pipeline.named_steps['tfidf']
        X_vec = tfidf.transform(X)
        if hasattr(clf, 'predict_proba'):
            probs = clf.predict_proba(X_vec)[0]
            classes = clf.classes_
        elif hasattr(clf, 'decision_function'):
            scores = clf.decision_function(X_vec)[0]
            e_scores = np.exp(scores - np.max(scores))
            probs = e_scores / e_scores.sum()
            classes = clf.classes_
        else:
            return {}
        top_indices = np.argsort(probs)[::-1][:top_k]
        return {classes[i]: round(float(probs[i]) * 100, 1) for i in top_indices}

    # ------------------------------------------------------------------
    # Single-question prediction
    # ------------------------------------------------------------------
    def predict(self, question_text):
        """Predict category, difficulty and probability for one question."""
        if not self.cat_model or not self.diff_model or not self.prob_model:
            return None

        X = [question_text]

        category   = self.cat_model.predict(X)[0]
        difficulty = self.diff_model.predict(X)[0]
        probability = self.prob_model.predict(X)[0]

        cat_confidence  = self._confidence_scores(self.cat_model,  X, top_k=5)
        diff_confidence = self._confidence_scores(self.diff_model, X, top_k=5)

        return {
            'Category':            category,
            'Difficulty':          difficulty,
            'Probability':         probability,
            'CategoryConfidence':  cat_confidence,
            'DifficultyConfidence': diff_confidence,
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------
    def predict_batch(self, questions: list) -> list:
        """
        Run predictions on a list of question strings.

        Returns:
            A list of result dicts (same structure as predict()),
            with an additional 'Question' key echoing the input.
            Skips empty/invalid strings silently.
        """
        results = []
        for q in questions:
            if not q or not str(q).strip():
                continue
            res = self.predict(str(q).strip())
            if res:
                res['Question'] = str(q).strip()
                results.append(res)
        return results

    # ------------------------------------------------------------------
    # Model introspection
    # ------------------------------------------------------------------
    def get_model_info(self) -> dict:
        """Return a human-readable summary of the currently loaded models."""
        def _clf_name(pipeline):
            if pipeline is None:
                return 'Not loaded'
            clf = pipeline.named_steps.get('clf') or pipeline.named_steps.get('reg')
            return type(clf).__name__ if clf else 'Unknown'

        return {
            'Category Model':    _clf_name(self.cat_model),
            'Difficulty Model':  _clf_name(self.diff_model),
            'Probability Model': _clf_name(self.prob_model),
            'NN Model':          type(self.nn_model).__name__ if self.nn_model else 'Not loaded',
            'Dataset Size':      len(self.df) if self.df is not None else 0,
        }

    # ------------------------------------------------------------------
    # Related questions via nearest neighbours
    # ------------------------------------------------------------------
    def get_related_questions(self, question_text, top_k=20):
        """Return the top_k most similar questions from the training corpus."""
        if not self.nn_model or not self.nn_vectorizer or self.df is None:
            return []
        X_vec = self.nn_vectorizer.transform([question_text])
        distances, indices = self.nn_model.kneighbors(X_vec, n_neighbors=top_k)
        return [self.df.iloc[idx]['Question'] for idx in indices[0]]
