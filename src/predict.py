import joblib
import os

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

        # Models expect an iterable of strings
        X = [question_text]

        category = self.cat_model.predict(X)[0]
        difficulty = self.diff_model.predict(X)[0]
        probability = self.prob_model.predict(X)[0]

        return {
            'Category': category,
            'Difficulty': difficulty,
            'Probability': probability
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
