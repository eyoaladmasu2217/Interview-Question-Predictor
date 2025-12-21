import joblib
import os

class Predictor:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.cat_model = None
        self.diff_model = None
        self.prob_model = None
        self.load_models()

    def load_models(self):
        try:
            self.cat_model = joblib.load(os.path.join(self.model_dir, 'category_model.pkl'))
            self.diff_model = joblib.load(os.path.join(self.model_dir, 'difficulty_model.pkl'))
            self.prob_model = joblib.load(os.path.join(self.model_dir, 'probability_model.pkl'))
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
