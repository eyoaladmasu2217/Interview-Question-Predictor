import pandas as pd
import numpy as np

def update_csv_with_probabilities(filepath):
    try:
        try:
            df = pd.read_csv(filepath, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(filepath, encoding='ISO-8859-1')
        
        print("Columns found:", df.columns)
        
        # Define base probabilities based on difficulty
        def get_base_prob(difficulty):
            d = str(difficulty).lower()
            if 'easy' in d:
                return 0.85
            elif 'medium' in d:
                return 0.60
            elif 'hard' in d:
                return 0.30
            else:
                return 0.50 # Default

        # Apply probabilities with some randomness
        np.random.seed(42)
        df['Probability'] = df['Difficulty'].apply(get_base_prob)
        
        # Add noise: +/- 0.05
        noise = np.random.uniform(-0.05, 0.05, size=len(df))
        df['Probability'] = df['Probability'] + noise
        
        # Clip to 0-1 range
        df['Probability'] = df['Probability'].clip(0.01, 0.99)
        
        # Round to 2 decimal places
        df['Probability'] = df['Probability'].round(2)
        
        # Save back
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"Successfully updated {filepath} with 'Probability' column.")
        print(df[['Question', 'Difficulty', 'Probability']].head())

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    update_csv_with_probabilities('data/Software Questions.csv')
