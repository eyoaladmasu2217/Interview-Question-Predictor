# Interview Question Predictor 

A Machine Learning application that predicts the **Category**, **Difficulty**, and **Probability** of software engineering interview questions. Built with Python, Scikit-learn, and Streamlit.

##  Features
- **Predict Category**: Classifies questions into topics like "Arrays", "System Design", "General Programming", etc.
- **Predict Difficulty**: Estimates if a question is Easy, Medium, or Hard.
- **Predict Probability**: Calculates the likelihood of a question appearing in an interview.
- **Interactive UI**: Simple and clean web interface using Streamlit.
- **Custom Training**: Easily retrain the model with your own dataset.

## Tech Stack
- **Python 3.x**
- **Streamlit** (Frontend)
- **Scikit-learn** (Machine Learning)
- **Pandas** (Data Manipulation)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/eyoaladmasu2217/interview-predictor.git
   cd interview-predictor
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the application:
   ```bash
   streamlit run app.py
   ```

2. Open your browser to the local URL provided (usually `http://localhost:8501`).

3. Enter an interview question and click **Predict**!

## Dataset
The model is trained on a CSV dataset containing software engineering interview questions. You can add your own data to `data/Software Questions.csv` and click "Retrain Models" in the app sidebar.

## License
MIT
