import streamlit as st
import os
from src.predict import Predictor
from src.train import train_models

# Page Config
st.set_page_config(
    page_title="Interview Question Predictor",
    page_icon="🔮",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 16px;
    }
    .stTextArea>div>div>textarea {
        border-radius: 10px;
        border: 1px solid #ddd;
    }
    .prediction-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin-top: 20px;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
    }
    .metric-value {
        font-size: 24px;
        font-weight: bold;
        color: #333;
    }
    </style>
    """, unsafe_allow_html=True)

# Header
st.title("🔮 Interview Question Predictor")
st.markdown("Enter a software engineering interview question below to predict its **Category**, **Difficulty**, and **Probability** of appearing.")

# Sidebar for training
with st.sidebar:
    st.header("Settings")
    if st.button("Retrain Models"):
        with st.spinner("Training models..."):
            train_models()
        st.success("Models trained successfully!")

# Main Input
question = st.text_area("Enter Question:", height=150, placeholder="e.g., Explain the difference between a process and a thread.")

if st.button("Predict"):
    if question:
        predictor = Predictor()
        if not predictor.cat_model:
             st.error("Models not found. Please click 'Retrain Models' in the sidebar first.")
        else:
            with st.spinner("Analyzing..."):
                result = predictor.predict(question)
            
            # Display Results
            st.markdown(f"""
            <div class="prediction-card">
                <div style="display: flex; justify-content: space-around; text-align: center;">
                    <div>
                        <div class="metric-label">Category</div>
                        <div class="metric-value" style="color: #2196F3;">{result['Category']}</div>
                    </div>
                    <div>
                        <div class="metric-label">Difficulty</div>
                        <div class="metric-value" style="color: {
                            '#4CAF50' if result['Difficulty'] == 'Easy' else 
                            '#FF9800' if result['Difficulty'] == 'Medium' else 
                            '#F44336'
                        };">{result['Difficulty']}</div>
                    </div>
                    <div>
                        <div class="metric-label">Probability</div>
                        <div class="metric-value">{result['Probability']:.1%}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("Please enter a question first.")

# Footer
st.markdown("---")
st.markdown("*Powered by Scikit-learn & Streamlit*")
