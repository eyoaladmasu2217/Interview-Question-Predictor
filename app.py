import streamlit as st
import os
from src.predict import Predictor
from src.train import train_models

# Page Config
st.set_page_config(
    page_title="Interview Question Predictor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif;
        color: #333;
    }
    
    .stApp {
        background-color: #f8f9fa;
    }
    
    h1, h2, h3 {
        color: #1a237e; /* Navy Blue */
        font-weight: 500;
    }
    
    .stButton>button {
        background-color: #1a237e;
        color: white;
        border: none;
        border-radius: 4px;
        padding: 10px 20px;
        font-weight: 500;
        transition: background-color 0.3s;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #0d47a1;
    }
    
    .stTextArea>div>div>textarea {
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 10px;
    }
    
    .prediction-container {
        display: flex;
        gap: 20px;
        margin-top: 20px;
        margin-bottom: 30px;
    }
    
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        flex: 1;
        text-align: center;
        border-top: 4px solid #1a237e;
    }
    
    .metric-label {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #666;
        margin-bottom: 8px;
    }
    
    .metric-value {
        font-size: 24px;
        font-weight: 700;
        color: #333;
    }
    
    .related-questions-box {
        background-color: white;
        padding: 25px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-top: 20px;
    }
    
    .related-item {
        padding: 12px 0;
        border-bottom: 1px solid #eee;
        font-size: 15px;
    }
    
    .related-item:last-child {
        border-bottom: none;
    }
    
    .sidebar-content {
        padding: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("Control Panel")
    st.markdown("Manage your model training and settings here.")
    st.markdown("---")
    if st.button("Retrain Models"):
        with st.spinner("Training models..."):
            train_models()
        st.success("Models trained successfully.")

# Main Content
st.title("Interview Question Predictor")
st.markdown("### Analyze software engineering interview questions with AI.")

col1, col2 = st.columns([2, 1])

with col1:
    question = st.text_area("Input Question", height=150, placeholder="Type your interview question here...")
    
    if st.button("Analyze Question"):
        if question:
            predictor = Predictor()
            if not predictor.cat_model:
                 st.error("Models not found. Please retrain models from the sidebar.")
            else:
                with st.spinner("Processing..."):
                    result = predictor.predict(question)
                
                # Display Metrics
                st.markdown(f"""
                <div class="prediction-container">
                    <div class="metric-card">
                        <div class="metric-label">Category</div>
                        <div class="metric-value">{result['Category']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Difficulty</div>
                        <div class="metric-value" style="color: {
                            '#2e7d32' if result['Difficulty'] == 'Easy' else 
                            '#f57c00' if result['Difficulty'] == 'Medium' else 
                            '#c62828'
                        };">{result['Difficulty']}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Probability</div>
                        <div class="metric-value">{result['Probability']:.1%}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Related Questions
                st.markdown("### Related Questions")
                related_questions = predictor.get_related_questions(question)
                
                st.markdown('<div class="related-questions-box">', unsafe_allow_html=True)
                if related_questions:
                    for idx, q in enumerate(related_questions, 1):
                        st.markdown(f'<div class="related-item"><strong>{idx}.</strong> {q}</div>', unsafe_allow_html=True)
                else:
                    st.info("No related questions found.")
                st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.warning("Please enter a question to analyze.")

with col2:
    st.markdown("""
    ### About
    This tool uses machine learning to classify interview questions and estimate their difficulty and likelihood of appearing in interviews.
    
    **Features:**
    - Category Classification
    - Difficulty Estimation
    - Probability Scoring
    - Similar Question Retrieval
    """)
