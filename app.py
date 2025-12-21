import streamlit as st
import os
from src.predict import Predictor
from src.train import train_models

# Page Config
st.set_page_config(
    page_title="Interview Predictor and Reccomendation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Dashboard Look
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global Reset */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1f2937;
        background-color: #f3f4f6;
    }
    
    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    
    .main-header h1 {
        color: white !important;
        font-weight: 700;
        margin: 0;
        font-size: 2.25rem;
    }
    
    .main-header p {
        color: #e5e7eb;
        margin-top: 0.5rem;
        font-size: 1.1rem;
    }
    
    /* Card Styling */
    .dashboard-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06);
        height: 100%;
        border: 1px solid #e5e7eb;
        transition: transform 0.2s;
    }
    
    .dashboard-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05);
    }
    
    /* Metric Styling */
    .metric-label {
        font-size: 0.875rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #6b7280;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 1.875rem;
        font-weight: 700;
        color: #111827;
    }
    
    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
    }
    
    .badge-easy { background-color: #d1fae5; color: #065f46; }
    .badge-medium { background-color: #ffedd5; color: #9a3412; }
    .badge-hard { background-color: #fee2e2; color: #991b1b; }
    
    .badge-category { background-color: #e0f2fe; color: #075985; }
    
    /* Progress Bar Container */
    .progress-container {
        width: 100%;
        background-color: #e5e7eb;
        border-radius: 9999px;
        height: 12px;
        margin-top: 10px;
    }
    
    .progress-bar {
        height: 100%;
        border-radius: 9999px;
        transition: width 0.5s ease-in-out;
    }
    
    /* Related Questions */
    .related-card {
        background: #f9fafb;
        border: 1px solid #e5e7eb;
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.95rem;
        color: #374151;
        display: flex;
        align-items: center;
    }
    
    .related-number {
        background: #e5e7eb;
        color: #4b5563;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 12px;
        flex-shrink: 0;
    }
    
    /* Button Styling */
    .stButton>button {
        background-color: #2563eb;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        border-radius: 8px;
        border: none;
        box-shadow: 0 4px 6px -1px rgba(37, 99, 235, 0.2);
    }
    
    .stButton>button:hover {
        background-color: #1d4ed8;
        box-shadow: 0 6px 8px -1px rgba(37, 99, 235, 0.3);
    }
    
    /* Input Area */
    .stTextArea textarea {
        border-radius: 8px;
        border: 1px solid #d1d5db;
        padding: 1rem;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea:focus {
        border-color: #3b82f6;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.info("Model Status: Active")
    if st.button("Retrain System"):
        with st.spinner("Optimizing models..."):
            train_models()
        st.success("System Updated")
    
    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    # Placeholder stats - in a real app, read from df
    st.metric("Total Questions", "201")
    st.metric("Categories", "5")

# Header
st.markdown("""
    <div class="main-header">
        <h1>Interview Predictor and Reccomendation</h1>
        <p>AI analytics for software engineering interview preparation.</p>
    </div>
""", unsafe_allow_html=True)

# Main Layout
col_input, col_results = st.columns([1, 1.2])

with col_input:
    st.markdown("### Input Analysis")
    question = st.text_area("Enter technical question", height=200, placeholder="e.g., Explain the difference between REST and GraphQL...")
    
    analyze_btn = st.button("Generate Analysis", use_container_width=True)

if analyze_btn:
    if not question:
        with col_results:
            st.warning("Please enter a question to generate analysis.")
    else:
        predictor = Predictor()
        if not predictor.cat_model:
             st.error("System initialization required. Please retrain models.")
        else:
            with st.spinner("Running inference..."):
                result = predictor.predict(question)
            
            with col_results:
                st.markdown("###  Analysis Results")
                
                # Metrics Row
                m1, m2 = st.columns(2)
                
                # Category Card
                with m1:
                    st.markdown(f"""
                    <div class="dashboard-card">
                        <div class="metric-label">Category</div>
                        <div class="metric-value" style="font-size: 1.25rem;">{result['Category']}</div>
                        <div style="margin-top: 8px;"><span class="badge badge-category">Technical</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Difficulty Card
                diff_color = 'badge-easy' if result['Difficulty'] == 'Easy' else 'badge-medium' if result['Difficulty'] == 'Medium' else 'badge-hard'
                with m2:
                    st.markdown(f"""
                    <div class="dashboard-card">
                        <div class="metric-label">Difficulty</div>
                        <div class="metric-value">{result['Difficulty']}</div>
                        <div style="margin-top: 8px;"><span class="badge {diff_color}">Level</span></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Probability Section
                prob_score = result['Probability'] * 100
                prob_color = '#22c55e' if prob_score > 70 else '#f59e0b' if prob_score > 40 else '#ef4444'
                
                st.markdown(f"""
                <div class="dashboard-card" style="margin-top: 1rem;">
                    <div class="metric-label">Appearance Probability</div>
                    <div style="display: flex; justify-content: space-between; align-items: end;">
                        <div class="metric-value">{prob_score:.1f}%</div>
                        <div style="color: {prob_color}; font-weight: 600;">{
                            'High Likelihood' if prob_score > 70 else 'Moderate' if prob_score > 40 else 'Low Likelihood'
                        }</div>
                    </div>
                    <div class="progress-container">
                        <div class="progress-bar" style="width: {prob_score}%; background-color: {prob_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # Related Questions
                st.markdown("###  Similar Questions")
                related = predictor.get_related_questions(question)
                
                if related:
                    for idx, q in enumerate(related, 1):
                        st.markdown(f"""
                        <div class="related-card">
                            <div class="related-number">{idx}</div>
                            <div>{q}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("No similar questions found in database.")

else:
    with col_results:
        st.info(" Enter a question and click 'Generate Analysis' to see results.")
