import io
import streamlit as st
import pandas as pd
import os
from src.predict import Predictor
from src.train import train_models
from src.utils import validate_question, clean_text, extract_keywords, detect_question_type

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Interview Predictor and Recommendation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
        color: #1f2937;
        background-color: #f3f4f6;
    }

    /* Header */
    .main-header {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1), 0 2px 4px -1px rgba(0,0,0,0.06);
    }
    .main-header h1 { color: white !important; font-weight: 700; margin: 0; font-size: 2.25rem; }
    .main-header p  { color: #e5e7eb; margin-top: 0.5rem; font-size: 1.1rem; }

    /* Cards */
    .dashboard-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px 0 rgba(0,0,0,0.1), 0 1px 2px 0 rgba(0,0,0,0.06);
        height: 100%;
        border: 1px solid #e5e7eb;
        transition: transform 0.2s;
    }
    .dashboard-card:hover { transform: translateY(-2px); box-shadow: 0 10px 15px -3px rgba(0,0,0,0.1); }

    /* Metrics */
    .metric-label {
        font-size: 0.875rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 0.05em;
        color: #6b7280; margin-bottom: 0.5rem;
    }
    .metric-value { font-size: 1.875rem; font-weight: 700; color: #111827; }

    /* Badges */
    .badge { display: inline-block; padding: 0.25rem 0.75rem; border-radius: 9999px; font-size: 0.875rem; font-weight: 500; }
    .badge-easy       { background-color: #d1fae5; color: #065f46; }
    .badge-medium     { background-color: #ffedd5; color: #9a3412; }
    .badge-hard       { background-color: #fee2e2; color: #991b1b; }
    .badge-category   { background-color: #e0f2fe; color: #075985; }
    .badge-type       { background-color: #f3e8ff; color: #6b21a8; }
    .badge-keyword    { background-color: #fef9c3; color: #713f12; margin: 2px; display: inline-block; padding: 2px 10px; border-radius: 9999px; font-size: 0.8rem; }

    /* Progress bar */
    .progress-container { width: 100%; background-color: #e5e7eb; border-radius: 9999px; height: 12px; margin-top: 10px; }
    .progress-bar       { height: 100%; border-radius: 9999px; transition: width 0.5s ease-in-out; }

    /* Related question cards */
    .related-card {
        background: #f9fafb; border: 1px solid #e5e7eb; border-radius: 8px;
        padding: 1rem; margin-bottom: 0.75rem; font-size: 0.95rem; color: #374151;
        display: flex; align-items: center;
    }
    .related-number {
        background: #e5e7eb; color: #4b5563; width: 24px; height: 24px;
        border-radius: 50%; display: flex; align-items: center; justify-content: center;
        font-size: 0.75rem; font-weight: 600; margin-right: 12px; flex-shrink: 0;
    }

    /* Buttons */
    .stButton>button {
        background-color: #2563eb; color: white; font-weight: 600;
        padding: 0.75rem 1.5rem; border-radius: 8px; border: none;
        box-shadow: 0 4px 6px -1px rgba(37,99,235,0.2);
    }
    .stButton>button:hover { background-color: #1d4ed8; box-shadow: 0 6px 8px -1px rgba(37,99,235,0.3); }

    /* Text area */
    .stTextArea textarea { border-radius: 8px; border: 1px solid #d1d5db; padding: 1rem; font-family: 'Inter', sans-serif; }
    .stTextArea textarea:focus { border-color: #3b82f6; box-shadow: 0 0 0 2px rgba(59,130,246,0.2); }

    /* Batch result table tweaks */
    .batch-result-row { background: white; border-radius: 8px; padding: 0.75rem 1rem; margin-bottom: 0.5rem; border: 1px solid #e5e7eb; font-size: 0.9rem; }
    </style>
    """, unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")
    st.info("Model Status: Active")
    if st.button("Retrain System"):
        with st.spinner("Optimising models…"):
            train_models()
        st.success("System Updated")

    st.markdown("---")
    st.markdown("### 📊 Dataset Stats")
    @st.cache_data(show_spinner=False)
    def load_dataset():
        return pd.read_csv('data/Software Questions.csv', encoding='utf-8')

    try:
        _df = load_dataset()
        st.metric("Total Questions",  len(_df))
        st.metric("Categories",       _df['Category'].nunique()   if 'Category'   in _df.columns else "–")
        st.metric("Difficulty Levels", _df['Difficulty'].nunique() if 'Difficulty' in _df.columns else "–")
    except Exception:
        st.metric("Total Questions",  "–")
        st.metric("Categories",       "–")

    # Model info panel
    st.markdown("---")
    st.markdown("### 🤖 Active Models")
    try:
        _pred = Predictor()
        info = _pred.get_model_info()
        for k, v in info.items():
            st.caption(f"**{k}**: {v}")
    except Exception:
        st.caption("Models not loaded yet.")

    # JSON metrics (if available)
    if os.path.exists('model_metrics.json'):
        import json
        st.markdown("---")
        st.markdown("### 📈 Training Metrics")
        with open('model_metrics.json') as _jf:
            _m = json.load(_jf)
        st.caption(f"Category winner: **{_m['category']['winner']}**")
        st.caption(f"Difficulty winner: **{_m['difficulty']['winner']}**")
        st.caption(f"Prob. MSE: **{_m['probability_mse']}**")

    # Recent prediction history
    if st.session_state['history']:
        st.markdown("---")
        st.markdown("### 🕑 Recent Predictions")
        for item in reversed(st.session_state['history'][-5:]):
            st.markdown(
                f"<small><b>{item['category']}</b> · {item['difficulty']} · "
                f"{item['prob']:.0f}% prob</small><br>"
                f"<small style='color:#6b7280'>{item['question'][:60]}…</small>",
                unsafe_allow_html=True
            )
            st.markdown("<hr style='margin:6px 0; border-color:#e5e7eb'>", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
    <div class="main-header">
        <h1>Interview Predictor and Recommendation</h1>
        <p>AI-powered analytics for software engineering interview preparation.</p>
    </div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_single, tab_batch = st.tabs(["🔍 Single Analyser", "📋 Batch Analyser"])


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 1 – Single Question Analyser
# ═══════════════════════════════════════════════════════════════════════════════
with tab_single:
    col_input, col_results = st.columns([1, 1.2])

    with col_input:
        st.markdown("### Input Analysis")
        question = st.text_area(
            "Enter technical question", height=200,
            placeholder="e.g., Explain the difference between REST and GraphQL…"
        )
        analyze_btn = st.button("Generate Analysis", use_container_width=True, key="single_analyze")

    if analyze_btn:
        is_valid, error_msg = validate_question(question)
        if not is_valid:
            with col_results:
                st.warning(f"⚠️ {error_msg}")
        else:
            predictor = Predictor()
            if not predictor.cat_model:
                st.error("System initialisation required. Please retrain models.")
            else:
                cleaned_question = clean_text(question)
                with st.spinner("Running inference…"):
                    result = predictor.predict(cleaned_question)

                with col_results:
                    st.markdown("### Analysis Results")

                    # ── Question Type badge ───────────────────────────────────
                    q_type = detect_question_type(question)
                    st.markdown(
                        f"<div style='margin-bottom:1rem'>"
                        f"Question type: <span class='badge badge-type'>💡 {q_type}</span>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

                    # ── Category & Difficulty cards ───────────────────────────
                    m1, m2 = st.columns(2)

                    with m1:
                        conf_html = ""
                        if result.get('CategoryConfidence'):
                            conf_html = "<div style='margin-top:10px'>"
                            for cat, pct in result['CategoryConfidence'].items():
                                bar_color  = '#3b82f6' if cat == result['Category'] else '#d1d5db'
                                text_color = '#1e3a8a' if cat == result['Category'] else '#6b7280'
                                conf_html += f"""
                                <div style='margin-bottom:5px'>
                                    <div style='display:flex;justify-content:space-between;font-size:0.75rem;color:{text_color};font-weight:500'>
                                        <span>{cat}</span><span>{pct}%</span>
                                    </div>
                                    <div style='background:#e5e7eb;border-radius:999px;height:6px;margin-top:2px'>
                                        <div style='width:{pct}%;background:{bar_color};height:100%;border-radius:999px'></div>
                                    </div>
                                </div>"""
                            conf_html += "</div>"
                        st.markdown(f"""
                        <div class="dashboard-card">
                            <div class="metric-label">Category</div>
                            <div class="metric-value" style="font-size:1.25rem">{result['Category']}</div>
                            <div style="margin-top:8px"><span class="badge badge-category">Technical</span></div>
                            {conf_html}
                        </div>
                        """, unsafe_allow_html=True)

                    with m2:
                        diff_color = ('badge-easy'   if result['Difficulty'] == 'Easy'
                                 else 'badge-medium' if result['Difficulty'] == 'Medium'
                                 else 'badge-hard')
                        # Difficulty confidence mini-bars
                        diff_conf_html = ""
                        if result.get('DifficultyConfidence'):
                            diff_conf_html = "<div style='margin-top:10px'>"
                            for lvl, pct in result['DifficultyConfidence'].items():
                                bar_color  = '#6366f1' if lvl == result['Difficulty'] else '#d1d5db'
                                text_color = '#3730a3' if lvl == result['Difficulty'] else '#6b7280'
                                diff_conf_html += f"""
                                <div style='margin-bottom:5px'>
                                    <div style='display:flex;justify-content:space-between;font-size:0.75rem;color:{text_color};font-weight:500'>
                                        <span>{lvl}</span><span>{pct}%</span>
                                    </div>
                                    <div style='background:#e5e7eb;border-radius:999px;height:6px;margin-top:2px'>
                                        <div style='width:{pct}%;background:{bar_color};height:100%;border-radius:999px'></div>
                                    </div>
                                </div>"""
                            diff_conf_html += "</div>"
                        st.markdown(f"""
                        <div class="dashboard-card">
                            <div class="metric-label">Difficulty</div>
                            <div class="metric-value">{result['Difficulty']}</div>
                            <div style="margin-top:8px"><span class="badge {diff_color}">Level</span></div>
                            {diff_conf_html}
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Probability bar ───────────────────────────────────────
                    prob_score = result['Probability'] * 100
                    prob_color = '#22c55e' if prob_score > 70 else '#f59e0b' if prob_score > 40 else '#ef4444'
                    prob_label = 'High Likelihood' if prob_score > 70 else 'Moderate' if prob_score > 40 else 'Low Likelihood'

                    st.markdown(f"""
                    <div class="dashboard-card" style="margin-top:1rem">
                        <div class="metric-label">Appearance Probability</div>
                        <div style="display:flex;justify-content:space-between;align-items:end">
                            <div class="metric-value">{prob_score:.1f}%</div>
                            <div style="color:{prob_color};font-weight:600">{prob_label}</div>
                        </div>
                        <div class="progress-container">
                            <div class="progress-bar" style="width:{prob_score}%;background-color:{prob_color}"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    # ── Keyword Highlights ────────────────────────────────────
                    keywords = extract_keywords(question, top_n=7)
                    if keywords:
                        kw_chips = "".join(
                            f"<span class='badge-keyword'>🔑 {kw}</span>" for kw in keywords
                        )
                        st.markdown(f"""
                        <div class="dashboard-card" style="margin-top:1rem">
                            <div class="metric-label">Key Concepts Detected</div>
                            <div style="margin-top:8px">{kw_chips}</div>
                        </div>
                        """, unsafe_allow_html=True)

                    # ── Save to history ───────────────────────────────────────
                    st.session_state['history'].append({
                        'question':   question[:80],
                        'category':   result['Category'],
                        'difficulty': result['Difficulty'],
                        'prob':       prob_score,
                    })

                    # ── Export single result as CSV ───────────────────────────
                    export_df = pd.DataFrame([{
                        'Question':            question,
                        'Category':            result['Category'],
                        'Difficulty':          result['Difficulty'],
                        'Probability (%)':     round(prob_score, 2),
                        'Question Type':       q_type,
                        'Keywords':            ', '.join(keywords),
                    }])
                    csv_bytes = export_df.to_csv(index=False).encode('utf-8')
                    json_bytes = export_df.to_json(orient="records").encode('utf-8')
                    
                    e_col1, e_col2 = st.columns(2)
                    with e_col1:
                        st.download_button(
                            label="⬇️ Export Result as CSV",
                            data=csv_bytes,
                            file_name="prediction_result.csv",
                            mime="text/csv",
                            key="export_single_csv",
                            use_container_width=True
                        )
                    with e_col2:
                        st.download_button(
                            label="⬇️ Export Result as JSON",
                            data=json_bytes,
                            file_name="prediction_result.json",
                            mime="application/json",
                            key="export_single_json",
                            use_container_width=True
                        )

                    # ── Similar Questions ─────────────────────────────────────
                    st.markdown("### 🔗 Similar Questions")
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
            st.info("Enter a question and click 'Generate Analysis' to see results.")


# ═══════════════════════════════════════════════════════════════════════════════
#  TAB 2 – Batch Analyser
# ═══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("### 📋 Batch Question Analyser")
    st.markdown(
        "Upload a CSV file with a **Question** column, or paste multiple questions "
        "(one per line) below. The system will predict Category, Difficulty, and "
        "Probability for each question and let you download the results."
    )

    b_col1, b_col2 = st.columns([1, 1])

    with b_col1:
        uploaded_file = st.file_uploader("Upload CSV (must have a 'Question' column)", type=["csv"])

    with b_col2:
        pasted_text = st.text_area(
            "Or paste questions here (one per line)", height=180,
            placeholder="What is a binary search tree?\nExplain recursion.\nDesign a URL shortener."
        )

    run_batch = st.button("▶ Run Batch Analysis", use_container_width=True, key="batch_analyze")

    if run_batch:
        questions_list = []

        # From uploaded CSV
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                if 'Question' not in batch_df.columns:
                    st.error("CSV must contain a 'Question' column.")
                else:
                    questions_list = batch_df['Question'].dropna().tolist()
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")

        # From pasted text (fallback / addition)
        if pasted_text.strip():
            pasted_qs = [ln.strip() for ln in pasted_text.strip().splitlines() if ln.strip()]
            questions_list.extend(pasted_qs)

        if not questions_list:
            st.warning("Please upload a CSV or paste at least one question.")
        else:
            predictor = Predictor()
            if not predictor.cat_model:
                st.error("Models not loaded. Please retrain from the sidebar first.")
            else:
                with st.spinner(f"Analysing {len(questions_list)} question(s)…"):
                    # Validate then predict
                    valid_qs, invalid_qs = [], []
                    for q in questions_list:
                        ok, _ = validate_question(q)
                        if ok:
                            valid_qs.append(clean_text(q))
                        else:
                            invalid_qs.append(q)

                    raw_results = predictor.predict_batch(valid_qs)

                if invalid_qs:
                    st.warning(f"⚠️ Skipped {len(invalid_qs)} invalid / too-short question(s).")

                if raw_results:
                    rows = []
                    for r in raw_results:
                        prob_pct = r['Probability'] * 100
                        rows.append({
                            'Question':        r['Question'],
                            'Category':        r['Category'],
                            'Difficulty':      r['Difficulty'],
                            'Probability (%)': round(prob_pct, 2),
                            'Question Type':   detect_question_type(r['Question']),
                            'Keywords':        ', '.join(extract_keywords(r['Question'], top_n=5)),
                        })

                    results_df = pd.DataFrame(rows)

                    st.success(f"✅ Analysed **{len(results_df)}** question(s) successfully.")

                    # Summary stats
                    s1, s2, s3 = st.columns(3)
                    with s1:
                        top_cat = results_df['Category'].value_counts().idxmax()
                        st.metric("Top Category", top_cat)
                    with s2:
                        avg_prob = results_df['Probability (%)'].mean()
                        st.metric("Avg Probability", f"{avg_prob:.1f}%")
                    with s3:
                        top_diff = results_df['Difficulty'].value_counts().idxmax()
                        st.metric("Most Common Difficulty", top_diff)

                    # Results table
                    st.dataframe(results_df, use_container_width=True, height=350)

                    # Export
                    csv_out = results_df.to_csv(index=False).encode('utf-8')
                    json_out = results_df.to_json(orient="records").encode('utf-8')
                    
                    e_col1, e_col2 = st.columns(2)
                    with e_col1:
                        st.download_button(
                            label="⬇️ Download Full Results as CSV",
                            data=csv_out,
                            file_name="batch_predictions.csv",
                            mime="text/csv",
                            key="export_batch_csv",
                            use_container_width=True
                        )
                    with e_col2:
                        st.download_button(
                            label="⬇️ Download Full Results as JSON",
                            data=json_out,
                            file_name="batch_predictions.json",
                            mime="application/json",
                            key="export_batch_json",
                            use_container_width=True
                        )
                else:
                    st.info("No valid results were produced. Check your input questions.")
