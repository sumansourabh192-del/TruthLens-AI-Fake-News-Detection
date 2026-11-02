 
import streamlit as st
import joblib
from datetime import datetime

# Load ML model
model = joblib.load("model/fake_news_model.pkl")

st.set_page_config(
    page_title="TruthLens | AI News Intelligence",
    page_icon="T",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

if "last_result" not in st.session_state:
    st.session_state.last_result = None

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #f5f7fb;
}

[data-testid="stSidebar"] {
    background: #111827;
}

[data-testid="stSidebar"] * {
    color: #e5e7eb !important;
}

[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #cbd5e1 !important;
}

.brand {
    font-size: 27px;
    font-weight: 800;
    color: white;
    padding: 18px 0 4px 0;
}

.brand-sub {
    color: #94a3b8;
    font-size: 12px;
    margin-bottom: 30px;
}

.page-title {
    font-size: 32px;
    font-weight: 800;
    color: #111827;
    margin-bottom: 4px;
}

.page-subtitle {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 25px;
}

.metric-card {
    background: white;
    padding: 22px;
    border: 1px solid #e5eaf2;
    border-radius: 13px;
    min-height: 112px;
}

.metric-label {
    color: #64748b;
    font-size: 13px;
    font-weight: 600;
}

.metric-value {
    color: #111827;
    font-size: 28px;
    font-weight: 800;
    margin-top: 9px;
}

.panel {
    background: white;
    border: 1px solid #e5eaf2;
    border-radius: 14px;
    padding: 24px;
    margin-top: 15px;
}

.result-fake {
    background: #fff1f2;
    border: 1px solid #fecdd3;
    border-radius: 12px;
    padding: 20px;
    color: #be123c;
    font-size: 23px;
    font-weight: 800;
}

.result-real {
    background: #ecfdf5;
    border: 1px solid #a7f3d0;
    border-radius: 12px;
    padding: 20px;
    color: #047857;
    font-size: 23px;
    font-weight: 800;
}

div.stButton > button {
    background: #2563eb;
    color: white;
    border: none;
    border-radius: 9px;
    height: 46px;
    font-weight: 700;
}

div.stButton > button:hover {
    background: #1d4ed8;
    color: white;
}

textarea {
    border-radius: 10px !important;
}

hr {
    border-color: #e5eaf2;
}
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown('<div class="brand">? TRUTHLENS</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="brand-sub">? AI NEWS INTELLIGENCE PLATFORM</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "??  Dashboard",
            "??  Analyze News",
            "??  Prediction History",
            "??  About Project"
        ],
        label_visibility="visible"
    )

    st.markdown("---")
    st.markdown("### ?? SYSTEM INFORMATION")
    st.caption("?? TF-IDF + Logistic Regression")
    st.caption("?? Model Accuracy: 98.99%")
    st.caption("?? Version 1.0")

# Statistics
history = st.session_state.history
total = len(history)
fake_total = sum(x["prediction"] == "Fake News" for x in history)
real_total = total - fake_total

# Dashboard
if page == "??  Dashboard":

    st.markdown('<div class="page-title">Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Monitor your news analysis activity and model performance.</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)

    metrics = [
        (c1, "MODEL ACCURACY", "98.99%"),
        (c2, "ARTICLES ANALYZED", str(total)),
        (c3, "FAKE NEWS", str(fake_total)),
        (c4, "REAL NEWS", str(real_total))
    ]

    for col, label, value in metrics:
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Welcome to TruthLens")
    st.write(
        "Analyze news articles using a machine learning model "
        "trained to identify patterns associated with fake and real news."
    )
    st.write("Select **Analyze News** from the sidebar to get started.")
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Recent Activity")

    if history:
        for item in history[:5]:
            st.write(
                f"**{item['prediction']}**  |  "
                f"{item['confidence']:.2f}% confidence  |  {item['time']}"
            )
            st.divider()
    else:
        st.info("No analysis activity yet.")
    st.markdown('</div>', unsafe_allow_html=True)

# Analyze News
elif page == "??  Analyze News":

    st.markdown('<div class="page-title">News Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Submit an article for machine learning classification.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Article Input")

    news = st.text_area(
        "News article",
        height=250,
        placeholder="Paste the complete news article here...",
        label_visibility="collapsed"
    )

    if st.button("Analyze Article", use_container_width=True):

        if news.strip():

            prediction = int(model.predict([news])[0])
            probabilities = model.predict_proba([news])[0]
            confidence = float(max(probabilities) * 100)

            result = "Fake News" if prediction == 0 else "Real News"

            record = {
                "article": news[:200],
                "prediction": result,
                "confidence": confidence,
                "time": datetime.now().strftime("%d %b %Y, %I:%M %p")
            }

            st.session_state.history.insert(0, record)
            st.session_state.last_result = record

        else:
            st.warning("Please enter a news article.")

    st.markdown('</div>', unsafe_allow_html=True)

    result = st.session_state.last_result

    if result:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Classification Result")

        if result["prediction"] == "Fake News":
            st.markdown(
                '<div class="result-fake">FAKE NEWS DETECTED</div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                '<div class="result-real">REAL NEWS DETECTED</div>',
                unsafe_allow_html=True
            )

        st.write("")
        st.metric("Prediction Confidence", f"{result['confidence']:.2f}%")
        st.progress(result["confidence"] / 100)

        st.caption(
            "Model confidence is not proof of factual accuracy. "
            "Verify important claims using reliable sources."
        )
        st.markdown('</div>', unsafe_allow_html=True)

# Prediction History
elif page == "??  Prediction History":

    st.markdown('<div class="page-title">Prediction History</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Review previously analyzed news articles.</div>',
        unsafe_allow_html=True
    )

    if history:

        st.metric("Total Articles Analyzed", total)

        for i, item in enumerate(history, start=1):
            with st.expander(
                f"{i}. {item['prediction']} | {item['confidence']:.2f}%"
            ):
                st.write("**Article Preview:**")
                st.write(item["article"] + "...")
                st.write("**Prediction:**", item["prediction"])
                st.write("**Confidence:**", f"{item['confidence']:.2f}%")
                st.write("**Date:**", item["time"])

        if st.button("Clear All History"):
            st.session_state.history = []
            st.session_state.last_result = None
            st.rerun()

    else:
        st.info("No prediction history available. Analyze an article first.")

# About
elif page == "??  About Project":

    st.markdown('<div class="page-title">About TruthLens</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="page-subtitle">Machine Learning based news classification platform.</div>',
        unsafe_allow_html=True
    )

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Project Overview")

    st.write("""
    TruthLens is a Fake News Detection application that uses
    Natural Language Processing and Machine Learning to classify
    news articles into Fake News and Real News categories.
    """)

    st.subheader("Technology Stack")

    st.write("""
    - Python
    - Pandas and NumPy
    - Scikit-learn
    - TF-IDF Vectorization
    - Logistic Regression
    - Streamlit
    """)

    st.subheader("Model Information")
    st.write("Training articles: 39,105")
    st.write("Evaluation accuracy: 98.99% on the held-out dataset.")

    st.info(
        "The model identifies patterns in its training dataset. "
        "It does not independently verify whether a news claim is true."
    )

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.subheader("Developer Profile")

    st.markdown("## Suman Sourabh")
    st.write("Machine Learning and Python Developer")

    st.markdown(
        '<p><strong>Email:</strong> '
        '<a href="mailto:sumansourabh192@gmail.com">'
        'sumansourabh192@gmail.com</a></p>',
        unsafe_allow_html=True
    )

    st.write("Project: TruthLens - AI News Intelligence Platform")
    st.write("Technologies: Python, Scikit-learn, TF-IDF, Streamlit")

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("---")
st.caption("TruthLens | AI News Intelligence | Developed by Suman Sourabh")
