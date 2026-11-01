import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fake_news_pipeline.joblib"
METRICS_PATH = BASE_DIR / "results" / "metrics.json"

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
)

st.title("📰 Fake News Detection Using NLP")
st.write(
    "Enter a news article or headline to classify it using a machine-learning "
    "model trained on the supplied Fake and Real News dataset."
)
st.info(
    "Research prototype: this model learns patterns in its training dataset. "
    "It does not independently verify claims or establish whether a story is true."
)

if not MODEL_PATH.exists():
    st.warning(
        "The trained model has not been created yet. Follow the setup instructions "
        "in README.md, place Fake.csv and True.csv in the dataset folder, and run "
        "`python train_model.py`."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()

left, right = st.columns([2, 1])
with left:
    article = st.text_area(
        "Paste a news article or headline",
        height=260,
        placeholder="Paste the news text here...",
    )
with right:
    st.subheader("How to use")
    st.markdown(
        "1. Paste the article text.\n"
        "2. Select **Analyze news**.\n"
        "3. Review the model prediction and scores.\n\n"
        "For more reliable research, use the full article rather than a very short headline."
    )

if st.button("Analyze news", type="primary", use_container_width=True):
    if len(article.strip()) < 20:
        st.error("Please enter at least 20 characters so the model has some text to analyze.")
    else:
        prediction = int(model.predict([article])[0])
        label = "REAL" if prediction == 1 else "FAKE"

        st.divider()
        st.subheader("Prediction")
        if label == "REAL":
            st.success("The model classifies this text as: REAL")
        else:
            st.error("The model classifies this text as: FAKE")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([article])[0]
            classes = list(model.classes_)
            scores = {
                "FAKE": float(probabilities[classes.index(0)]) if 0 in classes else 0.0,
                "REAL": float(probabilities[classes.index(1)]) if 1 in classes else 0.0,
            }
            c1, c2 = st.columns(2)
            c1.metric("FAKE model score", f"{scores['FAKE']:.1%}")
            c2.metric("REAL model score", f"{scores['REAL']:.1%}")
            st.caption(
                "These are model scores/probabilities for the dataset labels, "
                "not a verified probability that the claim is true or false."
            )

        st.caption(f"Input length: {len(article.split())} words")

st.divider()
st.subheader("Model evaluation")
if METRICS_PATH.exists():
    try:
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        st.write(f"Selected model: **{metrics.get('best_model', 'Unknown')}**")
        if metrics.get("comparison"):
            frame = pd.DataFrame(metrics["comparison"])
            st.dataframe(frame, use_container_width=True, hide_index=True)
        matrix_path = BASE_DIR / "results" / "confusion_matrix.png"
        if matrix_path.exists():
            st.image(str(matrix_path), caption="Confusion matrix for the selected model")
    except Exception as exc:
        st.warning(f"Could not read evaluation results: {exc}")
else:
    st.caption("Evaluation results will appear here after model training.")
