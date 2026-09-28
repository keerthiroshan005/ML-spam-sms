from pathlib import Path
import joblib
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent / "models" / "spam_classifier.joblib"

st.set_page_config(page_title="SMS Shield", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)

model = load_model()

st.title("🛡️ SMS Shield")
st.subheader("Machine Learning Spam Message Detector")
st.write("Enter an SMS message and the trained NLP classifier will predict whether it is **Spam** or **Ham (Legitimate)**.")

if model is None:
    st.error("Model not found. Run `python download_dataset.py` followed by `python train.py` first.")
    st.stop()

message = st.text_area("SMS message", height=160, placeholder="Example: Congratulations! You have won a free prize...")

if st.button("Analyze Message", type="primary", use_container_width=True):
    if not message.strip():
        st.warning("Please enter a message.")
    else:
        pred = int(model.predict([message])[0])
        proba = float(model.predict_proba([message])[0, 1]) if hasattr(model, "predict_proba") else None
        if pred == 1:
            st.error("Prediction: SPAM")
            if proba is not None: st.metric("Spam probability", f"{proba:.1%}")
        else:
            st.success("Prediction: HAM / LEGITIMATE")
            if proba is not None: st.metric("Spam probability", f"{proba:.1%}")

with st.expander("Project details"):
    st.markdown("""
    **Pipeline:** text cleaning → TF-IDF word + character n-grams → Logistic Regression.

    **Improvement:** compared with a Naive Bayes unigram baseline, the final model uses word bigrams, character n-grams, sublinear TF-IDF, class balancing, and regularized Logistic Regression.
    """)
