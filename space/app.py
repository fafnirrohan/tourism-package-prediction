import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

st.title("Tourism Wellness Package Predictor")

@st.cache_resource
def load_model():
    try:
        path = hf_hub_download(repo_id="rohanchemical/tourism-wellness-model", filename="best_model.joblib", token=None)
        return joblib.load(path)
    except Exception as e:
        st.error("Model not available in HF Hub: " + str(e))
        return None

model = None
st.info("This is a demo streamlit app. Replace inputs with real features from training dataset.")
if st.button("Load model"):
    model = load_model()
    if model:
        st.success("Model loaded (demo).")
