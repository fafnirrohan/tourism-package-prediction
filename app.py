import streamlit as st
import joblib
from huggingface_hub import hf_hub_download

st.title('Tourism Wellness Package Predictor')

@st.cache_resource
def load_model():
    path = hf_hub_download(repo_id='rohanchemical/tourism-wellness-model', filename='best_model.joblib', token='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')
    return joblib.load(path)

model = load_model()
st.write('Model loaded. Replace the form below with real inputs to predict.')

if st.button('Predict sample'):
    st.write('Placeholder prediction')