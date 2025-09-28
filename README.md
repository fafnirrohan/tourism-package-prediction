# tourism-package-prediction

This repository contains the end-to-end MLOps project for Tourism Package Prediction.

- `src/models/train.py` — training + MLflow logging
- `src/deployment/push_model_to_hf.py` — helper to push model to Hugging Face
- `.github/workflows/pipeline.yml` — CI pipeline to train & push
- `app.py` — Streamlit demo (for HF Space)
- `Dockerfile`, `requirements.txt` — deployment artifacts

Replace placeholders and ensure `HF_TOKEN` and/or `GITHUB_TOKEN` are set in CI or local env.
