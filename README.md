# Giftly — Gift Recommendation System App

Deployment repository for the Giftly machine-learning application.

## Live architecture

```text
Streamlit interface -> FastAPI service -> trained recommendation model
```

## Features

- Context-aware gift recommendations using age, preference, occasion, budget
  and specific interests.
- Public FastAPI recommendation runtime with a cached local artifact for catalog browsing and model insights.
- Real product images and retailer links.
- Explore catalog with pagination and new picks.
- Saved shortlist and side-by-side gift comparison.
- Recommendation explanations and model-insight dashboard.
- English and Arabic interface support.

## Repository contents

- `streamlit_app.py` — Streamlit frontend.
- `app.py` — Vercel ASGI entry point.
- `api_main.py` — FastAPI endpoints.
- `gift_engine.py` — recommendation and ranking engine.
- `models/gift_recommender_v3.joblib` — v3 text-enhanced production model (stored with Git LFS).
- `gift_engine.py`, `text_block.py`, `exposure.py` — v3 ranking and text-feature runtime modules.
- `.streamlit/config.toml` — Streamlit visual theme.

## Run locally

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Terminal 1:

```powershell
uvicorn api_main:app --reload --port 8000
```

Terminal 2:

```powershell
$env:GIFT_RUNTIME = "fastapi"
$env:GIFT_API_URL = "http://127.0.0.1:8000"
streamlit run streamlit_app.py
```

## Deploy

### Streamlit Community Cloud

- Main file: `streamlit_app.py`
- Recommended fallback runtime: `Model`

### Vercel

- Framework preset: `Other`
- Root directory: `./`
- Python entry point: `app.py`
- Health endpoint: `/health`
- Interactive documentation: `/docs`

After Vercel deployment, configure Streamlit secrets:

```toml
GIFT_RUNTIME = "fastapi"
GIFT_API_URL = "https://your-vercel-project.vercel.app"
```

## Development and research repository

The full team repository containing data preparation, EDA, modeling notebooks,
reports and project history is available at:

[Mhd366/gift-recommendation-system](https://github.com/Mhd366/gift-recommendation-system)

This application was produced as a collaborative academic machine-learning
project. Product data belongs to the respective retailers.
