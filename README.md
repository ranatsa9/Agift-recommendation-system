<div align="center">

# 🎁 Giftly — Smart Gift Recommendation System

**Context-aware, budget-aware gift discovery for the Gulf market**

[![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.141-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.60-FF4B4B?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?logo=scikitlearn)](https://scikit-learn.org)

### [🚀 Open the live Giftly app](https://agift-recommendation-system.streamlit.app/)

[Live App](https://agift-recommendation-system.streamlit.app/) · [API Health](https://agift-recommendation-system-eta.vercel.app/health) · [API Docs](https://agift-recommendation-system-eta.vercel.app/docs) · [Results](#model-evaluation) · [Run Locally](#running-locally)

*Machine Learning Graduation Project — 2026*

</div>

---

## Overview

Giftly turns a short recipient profile into a ranked shortlist of real, purchasable gifts. Users provide the recipient's age, gift preference, occasion, maximum budget, and specific interests. The system returns relevant products with images, prices, match indicators, explanations, and retailer links.

The final application includes:

- A polished, responsive Streamlit interface
- A FastAPI recommendation service deployed on Vercel
- A v3 recommendation artifact covering **45,055 products**
- Specific interest filters layered over 15 learned interest families
- Hard age and maximum-budget constraints
- Context-aware relevance rules for ambiguous product keywords
- Budget-aware reranking that balances relevance and price proximity
- Product exploration, saved gifts, comparison, and model-insight pages
- English and Arabic interface support

```text
Example input
Age: 25 · Occasion: Graduation · Maximum budget: 400 SAR
Interest: Technology → Mobile Phones

Example output
Ranked, in-budget mobile phones with product images and purchase links
```

## Why this is not a classic recommender

The dataset has no user–item interaction history: no purchases, ratings, or repeat-user behaviour. Collaborative filtering would therefore have nothing meaningful to learn from. Giftly treats the task as **constrained matching and ranking**.

| Stage | Purpose |
|---|---|
| Hard filtering | Enforce age and maximum-budget constraints |
| Specific-interest gate | Reject contextually incorrect products before ranking |
| Model scoring | Combine interest, occasion, gender, text, price, and catalog-quality signals |
| Budget reranking | Prefer relevant products near 70% of the maximum budget without forcing luxury products |
| Diversification | Prevent one subcategory or repeated item from dominating the shortlist |

Relevance remains dominant in the final reranking: **80% recommendation score and 20% price proximity**. Price can reorder relevant products, but it cannot make an irrelevant item eligible.

## Final system architecture

```text
User
  │
  ▼
Streamlit Cloud
  │  recipient profile + selected interests
  ▼
FastAPI on Vercel
  │
  ├── hard age and budget filtering
  ├── specific-interest inclusion/exclusion gates
  ├── global PC-game isolation
  ├── v3 model scoring
  ├── budget-proximity reranking
  └── diversity reranking
  │
  ▼
Gift cards with images, prices, explanations, and retailer links
```

### Production services

| Service | URL |
|---|---|
| Streamlit interface | [agift-recommendation-system.streamlit.app](https://agift-recommendation-system.streamlit.app/) |
| FastAPI production domain | [agift-recommendation-system-eta.vercel.app](https://agift-recommendation-system-eta.vercel.app/) |
| Health check | [`/health`](https://agift-recommendation-system-eta.vercel.app/health) |
| Interactive API documentation | [`/docs`](https://agift-recommendation-system-eta.vercel.app/docs) |

## Recommendation quality safeguards

The final system adds a strict semantic layer based on a manual error audit across the interface's specific interests.

Examples include:

- PC and Steam games can appear only under **PC & Console Gaming**.
- `foundation` must have beauty context before qualifying as Makeup.
- `canvas` shoes and bags cannot qualify as Painting & Drawing.
- Lenovo Yoga laptops cannot qualify as Yoga & Pilates.
- Watch Dogs cannot qualify as Watches.
- MacBook sleeves and “bookish” clothing cannot qualify as Books & Novels.
- Shoe cabinets cannot qualify as Shoes & Sneakers.
- Garden-named jewellery and gardenia perfume cannot qualify as Gardening.
- A cosmetic “tea set” cannot qualify as Coffee & Tea.

The system intentionally returns fewer results when the catalog does not contain enough trustworthy matches.

## Model evaluation

The v3 artifact reports the following held-out ranking metrics:

| Metric | Result |
|---|---:|
| Precision@5 | **0.896** |
| NDCG@10 | **0.902** |
| Precision@1 | **0.985** |
| Catalogue coverage over 3,000 queries | **61.29%** |
| Constraint violations | **0%** |
| P95 latency in offline evaluation | **40 ms** |

The evaluation measures ranking quality on an independently annotated reference set. Live serverless latency may be higher during a Vercel cold start.

## Dataset

| Attribute | Value |
|---|---:|
| Raw rows | 56,280 |
| Clean parent products | **45,055** |
| Purchase options | 55,069 |
| Gulf retailers | 25 |
| Learned interest families | 15 |
| Occasion labels | 10 |
| Median product price | 379 SAR |

The catalog combines products collected from Gulf retailers such as Jarir, Decathlon, Sun & Sand Sports, Steam, Floward, LEGO, and others. Product images are loaded from their original URLs and are not redistributed by the application.

## Interface features

### Gift Finder

- Recipient age and gift preference
- Optional occasion
- Maximum budget in SAR
- Up to five specific interests
- Configurable recommendation count
- Best-match, price, and category controls
- “New picks” refresh while preserving the recipient profile

### Explore

- Browse the catalog without completing the recommendation form
- Filter by category, occasion, and price
- Multi-page product exploration

### Saved and Compare

- Save promising gifts during the session
- Compare up to three products side by side
- Review price, match, category, and brand

### Model Insights

- Product and feature counts
- Number of learned gift segments
- NDCG@10 ranking quality
- Explanation of the filtering, scoring, and diversification pipeline

## API

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Service information |
| GET | `/health` | Service and model-version health check |
| GET | `/options` | Supported broad interests, occasions, and genders |
| POST | `/recommend` | Generate ranked gift recommendations |

Example request:

```bash
curl -X POST https://agift-recommendation-system-eta.vercel.app/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "age": 25,
    "gender": "Any",
    "occasion": "Graduation",
    "budget": 400,
    "interests": ["Technology"],
    "specific_interests": ["Mobile Phones"],
    "top_k": 5
  }'
```

## Project structure

```text
Agift-recommendation-system/
├── .streamlit/
│   └── config.toml                 Streamlit visual configuration
├── api/
│   └── index.py                    Vercel serverless entry point
├── configs/                        Taxonomy and preprocessing configuration
├── data/                           Processed data and annotations
├── docs/decisions/                 Architecture decision records
├── models/
│   └── gift_recommender_v3.joblib  Serialized v3 model artifact
├── notebooks/                      EDA, modelling, and evaluation notebooks
├── reports/                        EDA figures and evaluation outputs
├── src/                            Original reusable project modules
├── api_main.py                     Production FastAPI application
├── gift_engine.py                  Recommendation engine required by the artifact
├── text_block.py                   Character n-gram feature block
├── exposure.py                     Exposure and diversity utilities
├── streamlit_app.py                Production Streamlit interface
├── requirements.txt                Runtime dependencies
├── pyproject.toml                  Project metadata and dependency constraints
└── vercel.json                     Vercel routing configuration
```

## Running locally

### 1. Clone the final repository

```bash
git clone https://github.com/ranatsa9/Agift-recommendation-system.git
cd Agift-recommendation-system
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Start FastAPI

```bash
uvicorn api_main:app --reload --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 4. Start Streamlit in another terminal

```bash
streamlit run streamlit_app.py
```

Open [http://localhost:8501](http://localhost:8501).

In the Streamlit sidebar, enter:

```text
http://127.0.0.1:8000
```

and select **Check API connection**.

## Deployment notes

### Streamlit Community Cloud

- Repository: `ranatsa9/Agift-recommendation-system`
- Branch: `main`
- Main file: `streamlit_app.py`

Recommended secrets:

```toml
FASTAPI_URL = "https://agift-recommendation-system-eta.vercel.app"
GIFT_API_URL = "https://agift-recommendation-system-eta.vercel.app"
GIFT_RUNTIME = "fastapi"
```

### Vercel

The API is exposed through `api/index.py` and routed using `vercel.json`. Vercel Authentication must remain disabled for the production deployment so Streamlit Cloud can reach the API without a login cookie.

Always configure Streamlit with the stable production domain rather than a deployment-specific Vercel URL.

## Known limitations

1. The system has no behavioural interaction history, so the current solution is content- and rule-based rather than collaborative.
2. Some retailer image URLs may expire or block hotlinking; the interface displays a styled placeholder when an image cannot be retrieved.
3. Catalog categories contain noisy labels. The strict-interest layer mitigates known errors, but new ambiguous product names can still occur.
4. Vercel cold starts can make the first API request slower than subsequent requests.
5. Saved and compared products currently persist for the Streamlit session rather than a registered user account.

## Future work

- Collect anonymous clicks, saves, and outbound retailer visits
- Train a learning-to-rank model using real interaction outcomes
- Add multilingual semantic embeddings for Arabic and English product text
- Monitor incorrect-result reports and update relevance rules automatically
- Add persistent user accounts and saved collections
- Add retailer availability and price-change monitoring

## Legal and ethical notes

- Giftly processes no payments and stores no sensitive personal data.
- Product names, images, descriptions, and trademarks belong to their respective retailers.
- Images are referenced from original retailer URLs for academic, non-commercial use.
- Gender is treated as a soft signal and never as a hard eligibility filter.

---

<div align="center">

Built with 🎁 for the Gulf market

*Machine Learning Graduation Project — 2026*

</div>
