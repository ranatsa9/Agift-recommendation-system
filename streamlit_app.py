from __future__ import annotations

import base64
import html
import os
import re
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import joblib
import requests
import streamlit as st
from PIL import Image, ImageChops


st.set_page_config(
    page_title="Giftly | Thoughtful gift recommendations",
    page_icon="🎁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

def resolve_api_url() -> str:
    """Use Streamlit Cloud secrets first, then environment/local defaults."""
    try:
        secret_url = (
            st.secrets.get("FASTAPI_URL", "")
            or st.secrets.get("GIFT_API_URL", "")
        )
    except (FileNotFoundError, KeyError):
        secret_url = ""

    return str(
        secret_url
        or os.getenv("FASTAPI_URL")
        or os.getenv("GIFT_API_URL")
        or "https://agift-recommendation-system-pol9rq02r-g2-e3d4.vercel.app"
    ).strip().rstrip("/")


DEFAULT_API_URL = resolve_api_url()
DEFAULT_MODEL_PATH = os.getenv(
    "GIFT_MODEL_PATH", "models/gift_recommender_v1.joblib"
)
DEFAULT_DEMO_MODE = os.getenv("GIFT_DEMO_MODE", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
DEFAULT_OPTIONS = {
    "interests": [
        "Art & Creativity",
        "Beauty & Grooming",
        "Cooking & Food",
        "Fashion & Style",
        "Fragrance",
        "Gaming",
        "Gardening & Nature",
        "Home & Interiors",
        "Jewellery & Watches",
        "Kids & Play",
        "Outdoors & Travel",
        "Reading & Learning",
        "Sports & Fitness",
        "Technology",
        "Wellness",
    ],
    "occasions": [
        "Birthday",
        "Graduation",
        "Anniversary",
        "Wedding",
        "Eid",
        "MothersDay",
        "FathersDay",
        "NewBaby",
        "Housewarming",
        "ThankYou",
    ],
    "genders": ["Female", "Male", "Any"],
}

# The model was trained on 15 broad interest families. These more specific
# choices make the questionnaire feel personal while mapping safely back to
# the feature names the trained recommender understands.
SPECIFIC_INTERESTS = {
    "Painting & Drawing": "Art & Creativity",
    "Photography": "Art & Creativity",
    "Crafts & DIY": "Art & Creativity",
    "Music & Instruments": "Art & Creativity",
    "Skincare": "Beauty & Grooming",
    "Makeup": "Beauty & Grooming",
    "Haircare": "Beauty & Grooming",
    "Nail Care": "Beauty & Grooming",
    "Cooking": "Cooking & Food",
    "Coffee & Tea": "Cooking & Food",
    "Chocolate & Gourmet Food": "Cooking & Food",
    "Fashion Accessories": "Fashion & Style",
    "Handbags & Purses": "Fashion & Style",
    "Wallets & Card Holders": "Fashion & Style",
    "Shoes & Sneakers": "Fashion & Style",
    "Luxury Fragrances": "Fragrance",
    "Home Fragrances & Candles": "Fragrance",
    "PC & Console Gaming": "Gaming",
    "Board Games & Puzzles": "Gaming",
    "Gardening": "Gardening & Nature",
    "Plants & Flowers": "Gardening & Nature",
    "Home Décor": "Home & Interiors",
    "Kitchen & Dining": "Home & Interiors",
    "Home Organisation": "Home & Interiors",
    "Jewellery": "Jewellery & Watches",
    "Watches": "Jewellery & Watches",
    "Baby Gifts": "Kids & Play",
    "Toys & Collectibles": "Kids & Play",
    "Camping & Hiking": "Outdoors & Travel",
    "Travel Accessories": "Outdoors & Travel",
    "Beach & Picnic": "Outdoors & Travel",
    "Books & Novels": "Reading & Learning",
    "Journaling & Stationery": "Reading & Learning",
    "Courses & Learning": "Reading & Learning",
    "Gym & Strength Training": "Sports & Fitness",
    "Running": "Sports & Fitness",
    "Football": "Sports & Fitness",
    "Yoga & Pilates": "Sports & Fitness",
    "Mobile Phones": "Technology",
    "Phone Accessories": "Technology",
    "Computers & Accessories": "Technology",
    "Audio & Headphones": "Technology",
    "Self-Care & Relaxation": "Wellness",
    "Spa & Massage": "Wellness",
    "Mindfulness & Sleep": "Wellness",
}

DEMO_RECOMMENDATIONS = [
    {
        "parent_id": "demo-1",
        "product_name": "Signature Fragrance Gift Set",
        "brand": "Maison Collection",
        "category": "Beauty",
        "sub_category": "Fragrance",
        "price_min": 320,
        "price_median": 390,
        "match_score": 94.2,
        "interest_similarity": 0.93,
        "budget_fit": 0.97,
        "product_url": "https://example.com",
    },
    {
        "parent_id": "demo-2",
        "product_name": "Personalized Keepsake Box",
        "brand": "The Gift Studio",
        "category": "Home",
        "sub_category": "Keepsakes",
        "price_min": 245,
        "price_median": 295,
        "match_score": 91.8,
        "interest_similarity": 0.89,
        "budget_fit": 0.95,
        "product_url": "https://example.com",
    },
    {
        "parent_id": "demo-3",
        "product_name": "Premium Self-Care Ritual",
        "brand": "Calm & Co.",
        "category": "Wellness",
        "sub_category": "Self-care",
        "price_min": 280,
        "price_median": 350,
        "match_score": 89.4,
        "interest_similarity": 0.88,
        "budget_fit": 0.93,
        "product_url": "https://example.com",
    },
]


st.markdown(
    """
    <style>
      @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');
      :root { --ink:#442d37; --muted:#806873; --surface:#f8f0f3; --surface-2:#f1e3e9;
              --app-a:#f5edef; --app-b:#f0e4e9; --plum:#b95f84; --plum-2:#cf83a1;
              --blush:#ead5de; --gold:#ad567b; --gold-soft:#efdbe3;
              --line:rgba(151,87,113,.19); --shadow:rgba(91,55,70,.13); }
      .stApp { background:
          radial-gradient(circle at 8% 4%, rgba(224,190,204,.52), transparent 26rem),
          radial-gradient(circle at 94% 20%, rgba(217,174,193,.42), transparent 24rem),
          linear-gradient(145deg,var(--app-a),var(--app-b)); color:var(--ink); }
      .stApp:before { content:""; position:fixed; width:28rem; height:28rem; right:-8rem; bottom:-10rem;
                      border-radius:50%; background:linear-gradient(135deg,rgba(240,143,184,.24),rgba(214,95,145,.20));
                      filter:blur(18px); pointer-events:none; }
      html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
      .block-container { max-width:1240px; padding-top:4.75rem; padding-bottom:4rem; }
      h1, h2, h3 { font-family:'Playfair Display', serif !important; color:var(--ink) !important; }
      header[data-testid="stHeader"] { background:rgba(245,237,239,.95); border-bottom:1px solid var(--line); }
      header[data-testid="stHeader"] button, header[data-testid="stHeader"] svg,
      header[data-testid="stHeader"] a { color:var(--ink) !important; fill:currentColor !important; }
      [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] label,
      [data-testid="stSlider"] p, [data-testid="stCaptionContainer"] p {
        color:var(--muted) !important;
      }
      [data-testid="stAlert"] { background:var(--surface-2) !important; border:1px solid var(--line) !important; }
      [data-testid="stAlert"] p { color:var(--muted) !important; }
      .topbar { display:flex; align-items:center; justify-content:space-between; margin:0 0 1.1rem; }
      .brand { font-weight:700; letter-spacing:.01em; color:var(--plum); font-size:1.25rem; }
      .brand-dot { color:var(--gold); }
      .top-pill { padding:.42rem .76rem; border:1px solid rgba(255,255,255,.52); border-radius:99px;
                  color:var(--muted); background:rgba(255,255,255,.38); backdrop-filter:blur(16px);
                  -webkit-backdrop-filter:blur(16px); box-shadow:inset 0 1px 0 rgba(255,255,255,.6);
                  font-size:.76rem; }
      .hero { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:32px; padding:3.6rem 3.2rem 2.6rem;
              background:linear-gradient(125deg,rgba(250,244,246,.76),rgba(235,213,222,.58),rgba(226,193,207,.44));
              backdrop-filter:blur(24px) saturate(135%); -webkit-backdrop-filter:blur(24px) saturate(135%);
              box-shadow:0 24px 70px var(--shadow); }
      .gift-art { position:absolute; z-index:0; right:2.8rem; top:1rem; width:275px; height:310px;
                  pointer-events:none; opacity:.90; }
      .cascade-card { position:absolute; display:flex; align-items:flex-end; box-sizing:border-box;
                  padding:.75rem .9rem; border-radius:17px; color:rgba(255,255,255,.84);
                  border:1px solid rgba(255,255,255,.34); font-size:.56rem; font-weight:800;
                  letter-spacing:.16em; box-shadow:0 15px 34px rgba(0,0,0,.18),
                  inset 0 1px 0 rgba(255,255,255,.30); backdrop-filter:blur(8px); }
      .cascade-card:after { content:""; position:absolute; right:14px; top:14px; width:18px; height:13px;
                  border:1px solid rgba(255,255,255,.50); border-radius:4px; }
      .cascade-one { width:205px; height:112px; right:0; top:-2.2rem; transform:rotate(9deg);
                  background:linear-gradient(135deg,rgba(154,111,194,.82),rgba(202,105,151,.76)); opacity:.62; }
      .cascade-two { width:205px; height:112px; right:2.8rem; top:1.1rem; transform:rotate(-6deg);
                  background:linear-gradient(135deg,#d8a6d0,#bd73a7); }
      .cascade-three { width:156px; height:84px; right:.7rem; top:9.2rem; transform:rotate(5deg);
                  background:linear-gradient(135deg,rgba(186,145,210,.86),rgba(143,104,182,.84)); opacity:.82; }
      .cascade-four { width:112px; height:62px; right:4.1rem; top:15.1rem; transform:rotate(-7deg);
                  background:linear-gradient(135deg,rgba(225,164,197,.82),rgba(190,107,154,.78)); opacity:.70; }
      .cascade-four:after { display:none; }
      .gift-shadow { position:absolute; right:3rem; top:18rem; width:130px; height:16px;
                  border-radius:50%; background:rgba(0,0,0,.18); filter:blur(10px); }
      .eyebrow { color:var(--plum-2); text-transform:uppercase; letter-spacing:.17em; font-weight:700; font-size:.72rem; }
      .hero h1 { position:relative; z-index:1; font-size:clamp(2.7rem,5vw,4.9rem); line-height:1.01;
                 margin:.65rem 0 1rem; max-width:790px; letter-spacing:-.035em; }
      .hero h1 em { color:var(--plum-2); font-style:italic; }
      .hero p { position:relative; z-index:1; color:var(--muted); font-size:1.06rem; max-width:650px; margin:0; line-height:1.7; }
      .hero-proof { position:relative; z-index:1; display:flex; flex-wrap:wrap; gap:.65rem; margin-top:1.55rem; }
      .page-hero { border:1px solid rgba(255,255,255,.54); border-radius:26px; padding:2.15rem 2.35rem;
                   background:rgba(247,239,242,.62); backdrop-filter:blur(22px) saturate(115%);
                   -webkit-backdrop-filter:blur(22px) saturate(135%); box-shadow:0 16px 44px var(--shadow);
                   margin-bottom:1.5rem; }
      .page-hero h1 { font-size:clamp(2.2rem,4vw,3.5rem); margin:.3rem 0 .45rem; }
      .page-hero p { color:var(--muted); margin:0; max-width:720px; line-height:1.65; }
      .proof-chip { display:inline-flex; align-items:center; gap:.38rem; padding:.46rem .72rem; border-radius:99px;
                    background:rgba(255,255,255,.42); border:1px solid rgba(255,255,255,.64);
                    backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px);
                    color:var(--plum); font-size:.75rem; font-weight:600; }
      .section-kicker { text-transform:uppercase; letter-spacing:.12em; color:var(--gold); font-weight:700; font-size:.72rem; }
      div[data-testid="stForm"] { background:rgba(247,239,242,.64); border:1px solid rgba(255,255,255,.54);
              border-radius:24px; padding:1.25rem 1.35rem 1.4rem; box-shadow:0 16px 42px var(--shadow),
              inset 0 1px 0 rgba(255,255,255,.72); backdrop-filter:blur(24px) saturate(140%);
              -webkit-backdrop-filter:blur(24px) saturate(140%); }
      div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input { border-radius:12px !important; }
      .stButton > button, .stFormSubmitButton > button { width:100%; border-radius:13px; min-height:3rem; border:none;
          background:linear-gradient(135deg,var(--plum),var(--plum-2)); color:white; font-weight:700;
          box-shadow:0 8px 22px rgba(216,101,152,.24); transition:all .2s ease; }
      .stButton > button:hover, .stFormSubmitButton > button:hover { background:linear-gradient(135deg,#c94f85,#df80a8);
          color:white; transform:translateY(-1px); box-shadow:0 11px 28px rgba(216,101,152,.29); }
      .product-card { background:rgba(247,239,242,.64); border:1px solid rgba(255,255,255,.54);
                      border-radius:22px; padding:1rem; box-shadow:0 14px 38px var(--shadow),
                      inset 0 1px 0 rgba(255,255,255,.72); height:425px; box-sizing:border-box; margin-bottom:.65rem;
                      backdrop-filter:blur(22px) saturate(140%); -webkit-backdrop-filter:blur(22px) saturate(140%);
                      display:flex; flex-direction:column;
                      transition:transform .28s ease, box-shadow .28s ease, border-color .28s ease;
                      overflow:hidden; transform-origin:center; }
      .product-card:hover { transform:translateY(-9px) scale(1.018);
                      border-color:rgba(185,95,132,.46);
                      box-shadow:0 24px 55px rgba(91,55,70,.22), 0 0 0 3px rgba(185,95,132,.08); }
      .product-image-frame { position:relative; width:100%; height:225px; overflow:hidden; border-radius:14px;
                      background:var(--surface-2); isolation:isolate; }
      .product-image-backdrop { position:absolute; inset:-18px; width:calc(100% + 36px); height:calc(100% + 36px);
                      object-fit:cover; filter:blur(16px) saturate(.78); opacity:.48; transform:scale(1.08); }
      .product-image { position:relative; z-index:1; width:100%; height:100%; object-fit:contain;
                      object-position:center; display:block; transition:transform .35s ease, filter .35s ease; }
      .product-card:hover .product-image { transform:scale(1.045); filter:saturate(1.06) contrast(1.02); }
      .placeholder { position:relative; height:225px; border-radius:14px; display:flex; align-items:center;
                     justify-content:center; flex-direction:column; gap:.5rem; overflow:hidden;
                     background:radial-gradient(circle at 20% 15%,rgba(255,255,255,.72),transparent 34%),
                                linear-gradient(145deg,#efd9e2,#e2c0cf); }
      .placeholder:before,.placeholder:after { content:""; position:absolute; border-radius:50%;
                     border:1px solid rgba(185,95,132,.18); }
      .placeholder:before { width:120px; height:120px; right:-45px; top:-38px; }
      .placeholder:after { width:85px; height:85px; left:-32px; bottom:-28px; }
      .placeholder-gift { position:relative; width:43px; height:33px; margin:12px 0 11px;
                     border:1.5px solid var(--plum); border-radius:6px 6px 9px 9px;
                     background:rgba(255,255,255,.18); opacity:.55;
                     transition:transform .3s ease, opacity .3s ease; }
      .placeholder-gift:before { content:""; position:absolute; left:-5px; right:-5px; top:-6px; height:8px;
                     border:1.5px solid var(--plum); border-radius:5px; background:rgba(255,255,255,.30); }
      .placeholder-gift:after { content:""; position:absolute; left:18px; top:-6px; width:6px; height:38px;
                     background:var(--plum); opacity:.55; }
      .placeholder-gift i:before,.placeholder-gift i:after { content:""; position:absolute; top:-20px;
                     width:20px; height:14px; border:2px solid var(--plum); }
      .placeholder-gift i:before { right:20px; border-radius:16px 4px 16px 4px; transform:rotate(12deg); }
      .placeholder-gift i:after { left:20px; border-radius:4px 16px 4px 16px; transform:rotate(-12deg); }
      .product-card:hover .placeholder-gift { transform:translateY(-2px) scale(1.04); opacity:.72; }
      .placeholder strong { color:var(--ink); font-size:.82rem; letter-spacing:.02em; }
      .placeholder small { font-family:'DM Sans',sans-serif; font-size:.64rem; text-transform:uppercase;
                           letter-spacing:.12em; color:var(--muted); font-weight:700; }
      .score { display:inline-block; color:var(--plum); background:var(--blush); border-radius:99px; padding:.28rem .58rem;
               font-size:.76rem; font-weight:700; margin-top:.8rem; }
      .category { color:var(--gold); text-transform:uppercase; letter-spacing:.08em; font-size:.68rem; font-weight:700; margin-top:.85rem; }
      .product-title { font-weight:700; font-size:1.03rem; line-height:1.28; margin:.35rem 0; min-height:2.65rem; }
      .product-meta { color:var(--muted); font-size:.82rem; }
      .price { font-size:1.02rem; font-weight:700; color:var(--plum); margin-top:.65rem; }
      .why { color:var(--muted); font-size:.78rem; border-top:1px solid #e2d4da; padding-top:.65rem; margin-top:auto; }
      .reason-tags { display:flex; flex-wrap:wrap; gap:.28rem; border-top:1px solid var(--line);
                     padding-top:.62rem; margin-top:auto; }
      .reason-tags span { padding:.24rem .45rem; border-radius:999px; background:var(--surface-2);
                     color:var(--muted); font-size:.66rem; font-weight:600; }
      .status { border-radius:16px; padding:.8rem 1rem; background:rgba(255,255,255,.40);
                border:1px solid rgba(255,255,255,.65); backdrop-filter:blur(16px);
                color:var(--plum); font-size:.88rem; }
      .results-note { border-left:3px solid var(--gold); padding:.1rem 0 .1rem .85rem; color:var(--muted);
                      font-size:.86rem; margin:.25rem 0 1rem; }
      .explore-intro { max-width:720px; color:var(--muted); line-height:1.65; margin:-.25rem 0 1.15rem; }
      div[data-testid="stSegmentedControl"] { margin:.15rem 0 1.25rem; }
      div[data-testid="stSegmentedControl"] > div { padding:.35rem; border:1px solid var(--line);
             border-radius:16px; background:rgba(255,255,255,.42); backdrop-filter:blur(18px);
             -webkit-backdrop-filter:blur(18px); box-shadow:0 8px 24px var(--shadow); }
      div[data-testid="stSegmentedControl"] button { border-radius:11px !important; font-weight:700 !important;
             color:var(--muted) !important; min-height:2.65rem; }
      div[data-testid="stSegmentedControl"] button[aria-pressed="true"] {
             color:white !important; background:linear-gradient(135deg,var(--plum),var(--plum-2)) !important;
             box-shadow:0 7px 18px rgba(216,101,152,.24); }
      div[data-baseweb="popover"], div[data-baseweb="menu"] {
              backdrop-filter:none !important; -webkit-backdrop-filter:none !important; }
      div[data-baseweb="menu"] { background:#17111f !important; }
      div[data-baseweb="menu"] li, div[data-baseweb="menu"] li * {
              color:#f7f1ff !important; font-weight:600 !important; opacity:1 !important;
              filter:none !important; text-shadow:none !important; transform:none !important; }
      .insight-grid { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:1rem; margin:.5rem 0 2rem; }
      .insight-card { padding:1.2rem 1.25rem; border:1px solid var(--line); border-radius:18px;
              background:rgba(255,255,255,.44); box-shadow:0 12px 30px var(--shadow);
              backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px); }
      .insight-card small { color:var(--muted); text-transform:uppercase; letter-spacing:.1em;
              font-weight:700; font-size:.7rem; }
      .insight-card strong { display:block; color:var(--ink); font:700 1.8rem 'Playfair Display',serif;
              margin:.3rem 0 .1rem; }
      .insight-card span { color:var(--muted); font-size:.78rem; }
      .model-flow { display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:.65rem; margin:.7rem 0 2.2rem; }
      .flow-step { position:relative; min-height:150px; padding:1.05rem; border-radius:17px;
              border:1px solid var(--line); background:linear-gradient(145deg,rgba(255,255,255,.56),rgba(252,234,242,.54)); }
      .flow-number { display:inline-grid; place-items:center; width:28px; height:28px; border-radius:9px;
              color:white; background:linear-gradient(135deg,var(--plum),var(--plum-2));
              font-size:.75rem; font-weight:800; }
      .flow-step b { display:block; color:var(--ink); margin:.75rem 0 .35rem; }
      .flow-step p { color:var(--muted); font-size:.82rem; line-height:1.45; margin:0; }
      .segment-card { min-height:185px; padding:1.15rem; margin-bottom:1rem; border:1px solid var(--line);
              border-radius:18px; background:rgba(255,255,255,.42); box-shadow:0 10px 26px var(--shadow); }
      .segment-card .segment-size { color:var(--plum); font-weight:800; font-size:.74rem;
              text-transform:uppercase; letter-spacing:.08em; }
      .segment-card h4 { color:var(--ink); font:700 1.15rem 'Playfair Display',serif; margin:.55rem 0 .8rem; }
      .segment-chip { display:inline-block; padding:.3rem .55rem; margin:.15rem .15rem .15rem 0;
              border-radius:999px; color:var(--muted); background:var(--surface-2); font-size:.72rem; }
      .segment-price { color:var(--ink); font-weight:700; margin-top:.8rem; }
      .compare-wrap { overflow-x:auto; margin:.65rem 0 1.5rem; border:1px solid var(--line);
                      border-radius:18px; background:rgba(30,23,38,.58); }
      .compare-row { display:grid; grid-template-columns:minmax(260px,2.3fr) .72fr 1fr 1.35fr 1fr;
                     gap:1rem; align-items:center; min-width:850px; padding:.9rem 1rem;
                     border-bottom:1px solid var(--line); transition:background .2s ease; }
      .compare-row:last-child { border-bottom:0; }
      .compare-row:not(.compare-head):hover { background:rgba(200,173,235,.08); }
      .compare-head { background:rgba(200,173,235,.10); color:var(--muted);
                      text-transform:uppercase; letter-spacing:.09em; font-size:.66rem; font-weight:800; }
      .compare-name { color:var(--ink); font-weight:700; line-height:1.35; }
      .compare-sub { color:var(--muted); font-size:.72rem; margin-top:.2rem; }
      .compare-price { display:inline-block; width:max-content; padding:.32rem .58rem; border-radius:999px;
                       background:var(--surface-2); color:var(--plum); font-weight:800; font-size:.76rem; }
      .match-value { color:var(--ink); font-size:.76rem; font-weight:800; margin-bottom:.32rem; }
      .match-track { height:6px; border-radius:99px; overflow:hidden; background:rgba(255,255,255,.10); }
      .match-fill { height:100%; border-radius:inherit; background:linear-gradient(90deg,var(--plum),var(--plum-2)); }
      .compare-cell { color:var(--muted); font-size:.78rem; line-height:1.35; }
      footer { visibility:hidden; }
      @media (min-width:0px) {
        :root { --ink:#f7f1ff; --muted:#c8bbd5; --surface:#211928; --surface-2:#2e2338;
                --app-a:#15101c; --app-b:#211725; --plum:#c8adeb; --plum-2:#d9a8cf;
                --blush:#382b46; --gold:#e9a7c1; --gold-soft:#432d3b;
                --line:rgba(237,220,250,.15); --shadow:rgba(0,0,0,.34); }
        .stApp { background:linear-gradient(145deg,var(--app-a),var(--app-b)); }
        header[data-testid="stHeader"] { background:rgba(21,16,28,.95); }
        .hero { background:radial-gradient(circle at 90% 15%,rgba(233,167,193,.14),transparent 30%),
                           linear-gradient(130deg,rgba(49,36,64,.74),rgba(61,39,57,.64));
                           border-color:rgba(237,220,250,.16); }
        .top-pill,.proof-chip { background:rgba(255,255,255,.065); border-color:rgba(255,255,255,.14);
                                box-shadow:inset 0 1px 0 rgba(255,255,255,.10); }
        div[data-testid="stForm"], .product-card { background:rgba(46,35,56,.66);
                border-color:rgba(237,220,250,.15); box-shadow:0 18px 48px rgba(0,0,0,.30),
                inset 0 1px 0 rgba(255,255,255,.10); }
        .page-hero { background:rgba(46,35,56,.60); border-color:rgba(237,220,250,.15); }
        div[data-baseweb="select"] > div, div[data-testid="stNumberInput"] input,
        div[data-baseweb="input"] > div { background:var(--surface-2) !important; color:var(--ink) !important; }
        .placeholder { background:linear-gradient(135deg,#3a2d4a,#4b3042); }
        .score { background:#433455; color:#e1cff5; }
        .why { border-color:var(--line); }
        .status { background:rgba(57,43,70,.70); border-color:rgba(237,220,250,.14); color:#e1cff5; }
        div[data-testid="stSegmentedControl"] > div { background:rgba(46,35,56,.62);
                border-color:rgba(237,220,250,.15); }
        .insight-card,.segment-card { background:rgba(46,35,56,.62); }
        .flow-step { background:linear-gradient(145deg,rgba(59,44,71,.72),rgba(46,35,56,.60)); }
      }
      @media(max-width:700px) {
        .block-container { padding-top:4.25rem; }
        .top-pill { display:none; }
        .hero { padding:2.2rem 1.4rem; border-radius:20px; }
        .gift-art { right:-2.5rem; top:.5rem; transform:scale(.62); transform-origin:top center; opacity:.32; }
        .product-image,.placeholder { height:210px; }
        .insight-grid { grid-template-columns:repeat(2,minmax(0,1fr)); }
        .model-flow { grid-template-columns:1fr; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


def safe_text(value: Any, fallback: str = "") -> str:
    if value is None:
        return fallback
    return html.escape(str(value))


def first_present(item: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = item.get(key)
        if value not in (None, "", "nan"):
            return value
    return None


def image_url(item: dict[str, Any]) -> str | None:
    value = first_present(
        item,
        ["image_url", "image_link", "image", "img_url", "product_image", "thumbnail"],
    )
    if not isinstance(value, str):
        return None
    value = value.strip()
    known_placeholders = (
        "/placeholder.",
        "loadingimages",
        "/online-only_new",
        "online_exclusive",
        "online-exclusive",
        "onlineexclusive",
        "online%20exclusive",
        "image-unavailable",
        "image_unavailable",
        "no-image",
        "no_image",
    )
    if any(pattern in value.lower() for pattern in known_placeholders):
        return None
    return value if value.startswith(("http://", "https://", "data:image/")) else None


@st.cache_data(ttl=3600, show_spinner=False)
def cached_image_source(url: str) -> str | None:
    """Fetch and normalize a retailer image into a consistent square canvas."""
    if url.startswith("data:image/"):
        return url
    try:
        response = requests.get(
            url,
            timeout=4,
            headers={"User-Agent": "Mozilla/5.0 Giftly/1.0"},
        )
        response.raise_for_status()
        content_type = response.headers.get("content-type", "").split(";")[0].lower()
        if not content_type.startswith("image/") or len(response.content) > 8_000_000:
            return None
        image = Image.open(BytesIO(response.content)).convert("RGB")
        width, height = image.size

        # Product-only images often include a large, flat white border. When all
        # four corners are similar, trim that border and rebuild a square canvas.
        corners = [
            image.getpixel((0, 0)),
            image.getpixel((width - 1, 0)),
            image.getpixel((0, height - 1)),
            image.getpixel((width - 1, height - 1)),
        ]
        corner_spread = max(
            max(abs(a[channel] - b[channel]) for channel in range(3))
            for a in corners
            for b in corners
        )
        background = tuple(sum(pixel[channel] for pixel in corners) // 4 for channel in range(3))
        if corner_spread < 24:
            difference = ImageChops.difference(
                image,
                Image.new("RGB", image.size, background),
            ).convert("L")
            mask = difference.point(lambda value: 255 if value > 18 else 0)
            bounds = mask.getbbox()
            if bounds:
                left, top, right, bottom = bounds
                pad_x = max(8, int((right - left) * 0.10))
                pad_y = max(8, int((bottom - top) * 0.10))
                bounds = (
                    max(0, left - pad_x),
                    max(0, top - pad_y),
                    min(width, right + pad_x),
                    min(height, bottom + pad_y),
                )
                image = image.crop(bounds)

        # Keep the retailer's original background. CSS fills the common frame;
        # this step only removes excessive flat margins around the product.
        max_side = 1100
        if max(image.size) > max_side:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
        output = BytesIO()
        image.save(output, format="JPEG", quality=90, optimize=True)
        encoded = base64.b64encode(output.getvalue()).decode("ascii")
        return f"data:image/jpeg;base64,{encoded}"
    except (requests.RequestException, OSError, ValueError):
        return None


@st.cache_data(ttl=86400, show_spinner=False)
def image_from_product_page(product_url: str) -> str | None:
    """Recover the main product image from retailer social-preview metadata."""
    if not product_url.startswith(("http://", "https://")):
        return None
    try:
        response = requests.get(
            product_url,
            timeout=6,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 Chrome/126.0 Safari/537.36"
                ),
                "Accept-Language": "en-US,en;q=0.8",
            },
        )
        response.raise_for_status()
        page = response.text[:1_500_000]
        patterns = (
            r'<meta[^>]+property=["\']og:image(?::secure_url)?["\'][^>]+content=["\']([^"\']+)',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']og:image(?::secure_url)?["\']',
            r'<meta[^>]+name=["\']twitter:image(?::src)?["\'][^>]+content=["\']([^"\']+)',
            r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']twitter:image(?::src)?["\']',
        )
        for pattern in patterns:
            match = re.search(pattern, page, flags=re.IGNORECASE)
            if match:
                candidate = html.unescape(match.group(1).strip())
                if any(
                    token in candidate.lower()
                    for token in (
                        "onlineexclusive",
                        "online_exclusive",
                        "online-exclusive",
                        "image-unavailable",
                        "image_unavailable",
                        "no-image",
                        "no_image",
                        "placeholder",
                    )
                ):
                    continue
                return urljoin(product_url, candidate)
    except requests.RequestException:
        return None
    return None


def normalize_response(payload: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if isinstance(payload, list):
        return payload, {}
    if not isinstance(payload, dict):
        raise ValueError("The API returned an unsupported response format.")
    recommendations = payload.get("recommendations", payload.get("results", payload.get("items", [])))
    if not isinstance(recommendations, list):
        raise ValueError("The API response does not contain a recommendation list.")
    return recommendations, payload


@st.cache_data(ttl=60, show_spinner=False)
def load_options(api_url: str) -> tuple[dict[str, list[str]], bool, str]:
    try:
        # Vercel may need several seconds to wake the model on a cold start.
        response = requests.get(f"{api_url}/options", timeout=30)
        response.raise_for_status()
        data = response.json()
        return {
            key: data.get(key) or DEFAULT_OPTIONS[key]
            for key in ("interests", "occasions", "genders")
        }, True, ""
    except requests.RequestException as exc:
        return DEFAULT_OPTIONS, False, f"{type(exc).__name__}: {exc}"
    except (ValueError, TypeError) as exc:
        return DEFAULT_OPTIONS, False, f"Invalid API response: {exc}"


def request_recommendations(api_url: str, query: dict[str, Any]) -> dict[str, Any]:
    response = requests.post(f"{api_url}/recommend", json=query, timeout=60)
    try:
        body = response.json()
    except ValueError:
        body = None
    if not response.ok:
        detail = body.get("detail") if isinstance(body, dict) else response.text
        raise RuntimeError(f"API error {response.status_code}: {detail or 'Unknown error'}")
    if body is None:
        raise RuntimeError("The API returned an empty or invalid response.")
    return body


@st.cache_resource(show_spinner=False)
def load_local_artifact(model_path: str) -> dict[str, Any]:
    # The import resolves the class referenced inside the joblib pickle.
    import gift_engine  # noqa: F401

    return joblib.load(model_path)


def local_recommendations(artifact: dict[str, Any], query: dict[str, Any]) -> dict[str, Any]:
    engine = artifact["recommender"]
    requested_top_k = int(query["top_k"])
    result_offset = int(query.get("result_offset", 0))
    # Score a deeper pool so strict interest filtering still returns a full,
    # diverse shortlist. Interests are a user-selected constraint in the UI,
    # even though the underlying model treats them as weighted similarities.
    candidate_k = min(250, max(160, requested_top_k * 20))
    recs, diagnostics = engine.recommend(
        age=query["age"],
        gender=query["gender"],
        occasion=query["occasion"],
        budget=query["budget"],
        interests=query["interests"],
        top_k=candidate_k,
    )

    if query["interests"] and not recs.empty:
        interest_columns = [
            f"int_{interest}"
            for interest in query["interests"]
            if f"int_{interest}" in engine.catalog.columns
        ]
        if interest_columns:
            tagged_ids = set(
                engine.catalog.loc[
                    engine.catalog[interest_columns].astype(bool).any(axis=1),
                    "parent_id",
                ].astype(str)
            )
            recs = recs[recs["parent_id"].astype(str).isin(tagged_ids)]

    recs = recs.iloc[result_offset : result_offset + requested_top_k].reset_index(drop=True)
    diagnostics = {
        **diagnostics,
        "strict_interest_filter": bool(query["interests"]),
        "n_returned": int(len(recs)),
    }
    records = recs.to_dict(orient="records")
    display_columns = [
        column
        for column in ("parent_id", "image_url", "description", "currency")
        if column in engine.catalog.columns
    ]
    if "parent_id" in display_columns:
        display = (
            engine.catalog[display_columns]
            .drop_duplicates("parent_id")
            .set_index("parent_id")
            .to_dict(orient="index")
        )
        for record in records:
            record.update(display.get(record.get("parent_id"), {}))
    names = artifact.get("cluster_names", {})
    return {
        "segment": names.get(diagnostics["cluster_id"], "Curated gifts"),
        "diagnostics": diagnostics,
        "recommendations": records,
    }


def generate_recommendations(
    data_source: str,
    artifact: dict[str, Any] | None,
    api_url: str,
    query: dict[str, Any],
    offset: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate one shortlist page while preserving the same recipient profile."""
    desired_count = int(query["top_k"])
    if data_source == "Demo":
        rotated = DEMO_RECOMMENDATIONS[offset:] + DEMO_RECOMMENDATIONS[:offset]
        payload = {
            "segment": "Thoughtful lifestyle gifts",
            "diagnostics": {"n_admissible": 128, "relaxation": None},
            "recommendations": rotated[:desired_count],
        }
    elif data_source == "Model":
        if artifact is None:
            raise RuntimeError("The model is unavailable. Check the model path and deployment files.")
        model_query = {**query, "result_offset": offset}
        payload = local_recommendations(artifact, model_query)
    else:
        api_query = {
            **query,
            "top_k": min(50, offset + desired_count),
        }
        payload = request_recommendations(api_url, api_query)

    recommendations, meta = normalize_response(payload)
    if data_source == "FastAPI" and offset:
        recommendations = recommendations[offset : offset + desired_count]
    return recommendations, meta


def explore_gifts(
    artifact: dict[str, Any],
    category: str,
    occasion: str,
    max_budget: float,
    seed: int,
    page: int = 0,
    limit: int = 12,
) -> tuple[list[dict[str, Any]], int]:
    """Return a varied, deterministic catalog shelf for non-personalized browsing."""
    catalog = artifact["recommender"].catalog
    mask = catalog["price_min"].le(max_budget)
    if category != "All categories":
        mask &= catalog["category"].eq(category)
    if occasion != "Any occasion":
        occasion_column = f"occ_{occasion}"
        if occasion_column in catalog.columns:
            mask &= catalog[occasion_column].astype(bool)

    pool = catalog.loc[mask].copy()
    if pool.empty:
        return [], 0
    if "has_image" in pool.columns:
        pool = pool.sort_values("has_image", ascending=False)
    # Keep the shelf varied before sampling so one large subcategory cannot dominate.
    if "sub_category" in pool.columns:
        pool = pool.groupby("sub_category", group_keys=False).head(30)
    pool = pool.sample(frac=1, random_state=42 + seed).reset_index(drop=True)
    total_items = len(pool)
    page_count = max(1, (total_items + limit - 1) // limit)
    page = min(max(0, page), page_count - 1)
    start = page * limit
    shelf = pool.iloc[start : start + limit]
    records = shelf.to_dict(orient="records")
    urls = [image_url(record) for record in records]
    with ThreadPoolExecutor(max_workers=8) as executor:
        embedded = list(
            executor.map(lambda url: cached_image_source(url) if url else None, urls)
        )
    for record, source in zip(records, embedded):
        record["_embedded_image"] = source
    return records, page_count


def product_card(item: dict[str, Any], rank: int, scope: str = "recommendation") -> None:
    name = safe_text(first_present(item, ["product_name", "name", "title"]), "Untitled gift")
    brand = safe_text(first_present(item, ["brand", "store", "source"]), "Independent brand")
    category = safe_text(first_present(item, ["sub_category", "category", "gift_type"]), "Curated gift")
    score_raw = first_present(item, ["match_score", "score", "similarity"])
    try:
        score = float(score_raw)
        score = score * 100 if score <= 1 else score
        score_label = f"{score:.0f}% match"
    except (TypeError, ValueError):
        score_label = f"Pick #{rank}"

    price_raw = first_present(item, ["price_median", "price", "price_min"])
    try:
        price = f"{float(price_raw):,.0f} SAR"
    except (TypeError, ValueError):
        price = "Price unavailable"

    raw_image = image_url(item)
    img = item.get("_embedded_image")
    if "_embedded_image" not in item:
        img = cached_image_source(raw_image) if raw_image else None
    if not img:
        product_url = first_present(item, ["product_url", "url", "link"])
        recovered_url = (
            image_from_product_page(product_url)
            if isinstance(product_url, str)
            else None
        )
        img = cached_image_source(recovered_url) if recovered_url else None
    visual = (
        f'<div class="product-image-frame">'
        f'<img class="product-image-backdrop" src="{html.escape(img, quote=True)}" alt="" aria-hidden="true">'
        f'<img class="product-image" src="{html.escape(img, quote=True)}" alt="{name}" loading="lazy">'
        f'</div>'
        if img
        else (
            '<div class="placeholder"><span class="placeholder-gift"><i></i></span>'
            '<strong>A thoughtful surprise</strong>'
            f'<small>{category or "Gift"} · image coming soon</small></div>'
        )
    )
    api_reasons = item.get("reasons")
    if isinstance(api_reasons, list):
        reasons = [str(reason) for reason in api_reasons if reason]
    else:
        reasons = []
        interest = first_present(item, ["interest_similarity"])
        budget_fit = first_present(item, ["budget_fit"])
        try:
            if interest is not None:
                reasons.append(f"{float(interest) * 100:.0f}% interest alignment")
            if budget_fit is not None:
                reasons.append(f"{float(budget_fit) * 100:.0f}% budget fit")
        except (TypeError, ValueError):
            reasons = []
    selected_details = item.get("_selected_specific_interests", [])
    selected_occasion = item.get("_selected_occasion")
    selected_budget = item.get("_selected_budget")
    friendly_reasons = []
    if selected_details:
        friendly_reasons.append(f"Matches {selected_details[0]}")
    if selected_occasion and selected_occasion != "No specific occasion":
        friendly_reasons.append(f"Suitable for {selected_occasion}")
    try:
        if selected_budget and price_raw is not None and float(price_raw) <= float(selected_budget):
            friendly_reasons.append(f"Within {float(selected_budget):,.0f} SAR")
    except (TypeError, ValueError):
        pass
    if not friendly_reasons:
        friendly_reasons = reasons[:2] or ["Selected for this profile"]
    reason_tags = "".join(
        f"<span>{safe_text(reason)}</span>" for reason in friendly_reasons[:3]
    )

    st.markdown(
        f"""
        <article class="product-card">
          {visual}
          <span class="score">{safe_text(score_label)}</span>
          <div class="category">{category}</div>
          <div class="product-title">{name}</div>
          <div class="product-meta">{brand}</div>
          <div class="price">{safe_text(price)}</div>
          <div class="reason-tags">{reason_tags}</div>
        </article>
        """,
        unsafe_allow_html=True,
    )
    item_id = str(first_present(item, ["parent_id", "id", "product_id"]) or f"rank-{rank}")
    saved = st.session_state.setdefault("saved_gifts", {})
    compared = st.session_state.setdefault("compare_gifts", {})
    action_cols = st.columns(3, gap="small")
    with action_cols[0]:
        is_saved = item_id in saved
        if st.button(
            "♥ Saved" if is_saved else "♡ Save",
            key=f"save_{scope}_{item_id}_{rank}",
            use_container_width=True,
        ):
            if is_saved:
                saved.pop(item_id, None)
            else:
                saved[item_id] = item
            st.rerun()
    with action_cols[1]:
        is_compared = item_id in compared
        compare_disabled = not is_compared and len(compared) >= 3
        if st.button(
            "✓ Added" if is_compared else "⇄ Compare",
            key=f"compare_{scope}_{item_id}_{rank}",
            disabled=compare_disabled,
            use_container_width=True,
            help="You can compare up to three gifts.",
        ):
            if is_compared:
                compared.pop(item_id, None)
            else:
                compared[item_id] = item
            st.rerun()
    url = first_present(item, ["product_url", "url", "link"])
    with action_cols[2]:
        if isinstance(url, str) and url.startswith(("http://", "https://")):
            st.link_button("View ↗", url, use_container_width=True)


language = st.sidebar.selectbox("Language / اللغة", ["English", "العربية"])
ARABIC = language == "العربية"
T = {
    "eyebrow": "طريقة أكثر خصوصية لاختيار الهدية" if ARABIC else "A more personal way to give",
    "hero": "اعثر على الهدية التي تبدو مناسبة تماماً." if ARABIC else "Find the gift that feels <em>just right.</em>",
    "intro": (
        "أخبرنا قليلاً عن الشخص والمناسبة، وسنحوّل التفاصيل إلى قائمة هدايا مختارة خصيصاً له."
        if ARABIC
        else "Tell us a little about the person and the moment. Our recommendation engine will turn those details into a shortlist made for them."
    ),
    "profile": "ملف المستلم" if ARABIC else "Recipient profile",
    "celebrating": "لمن نختار الهدية؟" if ARABIC else "Who are we celebrating?",
    "age": "عمر المستلم" if ARABIC else "Recipient age",
    "gender": "تفضيل الهدية" if ARABIC else "Gift preference",
    "occasion": "المناسبة" if ARABIC else "Occasion",
    "budget": "الميزانية القصوى (ر.س)" if ARABIC else "Maximum budget (SAR)",
    "interests": "الاهتمامات" if ARABIC else "Interests",
    "count": "عدد الاقتراحات" if ARABIC else "Number of recommendations",
    "find": "اعثر على هدايا مناسبة  ←" if ARABIC else "Find thoughtful gifts  →",
    "results": "النتائج المختارة" if ARABIC else "Curated results",
    "worth": "هدايا تستحق التقديم" if ARABIC else "Gifts worth giving",
}
direction = "rtl" if ARABIC else "ltr"

st.markdown(
    f"""
    <div class="topbar">
      <div class="brand">giftly<span class="brand-dot">.</span></div>
      <div class="top-pill">AI-curated · Gulf retailers · Real products</div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.segmented_control(
    "Main navigation",
    ["✦ Gift Finder", "⌘ Explore", "♡ Saved", "⇄ Compare", "◈ Model Insights"],
    default="✦ Gift Finder",
    selection_mode="single",
    label_visibility="collapsed",
)
page = page or "✦ Gift Finder"

if page == "✦ Gift Finder":
    st.markdown(
        f"""
        <section class="hero" dir="{direction}">
          <div class="gift-art" aria-hidden="true">
            <div class="cascade-card cascade-one">GIFTLY</div>
            <div class="cascade-card cascade-two">GIFT&nbsp; ♡</div>
            <div class="cascade-card cascade-three">FOR YOU</div>
            <div class="cascade-card cascade-four">♡</div>
            <div class="gift-shadow"></div>
          </div>
          <div class="eyebrow">{T["eyebrow"]}</div>
          <h1>{T["hero"]}</h1>
          <p>{T["intro"]}</p>
          <div class="hero-proof">
            <span class="proof-chip">✦ 45,000+ real gifts</span>
            <span class="proof-chip">◌ Budget-aware matching</span>
            <span class="proof-chip">♡ Personalized to their interests</span>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )
else:
    page_copy = {
        "⌘ Explore": (
            "Explore the collection",
            "Browse real gifts by category, occasion, and budget—no questionnaire required.",
        ),
        "♡ Saved": (
            "Your saved shortlist",
            "Keep promising ideas together while you decide on the perfect gift.",
        ),
        "⇄ Compare": (
            "Compare your favourites",
            "Review up to three gifts side by side before making the final choice.",
        ),
        "◈ Model Insights": (
            "Inside the recommendation engine",
            "See the data, evaluation results, and design choices behind every recommendation.",
        ),
    }
    page_title, page_description = page_copy[page]
    st.markdown(
        f"""
        <section class="page-hero">
          <div class="eyebrow">Giftly collection</div>
          <h1>{page_title}</h1>
          <p>{page_description}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    st.header("App settings")
    model_path = Path(DEFAULT_MODEL_PATH)
    if not model_path.is_absolute():
        model_path = Path(__file__).resolve().parent / model_path
    data_source = "FastAPI"
    if st.session_state.get("_active_runtime") != data_source:
        st.session_state.pop("gift_results", None)
        st.session_state.pop("gift_meta", None)
        st.session_state["_active_runtime"] = data_source
    api_url = DEFAULT_API_URL
    st.caption("The visual theme follows your device's light or dark system setting.")

artifact = None
artifact_error = None
api_connected = False
api_error = ""

# FastAPI serves recommendations, while the bundled artifact powers Explore
# and Model Insights. Loading is cached, so reruns do not reload the joblib file.
try:
    artifact = load_local_artifact(str(model_path))
except Exception as exc:
    artifact_error = str(exc)

if data_source == "FastAPI":
    options, api_connected, api_error = load_options(api_url)
elif artifact is not None:
    metadata = artifact["metadata"]
    options = {key: metadata[key] for key in ("interests", "occasions", "genders")}
else:
    options, api_connected = DEFAULT_OPTIONS, False

with st.sidebar:
    if data_source == "Model" and artifact is not None:
        st.success(f"Model ready · {artifact['metadata']['n_items']:,} products")
    elif data_source == "FastAPI" and api_connected:
        st.success("FastAPI connected")
    elif data_source == "Demo":
        st.info("Presentation demo data")
    else:
        st.error(f"{data_source} is unavailable")
        if data_source == "FastAPI":
            st.caption(f"API: {api_url}")
            if api_error:
                st.caption(api_error)

if page == "⌘ Explore":
    if artifact is None:
        st.error("The catalog is unavailable because the model could not be loaded.")
    else:
        catalog = artifact["recommender"].catalog
        explore_controls = st.columns([1.2, 1.2, 1, 0.72], gap="medium")
        categories = ["All categories"] + sorted(
            catalog["category"].dropna().astype(str).unique()
        )
        with explore_controls[0]:
            explore_category = st.selectbox("Browse category", categories)
        with explore_controls[1]:
            explore_occasion = st.selectbox(
                "Browse occasion",
                ["Any occasion"] + list(artifact["metadata"]["occasions"]),
            )
        with explore_controls[2]:
            explore_budget = st.select_slider(
                "Up to",
                options=[100, 200, 350, 500, 750, 1200, 2000, 5000, 10000],
                value=750,
                format_func=lambda value: f"{value:,} SAR",
            )
        with explore_controls[3]:
            st.write("")
            st.write("")
            if st.button("↻ New picks", use_container_width=True):
                st.session_state["explore_seed"] = st.session_state.get("explore_seed", 0) + 1
                st.session_state["explore_page"] = 0
                st.rerun()

        explore_seed = st.session_state.get("explore_seed", 0)
        explore_page = st.session_state.get("explore_page", 0)
        explore_items, explore_page_count = explore_gifts(
            artifact,
            explore_category,
            explore_occasion,
            float(explore_budget),
            explore_seed,
            explore_page,
        )
        if explore_page_count and explore_page >= explore_page_count:
            st.session_state["explore_page"] = explore_page_count - 1
            st.rerun()
        if explore_items:
            for start in range(0, len(explore_items), 4):
                explore_cols = st.columns(4, gap="medium")
                for offset, item in enumerate(explore_items[start : start + 4]):
                    with explore_cols[offset]:
                        product_card(
                            item,
                            explore_page * 12 + start + offset + 1,
                            scope=f"explore_{explore_seed}",
                        )
            page_controls = st.columns([1, 2, 1], gap="medium")
            with page_controls[0]:
                if st.button(
                    "← Previous",
                    disabled=explore_page <= 0,
                    use_container_width=True,
                ):
                    st.session_state["explore_page"] = explore_page - 1
                    st.rerun()
            with page_controls[1]:
                st.markdown(
                    f'<div class="status" style="text-align:center">Page '
                    f'{explore_page + 1:,} of {explore_page_count:,}</div>',
                    unsafe_allow_html=True,
                )
            with page_controls[2]:
                if st.button(
                    "Next →",
                    disabled=explore_page >= explore_page_count - 1,
                    use_container_width=True,
                ):
                    st.session_state["explore_page"] = explore_page + 1
                    st.rerun()
        else:
            st.info("No gifts match this combination. Try a higher budget or broader category.")
    st.stop()

if page == "♡ Saved":
    saved_page_items = list(st.session_state.get("saved_gifts", {}).values())
    if not saved_page_items:
        st.info("Your shortlist is empty. Save products from Gift Finder or Explore.")
    else:
        for start in range(0, len(saved_page_items), 3):
            saved_page_cols = st.columns(3, gap="large")
            for offset, item in enumerate(saved_page_items[start : start + 3]):
                with saved_page_cols[offset]:
                    product_card(item, start + offset + 1, scope="saved_page")
    st.stop()

if page == "⇄ Compare":
    compared_items = list(st.session_state.get("compare_gifts", {}).values())
    if not compared_items:
        st.info(
            "No gifts selected yet. Open Gift Finder or Explore and choose ⇄ Compare on up to three products."
        )
    else:
        compare_header = st.columns([3, 1])
        with compare_header[0]:
            st.markdown(f"### {len(compared_items)} gift{'s' if len(compared_items) != 1 else ''} selected")
        with compare_header[1]:
            if st.button("Clear comparison", use_container_width=True):
                st.session_state["compare_gifts"] = {}
                st.rerun()

        compare_cols = st.columns(len(compared_items), gap="large")
        for index, item in enumerate(compared_items):
            with compare_cols[index]:
                product_card(item, index + 1, scope="comparison")

        comparison_rows = []
        for item in compared_items:
            raw_score = first_present(item, ["match_score", "score", "similarity"])
            try:
                numeric_score = float(raw_score)
                numeric_score = numeric_score * 100 if numeric_score <= 1 else numeric_score
                score_text = f"{numeric_score:.0f}%"
            except (TypeError, ValueError):
                numeric_score = 0
                score_text = "—"
            raw_price = first_present(item, ["price_median", "price", "price_min"])
            try:
                price_text = f"{float(raw_price):,.0f} SAR"
            except (TypeError, ValueError):
                price_text = "Unavailable"
            comparison_rows.append(
                {
                    "Gift": first_present(item, ["product_name", "name", "title"]) or "Untitled gift",
                    "Price": price_text,
                    "Match": score_text,
                    "MatchValue": max(0, min(100, numeric_score)),
                    "Category": first_present(item, ["sub_category", "category", "gift_type"]) or "—",
                    "Brand": first_present(item, ["brand", "store", "source"]) or "—",
                }
            )
        st.markdown("### At-a-glance comparison")
        comparison_html = [
            """
            <div class="compare-wrap">
              <div class="compare-row compare-head">
                <div>Gift</div><div>Price</div><div>Match</div><div>Category</div><div>Brand</div>
              </div>
            """
        ]
        for row in comparison_rows:
            comparison_html.append(
                f"""
                <div class="compare-row">
                  <div>
                    <div class="compare-name">{safe_text(row["Gift"])}</div>
                    <div class="compare-sub">Recommended shortlist item</div>
                  </div>
                  <div><span class="compare-price">{safe_text(row["Price"])}</span></div>
                  <div>
                    <div class="match-value">{safe_text(row["Match"])}</div>
                    <div class="match-track"><div class="match-fill" style="width:{row["MatchValue"]:.0f}%"></div></div>
                  </div>
                  <div class="compare-cell">{safe_text(row["Category"])}</div>
                  <div class="compare-cell">{safe_text(row["Brand"])}</div>
                </div>
                """
            )
        comparison_html.append("</div>")
        comparison_markup = re.sub(
            r">\s+<",
            "><",
            "".join(comparison_html).strip(),
        )
        st.markdown(comparison_markup, unsafe_allow_html=True)
    st.stop()

if page == "◈ Model Insights":
    if artifact is None:
        st.error("Model metadata is unavailable.")
    else:
        model_meta = artifact["metadata"]
        st.markdown(
            f"""
            <div class="insight-grid">
              <div class="insight-card"><small>Product catalog</small><strong>{model_meta['n_items']:,}</strong><span>real gifts analysed</span></div>
              <div class="insight-card"><small>Feature space</small><strong>{model_meta['n_features']}</strong><span>signals per product</span></div>
              <div class="insight-card"><small>Learned segments</small><strong>{model_meta['k']}</strong><span>distinct gift profiles</span></div>
              <div class="insight-card"><small>Silhouette score</small><strong>{model_meta['silhouette']:.3f}</strong><span>cluster separation quality</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("### From preferences to presents")
        st.markdown(
            """
            <div class="model-flow">
              <div class="flow-step"><span class="flow-number">01</span><b>Filter</b><p>Apply age, occasion, preference and budget constraints.</p></div>
              <div class="flow-step"><span class="flow-number">02</span><b>Represent</b><p>Place the recipient and products in one shared feature space.</p></div>
              <div class="flow-step"><span class="flow-number">03</span><b>Score</b><p>Blend similarity, budget fit and catalog-quality signals.</p></div>
              <div class="flow-step"><span class="flow-number">04</span><b>Diversify</b><p>Prevent one product type from dominating the shortlist.</p></div>
              <div class="flow-step"><span class="flow-number">05</span><b>Explain</b><p>Return clear match, interest and budget indicators.</p></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        profile = artifact.get("cluster_profile")
        if profile is not None and not profile.empty:
            st.markdown("### Gift personalities discovered by the model")
            st.caption("Each card is a naturally occurring product group learned from the catalog.")
            segment_rows = profile.sort_values("n_items", ascending=False).to_dict(orient="records")
            for start in range(0, len(segment_rows), 3):
                segment_cols = st.columns(3, gap="medium")
                for offset, segment in enumerate(segment_rows[start : start + 3]):
                    interests_copy = str(segment.get("top_interests", "")).split(" + ")
                    occasions_copy = str(segment.get("top_occasions", "")).split(" + ")
                    chips = "".join(
                        f'<span class="segment-chip">{safe_text(label)}</span>'
                        for label in (interests_copy + occasions_copy)
                        if label and label != "nan"
                    )
                    with segment_cols[offset]:
                        st.markdown(
                            f"""
                            <div class="segment-card">
                              <div class="segment-size">{int(segment.get('n_items', 0)):,} products</div>
                              <h4>{safe_text(segment.get('archetype'), 'Gift segment')}</h4>
                              <div>{chips}</div>
                              <div class="segment-price">Typical price · {float(segment.get('median_price', 0)):,.0f} SAR</div>
                            </div>
                            """,
                            unsafe_allow_html=True,
                        )
    st.stop()

st.write("")

st.markdown(f'<div class="section-kicker">1 · {T["profile"]}</div>', unsafe_allow_html=True)
st.subheader(T["celebrating"])
with st.form("gift_profile", clear_on_submit=False):
    profile_cols = st.columns(4, gap="medium")
    with profile_cols[0]:
        age = st.number_input(T["age"], min_value=0, max_value=99, value=27, step=1)
    with profile_cols[1]:
        gender = st.selectbox(
            T["gender"],
            options["genders"],
            index=options["genders"].index("Any") if "Any" in options["genders"] else 0,
            help="Choose Any when you do not want gender to influence the results.",
        )
    with profile_cols[2]:
        occasion_choice = st.selectbox(
            T["occasion"],
            ["No specific occasion"] + list(options["occasions"]),
            index=1 if options["occasions"] else 0,
            help="Choose No specific occasion for an everyday or just-because gift.",
        )
        occasion = (
            "__none__"
            if occasion_choice == "No specific occasion"
            else occasion_choice
        )
    with profile_cols[3]:
        budget = st.number_input(T["budget"], min_value=1, value=500, step=50)

    preference_cols = st.columns([1.7, 1], gap="medium")
    with preference_cols[0]:
        specific_interests = st.multiselect(
            T["interests"],
            list(SPECIFIC_INTERESTS),
            default=["Luxury Fragrances"],
            max_selections=5,
            placeholder="Choose hobbies, tastes, or favourite activities",
        )
        st.caption("Choose up to five specific interests for a more focused match.")
    with preference_cols[1]:
        top_k = st.number_input(
            T["count"],
            min_value=3,
            max_value=12,
            value=6,
            step=1,
            help="Use − or + to adjust the size of your shortlist.",
        )
    submitted = st.form_submit_button(T["find"])

if data_source == "Model" and artifact is not None:
    connection_text = f"● Model ready — {artifact['metadata']['n_items']:,} products"
elif data_source == "Model":
    connection_text = "○ Model unavailable — check the deployment files"
elif data_source == "FastAPI" and api_connected:
    connection_text = "● Recommendation API connected"
elif data_source == "FastAPI":
    connection_text = "○ API offline — check the FastAPI URL"
else:
    connection_text = "● Presentation demo mode"
st.markdown(f'<div class="status">{connection_text}</div>', unsafe_allow_html=True)
if artifact_error:
    st.error(f"Could not load the model: {artifact_error}")

st.write("")
st.markdown(f'<div class="section-kicker">2 · {T["results"]}</div>', unsafe_allow_html=True)
st.subheader(T["worth"])

if submitted:
    interests = list(
        dict.fromkeys(
            SPECIFIC_INTERESTS[item]
            for item in specific_interests[:5]
            if SPECIFIC_INTERESTS[item] in options["interests"]
        )
    )
    query = {
        "age": int(age),
        "gender": gender,
        "occasion": occasion,
        "budget": float(budget),
        "interests": interests,
        "specific_interests": list(specific_interests),
        "top_k": int(top_k),
    }
    st.session_state["last_gift_query"] = query
    st.session_state["last_specific_interests"] = list(specific_interests)
    st.session_state["last_occasion_choice"] = occasion_choice
    st.session_state["recommendation_page"] = 0
    try:
        with st.spinner("Curating the best matches…"):
            recommendations, response_meta = generate_recommendations(
                data_source,
                artifact,
                api_url,
                query,
            )
        for recommendation in recommendations:
            recommendation["_selected_specific_interests"] = list(specific_interests)
            recommendation["_selected_occasion"] = occasion_choice
            recommendation["_selected_budget"] = float(budget)
        st.session_state["gift_results"] = recommendations
        st.session_state["gift_meta"] = response_meta
    except (requests.RequestException, RuntimeError, ValueError) as exc:
        st.error(str(exc))
        st.info("The recommendation service is unavailable. Please try again shortly.")

recommendations = st.session_state.get("gift_results", [])
meta = st.session_state.get("gift_meta", {})

if recommendations:
    segment = meta.get("segment")
    diagnostics = meta.get("diagnostics") or {}
    caption_parts = []
    if segment:
        caption_parts.append(f"Collection: {segment}")
    if diagnostics.get("n_admissible") is not None:
        caption_parts.append(f"{diagnostics['n_admissible']:,} eligible products considered")
    if diagnostics.get("relaxation"):
        caption_parts.append(f"Search expanded: {diagnostics['relaxation']}")
    if caption_parts:
        st.markdown(
            f'<div class="results-note">{safe_text(" · ".join(caption_parts))}</div>',
            unsafe_allow_html=True,
        )

    control_cols = st.columns([1, 1.5, 0.7], gap="medium")
    with control_cols[0]:
        sort_by = st.selectbox(
            "Sort recommendations",
            ["Best match", "Price: low to high", "Price: high to low"],
        )
    available_categories = sorted(
        {
            str(first_present(item, ["category", "sub_category", "gift_type"]))
            for item in recommendations
            if first_present(item, ["category", "sub_category", "gift_type"])
        }
    )
    with control_cols[1]:
        category_filter = st.multiselect(
            "Filter categories",
            available_categories,
            placeholder="Show every category",
        )
    with control_cols[2]:
        st.write("")
        st.write("")
        if st.button("↻ New picks", use_container_width=True):
            last_query = st.session_state.get("last_gift_query")
            if last_query:
                next_page = st.session_state.get("recommendation_page", 0) + 1
                page_size = int(last_query["top_k"])
                offset = next_page * page_size
                if data_source == "FastAPI":
                    offset %= max(page_size, 48 - page_size)
                try:
                    with st.spinner("Finding another thoughtful set…"):
                        fresh_recommendations, fresh_meta = generate_recommendations(
                            data_source,
                            artifact,
                            api_url,
                            last_query,
                            offset=offset,
                        )
                    if fresh_recommendations:
                        saved_interests = st.session_state.get("last_specific_interests", [])
                        saved_occasion = st.session_state.get("last_occasion_choice")
                        for recommendation in fresh_recommendations:
                            recommendation["_selected_specific_interests"] = list(saved_interests)
                            recommendation["_selected_occasion"] = saved_occasion
                            recommendation["_selected_budget"] = float(last_query["budget"])
                        st.session_state["gift_results"] = fresh_recommendations
                        st.session_state["gift_meta"] = fresh_meta
                        st.session_state["recommendation_page"] = next_page
                        st.rerun()
                    else:
                        st.warning("No additional matches were available for this profile.")
                except (requests.RequestException, RuntimeError, ValueError) as exc:
                    st.error(str(exc))

    visible_recommendations = [
        item
        for item in recommendations
        if not category_filter
        or str(first_present(item, ["category", "sub_category", "gift_type"]))
        in category_filter
    ]
    if sort_by != "Best match":
        visible_recommendations = sorted(
            visible_recommendations,
            key=lambda item: float(
                first_present(item, ["price_median", "price", "price_min"]) or 0
            ),
            reverse=sort_by == "Price: high to low",
        )

    for start in range(0, len(visible_recommendations), 3):
        cols = st.columns(3, gap="large")
        for offset, item in enumerate(visible_recommendations[start : start + 3]):
            with cols[offset]:
                product_card(item, start + offset + 1)
else:
    st.info("Complete the profile and select “Find thoughtful gifts” to reveal the collection.")

st.divider()
st.caption("Giftly uses unsupervised similarity, budget fit, and catalog quality signals. Prices and availability are provided by each retailer.")
