"""FastAPI adapter for the serialized gift recommender.

Place this file in your project (for example, app/main.py), keep gift_engine.py
on PYTHONPATH, and adjust ARTIFACT_PATH if your folder layout differs.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

import gift_engine  # noqa: F401 - required to resolve the pickled class


DEFAULT_ARTIFACT_PATH = Path(__file__).resolve().parent / "models" / "gift_recommender_v3.joblib"
ARTIFACT_PATH = Path(os.getenv("GIFT_MODEL_PATH", str(DEFAULT_ARTIFACT_PATH)))
ART = joblib.load(ARTIFACT_PATH)
ENGINE = ART["recommender"]
META = ART["metadata"]
NAMES = ART["cluster_names"]

app = FastAPI(title="Gift Recommender", version=META["version"])
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class GiftRequest(BaseModel):
    age: int = Field(..., ge=0, le=99)
    gender: Literal["Female", "Male", "Any"]
    occasion: str
    budget: float = Field(..., gt=0)
    interests: list[str] = Field(default_factory=list)
    specific_interests: list[str] = Field(default_factory=list)
    top_k: int = Field(6, ge=1, le=50)


SPECIFIC_INTEREST_PATTERNS = {
    "Painting & Drawing": (
        r"\bpainting\b|\bdrawing\b|art supplies|\bsketch(?:ing|book)?\b|"
        r"watercolou?r|acrylic (?:paint|colou?r)|paint brush|artist brush|"
        r"\beasel\b|colou?ring set|stretched canvas|artist canvas|"
        r"canvas (?:panel|board|pad)"
    ),
    "Photography": r"camera|photograph|photography|tripod|camera lens",
    "Crafts & DIY": r"craft|crochet|sewing|knitting|\bdiy\b",
    "Music & Instruments": r"guitar|piano|musical keyboard|drum|violin|ukulele|saxophone|flute|trumpet|musical instrument",
    "Skincare": r"skincare|skin care|face cream|serum|moisturi|cleanser",
    "Makeup": r"makeup|lipstick|mascara|foundation|concealer|eyeshadow",
    "Haircare": r"haircare|hair care|shampoo|conditioner|hair mask|hair dryer|hair straight",
    "Nail Care": r"nail care|nail polish|manicure|pedicure",
    "Cooking": (
        r"cookware|cooking (?:set|pot|pan|utensil|tool)|frying pan|saucepan|"
        r"\bcooking pot\b|kitchen utensil|chef knife|kitchen knife|"
        r"chopping board|cutting board|pressure cooker|slow cooker|rice cooker|"
        r"air fryer|food processor|hand blender|stand mixer"
    ),
    "Coffee & Tea": (
        r"coffee (?:machine|maker|grinder|beans|capsules|set)|"
        r"tea (?:set|cup|cups|pot|bags|leaves)|"
        r"espresso (?:machine|maker|cup|cups)|nespresso|coffee & tea"
    ),
    "Chocolate & Gourmet Food": r"chocolate|gourmet|sweets",
    "Fashion Accessories": r"fashion accessories|\baccessories\b|scarf|sunglasses|\bbelt\b",
    "Handbags & Purses": (
        r"handbag|\bpurse\b|clutch bag|clutch purse|evening clutch|"
        r"envelope clutch|tote bag|crossbody(?: bag)?|"
        r"shoulder bag|\bsatchel\b"
    ),
    "Wallets & Card Holders": (
        r"\bwallet\b|\bwallets\b|card holder|coin purse"
    ),
    "Shoes & Sneakers": r"\bshoes?\b|sneaker|sandal|slipper",
    "Luxury Fragrances": r"perfume|fragrance|eau de parfum|eau de toilette",
    "Home Fragrances & Candles": r"home fragrance|candle|diffuser|incense",
    "PC & Console Gaming": r"pc games|\bgaming\b|playstation|xbox|nintendo",
    "Board Games & Puzzles": r"board game|jigsaw puzzle|puzzle game|\bchess\b|\bmonopoly\b",
    "Gardening": r"garden|gardening|plant pot|watering can",
    "Plants & Flowers": r"flower|plant|bouquet",
    "Home Décor": r"home décor|home decor|decoration",
    "Kitchen & Dining": r"kitchen|dining|dinning|tableware",
    "Home Organisation": r"organi[sz]er|storage|shelving",
    "Jewellery": r"jewellery|jewelry|necklace|bracelet|earring|\bring\b",
    "Watches": r"\bwatch(?:es)?\b",
    "Baby Gifts": r"baby product|newborn|baby gift",
    "Toys & Collectibles": r"\btoy(?:s)?\b|collectible|collectable",
    "Camping & Hiking": r"camping|hiking|\btent\b|sleeping bag",
    "Travel Accessories": r"travel|luggage|suitcase|passport holder",
    "Beach & Picnic": r"beach|picnic",
    "Books & Novels": r"\bbooks?\b|novel",
    "Journaling & Stationery": r"stationery|journal|notebook|\bpen\b|\boffice\b",
    "Courses & Learning": r"educational|learning|course",
    "Gym & Strength Training": r"\bgym\b|strength|dumbbell|weight training|fitness",
    "Running": r"running (?:shoe|shoes|sneaker|sneakers|trainer|trainers)|treadmill|jogging (?:shoe|shoes)",
    "Football": r"football|soccer",
    "Yoga & Pilates": r"yoga|pilates",
    "Mobile Phones": r"\biphone(?:\s+\d+)?\b|\bsmartphone\b|\bmobile phone\b|\bsamsung galaxy\b|\bgoogle pixel\b",
    "Phone Accessories": (
        r"phone case|iphone case|phone holder|phone cover|phone strap|"
        r"phone charger|wireless charger|charging cable|screen protector|"
        r"phone mount|phone stand|smartphone crossbody"
    ),
    "Computers & Accessories": r"computer|laptop|keyboard|\bmouse\b|monitor",
    "Audio & Headphones": r"\baudio\b|headphone|earbud|speaker",
    "Self-Care & Relaxation": r"self-care|self care|body care|relaxation",
    "Spa & Massage": r"\bspa\b|massage|massager",
    "Mindfulness & Sleep": (
        r"mindfulness|meditation|sleep mask|eye mask|sleep aid|"
        r"weighted blanket|white noise|aromatherapy"
    ),
}

SPECIFIC_INTEREST_EXCLUSIONS = {
    "Mobile Phones": (
        r"case|holder|cover|strap|crossbody|charger|charging|cable|"
        r"screen protector|mount|stand|charm"
    ),
    "Coffee & Tea": r"eyeshadow|eye shadow|eyeliner|eye pencil|brow pencil|video game|pc game",
    "Running": r"pc game|video game|steam game|soundtrack",
    "Board Games & Puzzles": r"handbag|purse|tote|pouch|shoe|pump|sandal",
}


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "Giftly Recommendation API",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_version": META["version"]}


@app.get("/options")
def options() -> dict[str, list[str]]:
    return {key: META[key] for key in ("interests", "occasions", "genders")}


def add_display_fields(records: list[dict]) -> list[dict]:
    """Join image/display columns that the ranking result intentionally omits."""
    available = [
        column
        for column in ("parent_id", "image_url", "description", "currency")
        if column in ENGINE.catalog.columns
    ]
    if "parent_id" not in available:
        return records
    display = (
        ENGINE.catalog[available]
        .drop_duplicates("parent_id")
        .set_index("parent_id")
        .to_dict(orient="index")
    )
    for record in records:
        record.update(display.get(record.get("parent_id"), {}))
    return records


@app.post("/recommend")
def recommend(req: GiftRequest) -> dict:
    if req.occasion != "__none__" and req.occasion not in META["occasions"]:
        raise HTTPException(
            422,
            f"occasion must be '__none__' or one of {META['occasions']}",
        )
    unknown = sorted(set(req.interests) - set(META["interests"]))
    if unknown:
        raise HTTPException(422, f"unknown interests: {unknown}")

    targeted_interests = [
        name
        for name in req.specific_interests
        if name in SPECIFIC_INTEREST_PATTERNS
    ]
    catalog_mask = None
    if targeted_interests:
        searchable = (
            ENGINE.catalog[["product_name", "category", "sub_category", "description"]]
            .fillna("")
            .astype(str)
            .agg(" ".join, axis=1)
        )
        combined_mask = None
        for name in targeted_interests:
            interest_mask = searchable.str.contains(
                SPECIFIC_INTEREST_PATTERNS[name], case=False, regex=True
            )
            exclusion = SPECIFIC_INTEREST_EXCLUSIONS.get(name)
            if exclusion:
                interest_mask &= ~searchable.str.contains(
                    exclusion, case=False, regex=True
                )
            combined_mask = interest_mask if combined_mask is None else (combined_mask | interest_mask)
        catalog_mask = combined_mask.to_numpy()
    recs, diagnostics = ENGINE.recommend(
        age=req.age,
        gender=req.gender,
        occasion=req.occasion,
        budget=req.budget,
        interests=req.interests,
        top_k=req.top_k,
        catalog_mask=catalog_mask,
    )
    if targeted_interests:
        diagnostics["specific_interest_filter"] = req.specific_interests
        diagnostics["n_specific_matches"] = int(len(recs))
    records = add_display_fields(recs.to_dict(orient="records"))
    return {
        "segment": (
            " + ".join(name for name in req.specific_interests if name in SPECIFIC_INTEREST_PATTERNS)
            or NAMES.get(diagnostics["cluster_id"], "General")
        ),
        "diagnostics": diagnostics,
        "recommendations": records,
    }
