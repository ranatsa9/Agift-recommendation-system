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
    "Music & Instruments": r"guitar|piano|musical keyboard|drum|violin|ukulele|saxophone|flute|trumpet|musical instrument",
    "Skincare": r"skincare|skin care|face cream|serum|moisturi|cleanser",
    "Makeup": r"makeup|cosmetic|lipstick|mascara|foundation|concealer|eyeshadow|eye shadow|blush|bronzer",
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
    "Fashion Accessories": r"scarf|sunglasses|\bbelt\b|cufflinks?|\btie\b|hair accessory|passport holder|cardholder",
    "Handbags & Purses": (
        r"handbag|\bpurse\b|clutch bag|clutch purse|evening clutch|"
        r"envelope clutch|tote bag|crossbody(?: bag)?|"
        r"shoulder bag|\bsatchel\b"
    ),
    "Wallets & Card Holders": (
        r"\bwallet\b|\bwallets\b|card holder|coin purse"
    ),
    "Shoes & Sneakers": r"\bshoes?\b|sneaker|trainer|sandal|slipper|loafer|\bboot(?:s)?\b|\bheel(?:s)?\b|\bpumps?\b|\bflats?\b|\bmules?\b",
    "Luxury Fragrances": r"perfume|fragrance|eau de parfum|eau de toilette",
    "Home Fragrances & Candles": r"home fragrance|scented candle|\bcandle\b|reed diffuser|home diffuser|room spray|incense|wax melt|fragrance oil",
    "PC & Console Gaming": r"pc games|\bgaming\b|playstation|xbox|nintendo",
    "Board Games & Puzzles": r"board game|card game|jigsaw puzzle|tabletop game|family game|chess (?:set|board)|\bmonopoly\b",
    "Gardening": r"gardening tool|plant pot|planter|\bseeds?\b|\bsoil\b|watering can|pruning shear|garden gloves|grow kit|herb kit|indoor garden|terrarium",
    "Plants & Flowers": r"flower|plant|bouquet",
    "Home Décor": r"home décor|home decor|cushion|throw pillow|\brug\b|\blamp\b|\bmirror\b|wall art|photo frame|\bvase\b|\bclock\b|decorative tray|ornament|candle holder|sculpture",
    "Kitchen & Dining": r"cookware|bakeware|frying pan|saucepan|\bpot\b|knife set|cutting board|coffee machine|blender|mixer|air fryer|kettle|dinnerware|tableware|plates|bowls|cups|glasses|cutlery|serving tray",
    "Home Organisation": r"jewellery organi[sz]er|makeup organi[sz]er|desk organi[sz]er|drawer divider|storage basket|storage box|closet organi[sz]er|shoe organi[sz]er|cable organi[sz]er|spice rack|bathroom organi[sz]er",
    "Jewellery": r"necklace|pendant|bracelet|bangle|\bring\b|earrings?|ear studs?|anklet|brooch|\bcharm\b|jewellery set|jewelry set",
    "Watches": r"wristwatch|smartwatch|analog watch|digital watch|chronograph|automatic watch|quartz watch|watch set",
    "Baby Gifts": r"baby product|newborn|baby gift",
    "Toys & Collectibles": r"\btoy(?:s)?\b|action figure|collectible figure|collectable figure|\bdoll\b|\bplush\b|building set|playset|model kit|trading cards|educational toy|activity kit",
    "Camping & Hiking": r"camping|hiking|\btent\b|sleeping bag",
    "Travel Accessories": r"travel|luggage|suitcase|passport holder",
    "Beach & Picnic": r"beach|picnic",
    "Books & Novels": r"\bbooks?\b|novel|paperback|hardcover|fiction|nonfiction|biography|poetry|storybook|study book|activity book",
    "Journaling & Stationery": r"stationery|\bjournal\b|\bnotebook\b|\bplanner\b|\bdiary\b|sketchbook|\bpen\b|\bpencil\b|writing set|sticky notes|bookmark",
    "Courses & Learning": r"\bcourse\b|textbook|study guide|workbook|educational kit|learning toy|language learning|exam preparation|activity book|reference book|online class",
    "Gym & Strength Training": r"\bgym\b|strength|dumbbell|weight training|fitness",
    "Running": r"running (?:shoe|shoes|sneaker|sneakers|trainer|trainers)|treadmill|jogging (?:shoe|shoes)",
    "Football": r"football|soccer",
    "Yoga & Pilates": r"yoga mat|pilates mat|yoga block|yoga strap|resistance band|exercise ball|foam roller|pilates ring|reformer|yoga towel|meditation cushion",
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
    "Painting & Drawing": r"canvas (?:shoe|bag|belt|hat|pouch|leather)|pc game|video game|steam",
    "Makeup": r"pc game|video game|steam|software",
    "Coffee & Tea": r"skincare|skin care|cream|serum|lotion|perfume|fragrance|eyeshadow|eye shadow|eyeliner|eye pencil|brow pencil|video game|pc game|steam",
    "Fashion Accessories": r"cabinet|dresser|drawer|furniture|storage|\bpen\b|perfume|fragrance|skincare|makeup kit|pc game|video game|steam",
    "Shoes & Sneakers": r"shoe cabinet|shoe rack|shoe storage|shoe bag|shoe organi[sz]er|shoe tree|shoe cleaner|shoe care",
    "Home Fragrances & Candles": r"makeup|cosmetic|powder|\bbrush\b|foundation|blush|applicator|hair diffuser|hair dryer",
    "Running": r"pc game|video game|steam game|soundtrack",
    "Board Games & Puzzles": r"steam|pc game|video game|digital game|download|software|dlc|handbag|purse|tote|pouch|shoe|pump|sandal",
    "Gardening": r"jewellery|jewelry|necklace|bracelet|\bring\b|perfume|fragrance|eau de parfum|gardenia|voucher|experience package|pc game|video game|steam",
    "Home Décor": r"pc game|video game|steam|dlc|decoration pack|birthday decoration|graduation decoration|party decoration|balloon|chocolate|edible",
    "Kitchen & Dining": r"kitchen cart|kitchen trolley|kitchen cabinet|kitchen island|storage furniture|pc game|video game|steam|simulator",
    "Home Organisation": r"hydraulic bed|\bbed\b|mattress|large furniture|hair styler|hair dryer|beauty tool",
    "Jewellery": r"perfume|fragrance|eau de parfum|eau de toilette|makeup|blush|foundation|lipstick|skincare",
    "Watches": r"watch charger|charging cable|watch strap|watch dogs|pc game|video game|steam|necklace|bracelet|earring|perfume|makeup",
    "Toys & Collectibles": r"perfume|fragrance|discovery set|lighter|wallet|necklace|jewellery|jewelry|leather goods|makeup",
    "Journaling & Stationery": r"steam|pc game|video game|digital game|software|download|dlc|visual novel|soundtrack",
    "Courses & Learning": r"\bhat\b|clothing|\bshirt\b|\bbag\b|jewellery|jewelry|fashion accessory",
    "Books & Novels": r"macbook|laptop sleeve|book sleeve|book box|book-shaped|jewellery box|jewelry box|bookish|\bhat\b|\btoy\b|learning board",
    "Yoga & Pilates": r"lenovo|tablet|laptop|keyboard|stylus|electronics|bracelet|necklace|jewellery|jewelry",
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
