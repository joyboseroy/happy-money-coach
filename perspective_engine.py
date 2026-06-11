"""
perspective_engine.py

Rules-based ethical scoring engine.
Architecture: Transaction -> Category -> Ontology -> Perspective Score

NO AGENTS. The wisdom is in the ontology (ethical_ontology.json).
The LLM is only used to generate narrative explanations.

The six lenses:
  universal  - cross-traditional human values
  buddhist   - Noble Eightfold Path, Five Precepts, dana/sila/panna
  christian  - stewardship, charity, temperance, vocation
  jewish     - tzedakah, tikkun olam, bal tashchit, talmud torah
  islamic    - halal/haram, zakat, israf, amanah
  stoic      - virtue ethics, preferred indifferents, duty/kathêkon
"""

import json
import re
import requests
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from transaction_parser import Transaction

ONTOLOGY_PATH = Path(__file__).resolve().parent / "data" / "ethical_ontology.json"
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

LENSES = ["universal", "buddhist", "christian", "jewish", "islamic", "stoic"]

LENS_DESCRIPTIONS = {
    "universal":  "Universal Human Values",
    "buddhist":   "Buddhist Ethics",
    "christian":  "Christian Ethics",
    "jewish":     "Jewish Ethics (Halacha)",
    "islamic":    "Islamic Ethics (Shariah)",
    "stoic":      "Stoic Philosophy",
}


@dataclass
class EthicalScore:
    lens: str
    score: float            # -1.0 to 1.0
    score_label: str        # Encouraged / Neutral / Caution / Concern
    reason: str             # from ontology (brief)
    flags: list[str]        # specific concerns
    concepts: list[str]     # tradition-specific concepts referenced
    llm_explanation: str = ""  # generated narrative (optional, requires Ollama)


@dataclass
class EthicalProfile:
    tx_description: str
    tx_amount: float
    category: str
    human_need: str
    value: str
    virtue: str
    scores: dict[str, EthicalScore] = field(default_factory=dict)
    virtue_alignment: list[str] = field(default_factory=list)
    composite_score: float = 0.0
    composite_label: str = ""


def _load_ontology() -> dict:
    with open(ONTOLOGY_PATH) as f:
        return json.load(f)


def _score_to_label(score: float) -> str:
    if score >= 0.70:
        return "Encouraged"
    elif score >= 0.30:
        return "Positive"
    elif score >= -0.10:
        return "Neutral"
    elif score >= -0.40:
        return "Caution"
    else:
        return "Concern"


def _get_category_data(ontology: dict, category: str) -> dict:
    """Find the right ontology entry for a category."""
    cats = ontology.get("categories", {})
    if category in cats:
        return cats[category]
    # Fallback to closest match
    for k, v in cats.items():
        if k.lower() in category.lower() or category.lower() in k.lower():
            return v
    # Default to Shopping as most conservative fallback
    return cats.get("Shopping", {})


def _get_special_category(ontology: dict, description: str, category: str) -> Optional[dict]:
    """Check if transaction matches a special category (gambling, alcohol, fast fashion)."""
    desc_lower = description.lower()
    specials = ontology.get("special_categories", {})

    gambling_keywords = ["gamble", "casino", "bet", "lottery", "dream11", "mpl", "winzo",
                         "poker", "rummy cash", "fantasy league", "my11circle"]
    alcohol_keywords = ["liquor", "wine", "beer", "whisky", "rum", "alcohol", "bar ",
                        "pub ", "beverage store", "tasmac"]
    fashion_keywords = ["zara", "h&m", "shein", "fast fashion", "forever21", "f21"]

    if any(k in desc_lower for k in gambling_keywords):
        return specials.get("gambling")
    if any(k in desc_lower for k in alcohol_keywords):
        return specials.get("alcohol_tobacco")
    if any(k in desc_lower for k in fashion_keywords):
        return specials.get("fast_fashion")
    return None


# ---------------------------------------------------------------------------
# Core scoring function
# ---------------------------------------------------------------------------

def score_transaction(tx: Transaction, lenses: list[str] = None) -> EthicalProfile:
    """
    Score a single transaction across the requested lenses.
    Uses ontology only (no LLM). Call add_llm_explanations() separately.
    """
    if lenses is None:
        lenses = LENSES

    ontology = _load_ontology()
    special = _get_special_category(ontology, tx.description, tx.category)
    cat_data = special if special else _get_category_data(ontology, tx.category)

    profile = EthicalProfile(
        tx_description=tx.description,
        tx_amount=tx.amount,
        category=tx.category,
        human_need=cat_data.get("human_need", "Unknown"),
        value=cat_data.get("value", "Unknown"),
        virtue=cat_data.get("virtue", "Unknown"),
    )

    scores_list = []

    for lens in lenses:
        lens_data = cat_data.get(lens, {})
        if not lens_data:
            # Lens not present in ontology for this category — use universal as fallback
            lens_data = cat_data.get("universal", {"score": 0.0, "reason": "No specific guidance."})

        raw_score = float(lens_data.get("score", 0.0))
        reason = lens_data.get("reason", "")
        flags = lens_data.get("flags", [])
        concepts = (
            lens_data.get("concepts", []) +
            lens_data.get("references", []) +
            lens_data.get("figures", [])
        )

        ethical_score = EthicalScore(
            lens=lens,
            score=raw_score,
            score_label=_score_to_label(raw_score),
            reason=reason,
            flags=flags,
            concepts=concepts,
        )
        profile.scores[lens] = ethical_score
        scores_list.append(raw_score)

    # Composite: weighted average (universal counts 1.5x as anchor)
    if scores_list:
        universal_score = profile.scores.get("universal", EthicalScore("universal", 0, "Neutral", "", [], []))
        other_scores = [s for k, s in profile.scores.items() if k != "universal"]
        if other_scores:
            weighted = (universal_score.score * 1.5 + sum(s.score for s in other_scores)) / (1.5 + len(other_scores))
        else:
            weighted = universal_score.score
        profile.composite_score = round(weighted, 3)
        profile.composite_label = _score_to_label(profile.composite_score)

    # Virtue alignment
    virtue_map = ontology.get("virtue_map", {})
    profile.virtue_alignment = [
        v for v, data in virtue_map.items()
        if tx.category in data.get("categories", [])
    ]

    return profile


def score_batch(transactions: list[Transaction], lenses: list[str] = None) -> list[EthicalProfile]:
    """Score all transactions."""
    return [score_transaction(tx, lenses) for tx in transactions]


# ---------------------------------------------------------------------------
# LLM explanation layer (optional — requires Ollama)
# ---------------------------------------------------------------------------

def add_llm_explanation(profile: EthicalProfile, lens: str, use_llm: bool = True) -> str:
    """
    Generate a one-paragraph explanation for a specific lens using Qwen3.
    Falls back to the ontology reason if LLM unavailable.
    """
    if not use_llm or lens not in profile.scores:
        return profile.scores[lens].reason if lens in profile.scores else ""

    score_data = profile.scores[lens]
    lens_full = LENS_DESCRIPTIONS.get(lens, lens.title())

    lens_prompts = {
        "buddhist": "You are a Theravada Buddhist ethics teacher with deep knowledge of the Pali Canon, the Noble Eightfold Path, and the Vinaya.",
        "christian": "You are a Christian ethics scholar with knowledge of both Catholic social teaching and Protestant ethics traditions.",
        "jewish":   "You are a Jewish ethics scholar with knowledge of Talmud, Maimonides, and contemporary halachic reasoning.",
        "islamic":  "You are an Islamic finance and ethics scholar with knowledge of Quran, Hadith, fiqh, and contemporary fatawa.",
        "stoic":    "You are a Stoic philosophy scholar with deep knowledge of Epictetus, Marcus Aurelius, and Seneca.",
        "universal": "You are an ethicist drawing on cross-cultural human values research and positive psychology.",
    }

    system_persona = lens_prompts.get(lens, "You are an ethical philosophy scholar.")

    prompt = f"""{system_persona}

Analyse this financial transaction through the lens of {lens_full}.

Transaction: {profile.tx_description}
Category: {profile.category}
Amount: ₹{profile.tx_amount:,.0f}
Ethical Score: {score_data.score_label} ({score_data.score:.2f})
Key concepts from this tradition: {', '.join(score_data.concepts[:4]) if score_data.concepts else 'none specified'}
Flags: {', '.join(score_data.flags[:3]) if score_data.flags else 'none'}

Write one paragraph (3-4 sentences) explaining what this tradition says about this type of spending.
Be specific to this tradition's concepts and texts. Be honest about concerns without being preachy.
Write in third person as an analysis, not as a sermon. Do not start with 'I'."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "num_predict": 200}
            },
            timeout=30
        )
        text = response.json().get("response", "").strip()
        # Clean up any leading/trailing artifacts
        text = re.sub(r"^(Sure|Certainly|Of course)[,!.]?\s*", "", text)
        return text if text else score_data.reason
    except Exception as e:
        print(f"[perspective_engine] LLM error ({lens}): {e}")
        return score_data.reason


# ---------------------------------------------------------------------------
# Aggregate analytics
# ---------------------------------------------------------------------------

def get_lens_summary(profiles: list[EthicalProfile], lens: str) -> dict:
    """
    Summarise all transactions through one lens.
    Returns: {score_avg, best_category, worst_category, concern_count, encouraged_count}
    """
    scores = []
    cat_scores: dict[str, list[float]] = {}

    for p in profiles:
        if lens in p.scores:
            s = p.scores[lens].score
            scores.append(s)
            cat = p.category
            if cat not in cat_scores:
                cat_scores[cat] = []
            cat_scores[cat].append(s)

    if not scores:
        return {}

    avg_per_cat = {cat: sum(v)/len(v) for cat, v in cat_scores.items()}

    return {
        "lens": lens,
        "score_avg": round(sum(scores) / len(scores), 3),
        "score_label": _score_to_label(sum(scores) / len(scores)),
        "best_category": max(avg_per_cat, key=avg_per_cat.get) if avg_per_cat else "",
        "worst_category": min(avg_per_cat, key=avg_per_cat.get) if avg_per_cat else "",
        "concern_count": sum(1 for s in scores if s < -0.10),
        "encouraged_count": sum(1 for s in scores if s >= 0.70),
        "category_scores": avg_per_cat,
    }


def get_virtue_dashboard(profiles: list[EthicalProfile], category_summary: dict) -> dict:
    """
    Build the Virtue Dashboard:
    Instead of Food/Travel/Shopping, show Generosity/Learning/Consumption etc.
    """
    ontology = _load_ontology()
    virtue_map = ontology.get("virtue_map", {})

    virtue_spend: dict[str, float] = {}
    total_spend = sum(d["total_spend"] for cat, d in category_summary.items() if cat != "Income")

    for virtue, data in virtue_map.items():
        cats = data.get("categories", [])
        spend = sum(
            category_summary.get(cat, {}).get("total_spend", 0)
            for cat in cats
        )
        if spend > 0:
            virtue_spend[virtue] = spend

    # Uncategorised virtues go to "Consumption"
    categorised_spend = sum(virtue_spend.values())
    other_spend = total_spend - categorised_spend
    if other_spend > 0:
        virtue_spend["Consumption"] = virtue_spend.get("Consumption", 0) + other_spend

    # Convert to percentages
    if total_spend > 0:
        virtue_pct = {v: round(s / total_spend * 100, 1) for v, s in virtue_spend.items()}
    else:
        virtue_pct = {}

    return {
        "by_spend": virtue_spend,
        "by_pct": virtue_pct,
        "total_spend": total_spend,
    }


def get_ethical_leaderboard(profiles: list[EthicalProfile], lens: str = "universal", top_n: int = 5):
    """Top N most ethically positive transactions for a given lens."""
    scored = [(p, p.scores[lens].score) for p in profiles if lens in p.scores]
    return [p for p, _ in sorted(scored, key=lambda x: x[1], reverse=True)[:top_n]]


def get_concern_transactions(profiles: list[EthicalProfile], lens: str = "universal", threshold: float = -0.10):
    """Transactions that raise ethical concerns under a given lens."""
    return [p for p in profiles if lens in p.scores and p.scores[lens].score < threshold]
