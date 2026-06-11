"""
happy_money_tagger.py

Ken Honda's Happy Money framework applied to each transaction.

Core philosophy:
  - Money is energy. It carries emotional signatures.
  - Happy Money flows with gratitude, joy, meaning, and love.
  - Unhappy Money flows with stress, regret, obligation, or numbness.
  - The goal is not to spend less but to spend more consciously.

Honda Tags (emotional signatures):
  Gratitude   - receiving or giving money with appreciation
  Joy         - spending that creates genuine delight
  Meaning     - spending aligned with deep values
  Love        - spending for relationships and family
  Stress      - money movement driven by fear or obligation
  Regret      - purchases that leave a hollow feeling
  Neutral     - routine, neither happy nor unhappy
"""

import json
import re
import requests
from transaction_parser import Transaction

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"

HONDA_TAGS = ["Gratitude", "Joy", "Meaning", "Love", "Stress", "Regret", "Neutral"]

# PERMA mapping for Version 2 psychology layer
PERMA_MAP = {
    "Gratitude": "Meaning",
    "Joy": "Pleasure",
    "Meaning": "Meaning",
    "Love": "Relationships",
    "Stress": None,
    "Regret": None,
    "Neutral": "Engagement",
}

# Rule-based quick tags for known income types (no LLM needed)
_INCOME_KEYWORDS = ["salary", "bonus", "dividend", "interest", "freelance", "income", "refund", "cashback"]
_CHARITY_KEYWORDS = ["donation", "charity", "give india", "milaap", "ketto", "namdroling", "dhamma"]
_FAMILY_KEYWORDS = ["birthday", "anniversary", "family", "parents", "mother", "father", "gift"]


def _quick_tag(tx: Transaction) -> dict | None:
    """Fast rule-based tags for unambiguous cases."""
    desc_lower = tx.description.lower()

    if tx.tx_type == "Credit":
        for kw in _INCOME_KEYWORDS:
            if kw in desc_lower:
                return {
                    "honda_tag": "Gratitude",
                    "arigato_score": 88,
                    "happy_money": True,
                    "honda_reason": "Income received — the classic moment to practice Arigato money: gratitude for money arriving."
                }
        return {
            "honda_tag": "Gratitude",
            "arigato_score": 82,
            "happy_money": True,
            "honda_reason": "Money received. Ken Honda teaches: greet incoming money with gratitude."
        }

    for kw in _CHARITY_KEYWORDS:
        if kw in desc_lower:
            return {
                "honda_tag": "Meaning",
                "arigato_score": 93,
                "happy_money": True,
                "honda_reason": "Charity or spiritual offering — highest-scoring category in Happy Money. Money given with an open hand returns multiplied."
            }

    return None


def _ollama_honda_tag(tx: Transaction) -> dict:
    """
    Ask Qwen3 to apply Ken Honda's Happy Money framework to a transaction.
    """
    tags_str = ", ".join(HONDA_TAGS)

    prompt = f"""You are a financial well-being coach inspired by Ken Honda's Happy Money philosophy.

Ken Honda teaches that money is energy. Every transaction carries an emotional signature:
- Happy Money: flows with gratitude, joy, meaning, love — creates lasting well-being
- Unhappy Money: flows with stress, regret, obligation, or numbness — drains energy

Transaction to analyse:
  Date: {tx.date}
  Description: {tx.description}
  Merchant: {tx.merchant_clean or tx.description}
  Category: {tx.category}
  Amount: ₹{tx.amount:,.0f}
  Type: {tx.tx_type}

Assign:
1. honda_tag: one of [{tags_str}] — the dominant emotional signature of this money movement
2. arigato_score: integer 0–100 — how much this transaction aligns with Happy Money principles
   (100 = deeply meaningful, 0 = pure stress/regret)
3. happy_money: true if arigato_score >= 55, else false
4. honda_reason: one clear sentence explaining the score from a Ken Honda perspective

Respond ONLY with valid JSON, no markdown, no explanation:
{{
  "honda_tag": "<tag>",
  "arigato_score": <integer>,
  "happy_money": <true|false>,
  "honda_reason": "<one sentence>"
}}"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=30
        )
        response.raise_for_status()
        raw = response.json().get("response", "")
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            result = json.loads(match.group())
            # Validate
            if result.get("honda_tag") not in HONDA_TAGS:
                result["honda_tag"] = "Neutral"
            score = result.get("arigato_score", 50)
            result["arigato_score"] = max(0, min(100, int(score)))
            result["happy_money"] = result["arigato_score"] >= 55
            return result

    except Exception as e:
        print(f"[honda_tagger] Ollama error for '{tx.description}': {e}")

    # Fallback: use category-based heuristics
    return _category_fallback(tx)


def _category_fallback(tx: Transaction) -> dict:
    """Heuristic fallback when Ollama is unavailable."""
    category_defaults = {
        "Travel":        ("Joy",       78, True,  "Travel tends to create lasting positive memories according to happiness research."),
        "Learning":      ("Meaning",   82, True,  "Investment in knowledge aligns with long-term flourishing."),
        "Health":        ("Meaning",   70, True,  "Spending on health is an act of self-respect."),
        "Family":        ("Love",      88, True,  "Money spent to nurture relationships is among the highest-return spending."),
        "Charity":       ("Meaning",   93, True,  "Giving freely is the purest Happy Money flow."),
        "Entertainment": ("Joy",       60, True,  "Entertainment can bring genuine pleasure when chosen consciously."),
        "Food":          ("Neutral",   48, False, "Food delivery is convenient but rarely creates lasting satisfaction."),
        "Shopping":      ("Neutral",   32, False, "Material purchases often feel less satisfying than anticipated."),
        "Bills":         ("Stress",    20, False, "Obligatory payments carry stress energy."),
        "Investments":   ("Gratitude", 70, True,  "Investing for the future is a form of self-care and gratitude."),
        "Income":        ("Gratitude", 87, True,  "Welcome this money with an Arigato."),
    }
    cat = tx.category or "Shopping"
    tag, score, happy, reason = category_defaults.get(cat, ("Neutral", 45, False, "No specific happiness signal detected."))
    return {
        "honda_tag": tag,
        "arigato_score": score,
        "happy_money": happy,
        "honda_reason": reason
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def tag_transaction(tx: Transaction) -> Transaction:
    """Apply Ken Honda Happy Money tags to a single transaction."""
    quick = _quick_tag(tx)
    if quick:
        result = quick
    else:
        result = _ollama_honda_tag(tx)

    tx.honda_tag = result["honda_tag"]
    tx.arigato_score = result["arigato_score"]
    tx.happy_money = result["happy_money"]
    tx.honda_reason = result["honda_reason"]
    return tx


def tag_batch(transactions: list[Transaction]) -> list[Transaction]:
    """Tag all transactions."""
    return [tag_transaction(tx) for tx in transactions]


def get_energy_flow(transactions: list[Transaction]) -> dict:
    """
    Returns Ken Honda's Money Energy Flow summary:
      - Received with Gratitude (income)
      - Spent with Joy (happy_money=True, debit)
      - Spent with Stress (honda_tag in [Stress, Regret], debit)
      - Creating Meaning (honda_tag in [Meaning, Love], debit)
    """
    flow = {
        "received_with_gratitude": 0.0,
        "spent_with_joy": 0.0,
        "spent_with_stress": 0.0,
        "creating_meaning": 0.0,
        "neutral_spend": 0.0,
    }

    for tx in transactions:
        if tx.tx_type == "Credit":
            flow["received_with_gratitude"] += tx.amount
        elif tx.honda_tag in ("Meaning", "Love"):
            flow["creating_meaning"] += tx.amount
        elif tx.honda_tag in ("Stress", "Regret"):
            flow["spent_with_stress"] += tx.amount
        elif tx.happy_money:
            flow["spent_with_joy"] += tx.amount
        else:
            flow["neutral_spend"] += tx.amount

    return flow


def get_arigato_leaderboard(transactions: list[Transaction], top_n: int = 5) -> list[Transaction]:
    """Top N happiest transactions."""
    debits = [tx for tx in transactions if tx.tx_type == "Debit"]
    return sorted(debits, key=lambda t: t.arigato_score, reverse=True)[:top_n]


def get_regret_list(transactions: list[Transaction], top_n: int = 5) -> list[Transaction]:
    """Top N regret-flagged transactions."""
    debits = [tx for tx in transactions if tx.tx_type == "Debit"]
    return sorted(debits, key=lambda t: t.arigato_score)[:top_n]
