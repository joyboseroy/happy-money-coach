"""
categorizer.py
Maps transaction descriptions to spending categories.
Strategy:
  1. Check merchant knowledge base (fast, deterministic)
  2. Fall back to Qwen3 via Ollama for unknowns
  3. Cache new LLM results back into the KB
"""

import json
import re
import requests
from pathlib import Path
from typing import Optional

from transaction_parser import Transaction

MERCHANT_KB_PATH = Path(__file__).resolve().parent / "data" / "merchant_kb.json"

CATEGORIES = [
    "Food", "Travel", "Learning", "Health", "Family",
    "Entertainment", "Shopping", "Bills", "Charity", "Investments", "Income"
]

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


def _load_kb() -> dict:
    with open(MERCHANT_KB_PATH) as f:
        return json.load(f)


def _save_kb(kb: dict):
    with open(MERCHANT_KB_PATH, "w") as f:
        json.dump(kb, f, indent=2)


def _normalize(text: str) -> str:
    """Lowercase, normalise separators for KB lookup."""
    text = text.lower()
    # Replace common separators with spaces so "NEFT CR-ERICSSON" matches "neft cr"
    text = re.sub(r"[-/|_]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _kb_lookup(description: str, kb: dict) -> Optional[dict]:
    """
    Return KB entry if any merchant keyword appears in the description.
    Longer keys are checked first so 'neft cr salary' beats 'neft cr'.
    Credit transactions are forced to Income category regardless of KB.
    """
    norm = _normalize(description)

    # Sort by key length descending — more specific matches win
    sorted_merchants = sorted(kb["merchants"].items(), key=lambda x: len(x[0]), reverse=True)

    for merchant_key, data in sorted_merchants:
        norm_key = _normalize(merchant_key)
        if norm_key in norm:
            return data
    return None


def _ollama_classify(description: str) -> dict:
    """
    Ask Qwen3 to classify the transaction.
    Returns {"category": ..., "merchant_clean": ..., "happy_money_prior": ...}
    """
    categories_str = ", ".join(CATEGORIES)
    prompt = f"""You are a financial transaction classifier for Indian bank statements.

Transaction description: "{description}"

Classify this transaction. Respond ONLY with valid JSON, no explanation, no markdown.

Return exactly this structure:
{{
  "category": "<one of: {categories_str}>",
  "merchant_clean": "<human-readable merchant name>",
  "happy_money_prior": <float 0.0 to 1.0, estimate probability this spending creates happiness>
}}

Examples:
- "IRCTC WEB BOOKING" → {{"category": "Travel", "merchant_clean": "IRCTC", "happy_money_prior": 0.78}}
- "SWIGGY LTD" → {{"category": "Food", "merchant_clean": "Swiggy", "happy_money_prior": 0.45}}
- "AMAZON MKTPLACE PMTS" → {{"category": "Shopping", "merchant_clean": "Amazon", "happy_money_prior": 0.35}}
- "SALARY CREDIT" → {{"category": "Income", "merchant_clean": "Salary", "happy_money_prior": 0.85}}
"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=30
        )
        response.raise_for_status()
        raw = response.json().get("response", "")

        # Strip any markdown fences
        raw = re.sub(r"```json|```", "", raw).strip()

        # Extract first JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())

    except Exception as e:
        print(f"[categorizer] Ollama error for '{description}': {e}")

    # Fallback defaults
    return {
        "category": "Shopping",
        "merchant_clean": description[:40],
        "happy_money_prior": 0.35
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _make_clean_name(description: str) -> str:
    """
    Convert raw bank narration to a human-readable name.
    e.g. 'NEFT CR-ERICSSON INDIA PVT LTD SALARY APR' -> 'Ericsson India (Salary)'
         'UPI/ZERODHA/SHARES' -> 'Zerodha'
         'ATM WD 001234 SBI BRANCH' -> 'ATM Withdrawal'
    """
    desc = description.strip()

    # Strip common prefixes
    prefixes = ["NEFT CR-", "NEFT DR-", "IMPS CR-", "IMPS DR-",
                "UPI/", "RTGS CR-", "RTGS DR-", "ECS DR-", "ECS CR-",
                "NACH DR-", "BIL/", "MMT/", "INF/", "ATF/"]
    for p in prefixes:
        if desc.upper().startswith(p):
            desc = desc[len(p):]
            break

    # Take first meaningful segment (before next slash or long numeric)
    segment = re.split(r"[/|]", desc)[0].strip()
    # Remove trailing reference numbers
    segment = re.sub(r"\s+\d{6,}.*$", "", segment).strip()
    # Title-case and cap length
    return segment[:40].title() if segment else description[:40]


def categorize(tx: Transaction) -> Transaction:
    """Enrich a single Transaction with category and merchant_clean."""
    # Credits are always Income — salary, refunds, transfers in
    if tx.tx_type == "Credit":
        tx.category = "Income"
        tx.merchant_clean = tx.description[:50]
        return tx

    kb = _load_kb()
    result = _kb_lookup(tx.description, kb)

    if result is None:
        result = _ollama_classify(tx.description)
        # Cache into KB for future runs
        norm_key = _normalize(tx.description).split()[0] if tx.description else "unknown"
        if norm_key and len(norm_key) > 2 and norm_key not in kb["merchants"]:
            kb["merchants"][norm_key] = result
            _save_kb(kb)

    tx.category = result.get("category", "Shopping")
    tx.merchant_clean = result.get("merchant_clean", tx.description[:50])

    # If merchant_clean wasn't set by KB (KB entries don't have it), use cleaned description
    if not tx.merchant_clean or tx.merchant_clean == tx.description[:40]:
        tx.merchant_clean = _make_clean_name(tx.description)

    return tx


def categorize_batch(transactions: list[Transaction]) -> list[Transaction]:
    """Categorize a list of transactions."""
    return [categorize(tx) for tx in transactions]


def get_category_summary(transactions: list[Transaction]) -> dict:
    """
    Returns {category: {"total_spend": float, "count": int, "transactions": [...]}}
    Only includes Debit transactions.
    """
    summary = {}
    for tx in transactions:
        if tx.tx_type == "Credit":
            continue
        cat = tx.category or "Shopping"
        if cat not in summary:
            summary[cat] = {"total_spend": 0.0, "count": 0, "transactions": []}
        summary[cat]["total_spend"] += tx.amount
        summary[cat]["count"] += 1
        summary[cat]["transactions"].append(tx)
    return summary
