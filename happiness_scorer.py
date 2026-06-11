"""
happiness_scorer.py
Computes Happiness ROI per spending category.

Formula:
  happiness_roi = (positive_journal_mentions * intensity_weight) / log(spend + 1)
  Normalised to 0-100 across all categories.

Also computes a composite "Well-Being Score" per month.
"""

import math
from transaction_parser import Transaction
from journal_parser import JournalEvent, positive_category_counts


def compute_happiness_roi(
    category_summary: dict,         # from categorizer.get_category_summary()
    journal_events: list[JournalEvent],
) -> dict[str, dict]:
    """
    Returns per-category happiness ROI:
    {
      "Travel": {
        "spend": 12000,
        "positive_mentions": 4,
        "happiness_roi": 84,
        "happy_money_flag": True,
        "interpretation": "..."
      }, ...
    }
    """
    pos_counts = positive_category_counts(journal_events)

    # Weighted mentions (intensity matters)
    weighted_mentions: dict[str, float] = {}
    for e in journal_events:
        if e.sentiment == "positive":
            for cat in e.categories:
                weight = e.intensity / 10.0
                weighted_mentions[cat] = weighted_mentions.get(cat, 0.0) + weight

    raw_scores = {}
    for cat, data in category_summary.items():
        if cat == "Income":
            continue
        spend = data["total_spend"]
        mentions = weighted_mentions.get(cat, 0.0)
        # Avoid divide-by-zero; log prevents very large spends dominating
        if spend > 0:
            raw_score = (mentions + 0.1) / math.log(spend + 1)
        else:
            raw_score = 0.0
        raw_scores[cat] = raw_score

    # Normalise to 0-100
    if raw_scores:
        max_score = max(raw_scores.values()) or 1
        min_score = min(raw_scores.values())
        range_score = max_score - min_score or 1
    else:
        max_score = min_score = range_score = 1

    results = {}
    for cat, data in category_summary.items():
        if cat == "Income":
            continue
        raw = raw_scores.get(cat, 0.0)
        normalised = int(((raw - min_score) / range_score) * 85) + 10  # 10-95 range
        normalised = max(5, min(95, normalised))

        pos_mentions = pos_counts.get(cat, 0)
        spend = data["total_spend"]

        results[cat] = {
            "spend": spend,
            "positive_mentions": pos_mentions,
            "happiness_roi": normalised,
            "happy_money_flag": normalised >= 55,
            "interpretation": _interpret(cat, normalised, spend, pos_mentions)
        }

    return results


def _interpret(category: str, score: int, spend: float, mentions: int) -> str:
    """Generate a plain-English interpretation."""
    if score >= 75:
        return (
            f"{category} spending delivers strong happiness return. "
            f"₹{spend:,.0f} spent here generated {mentions} positive memories."
        )
    elif score >= 50:
        return (
            f"{category} spending is moderately fulfilling. "
            f"Consider whether more intentional choices here could lift the return."
        )
    elif score >= 25:
        return (
            f"{category} spending (₹{spend:,.0f}) generates limited positive memories. "
            f"This may be worth examining."
        )
    else:
        return (
            f"{category} spending rarely appears in positive journal entries. "
            f"Ken Honda would ask: is this money flowing with joy or obligation?"
        )


def compute_wellbeing_score(transactions: list[Transaction]) -> dict:
    """
    Overall monthly well-being score derived from Arigato scores.
    """
    debits = [tx for tx in transactions if tx.tx_type == "Debit" and tx.arigato_score > 0]
    if not debits:
        return {"score": 50, "label": "Neutral", "insight": "Not enough data."}

    # Weighted average: larger spends weighted more
    total_spend = sum(tx.amount for tx in debits)
    if total_spend == 0:
        return {"score": 50, "label": "Neutral", "insight": "Not enough spend data."}

    weighted_sum = sum(tx.arigato_score * tx.amount for tx in debits)
    score = int(weighted_sum / total_spend)

    if score >= 70:
        label = "Flourishing"
        insight = "Your money is flowing with purpose and joy this month."
    elif score >= 55:
        label = "Growing"
        insight = "Good momentum — a few spending shifts could lift this further."
    elif score >= 40:
        label = "Neutral"
        insight = "Mixed signals. Some joyful spending, some stress patterns."
    else:
        label = "Draining"
        insight = "Money appears to be flowing with more stress than joy. Worth reflecting on."

    return {"score": score, "label": label, "insight": insight}


def spending_reallocation_suggestion(
    roi_data: dict,
    top_regret_spend: float,
    top_joy_category: str
) -> str:
    """
    Generate a reallocation suggestion:
    'Consider redirecting X% of [low-ROI] toward [high-ROI].'
    """
    low_cats = [c for c, d in roi_data.items() if d["happiness_roi"] < 35 and d["spend"] > 2000]
    if not low_cats or not top_joy_category:
        return ""

    lowest = min(low_cats, key=lambda c: roi_data[c]["happiness_roi"])
    low_spend = roi_data[lowest]["spend"]
    redirect = low_spend * 0.15  # suggest shifting 15%

    return (
        f"Consider redirecting ₹{redirect:,.0f} (15%) from {lowest} "
        f"toward {top_joy_category}. Based on your own patterns, "
        f"{top_joy_category} delivers {roi_data.get(top_joy_category, {}).get('happiness_roi', 70)} "
        f"happiness points vs {roi_data[lowest]['happiness_roi']} for {lowest}."
    )
