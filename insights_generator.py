"""
insights_generator.py
Uses Qwen3 to generate narrative insights combining:
  - Happiness ROI data
  - Ken Honda energy flow
  - User's own journal patterns
  - Well-being score

Generates 3-5 insights in Ken Honda's voice, grounded in the user's actual data.
"""

import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


def generate_insights(
    energy_flow: dict,
    roi_data: dict,
    wellbeing: dict,
    top_happy: list,
    top_regret: list,
    reallocation_hint: str = ""
) -> list[str]:
    """
    Generate 3-5 narrative insights in Ken Honda's Happy Money spirit.
    Returns a list of strings (each one insight).
    """
    # Build a compact data summary for the prompt
    flow_summary = {k: f"₹{v:,.0f}" for k, v in energy_flow.items() if v > 0}

    roi_summary = {
        cat: {"roi": d["happiness_roi"], "spend": f"₹{d['spend']:,.0f}"}
        for cat, d in roi_data.items()
    }

    happy_tx_summary = [
        f"{tx.merchant_clean} (₹{tx.amount:,.0f}, Arigato: {tx.arigato_score})"
        for tx in top_happy
    ]
    regret_tx_summary = [
        f"{tx.merchant_clean} (₹{tx.amount:,.0f}, Arigato: {tx.arigato_score})"
        for tx in top_regret
    ]

    prompt = f"""You are a financial well-being coach inspired by Ken Honda's Happy Money philosophy.

Here is a summary of the user's spending this month:

Money Energy Flow:
{json.dumps(flow_summary, indent=2)}

Happiness ROI by Category (score 0-100):
{json.dumps(roi_summary, indent=2)}

Overall Well-Being Score: {wellbeing['score']}/100 ({wellbeing['label']})

Highest Arigato transactions:
{chr(10).join(happy_tx_summary) or 'None identified'}

Lowest Arigato transactions:
{chr(10).join(regret_tx_summary) or 'None identified'}

Reallocation suggestion: {reallocation_hint or 'None'}

Write exactly 4 insights in Ken Honda's warm, wise voice:
- Ground each insight in the actual data above
- Reference specific categories, amounts, or patterns
- Tone: reflective, non-judgmental, encouraging
- Length: 2-3 sentences each
- Do not use bullet points in the insights themselves; write flowing prose
- End the last insight with a gentle Arigato practice suggestion

Return ONLY a JSON array of 4 strings. No markdown, no explanation.
["insight 1", "insight 2", "insight 3", "insight 4"]"""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.4, "num_predict": 800}
            },
            timeout=45
        )
        raw = response.json().get("response", "")
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            insights = json.loads(match.group())
            if isinstance(insights, list) and insights:
                return [str(i) for i in insights[:5]]
    except Exception as e:
        print(f"[insights_generator] Ollama error: {e}")

    # Fallback: template insights
    return _template_insights(energy_flow, roi_data, wellbeing, top_happy, top_regret)


def _template_insights(energy_flow, roi_data, wellbeing, top_happy, top_regret) -> list[str]:
    """Static template fallback when Ollama is unavailable."""
    insights = []

    # Insight 1: Energy flow
    joy_spend = energy_flow.get("spent_with_joy", 0)
    stress_spend = energy_flow.get("spent_with_stress", 0)
    if joy_spend > stress_spend:
        insights.append(
            f"Your money energy this month shows more joy than stress — "
            f"₹{joy_spend:,.0f} flowed with positive intention compared to ₹{stress_spend:,.0f} under pressure. "
            f"Ken Honda would say this is a sign of growing financial awareness."
        )
    else:
        insights.append(
            f"A notable portion of your spending — ₹{stress_spend:,.0f} — carried stress energy this month. "
            f"Ken Honda teaches that noticing this pattern is the first step toward changing it. "
            f"There is no judgment here, only awareness."
        )

    # Insight 2: Highest ROI category
    if roi_data:
        best_cat = max(roi_data, key=lambda c: roi_data[c]["happiness_roi"])
        best_roi = roi_data[best_cat]
        insights.append(
            f"{best_cat} spending returned the highest happiness this month "
            f"(score: {best_roi['happiness_roi']}/100 on ₹{best_roi['spend']:,.0f}). "
            f"This is where your money and your joy are most aligned — a pattern worth nurturing."
        )

    # Insight 3: Lowest ROI category
    if roi_data:
        worst_cat = min(roi_data, key=lambda c: roi_data[c]["happiness_roi"])
        worst_roi = roi_data[worst_cat]
        if worst_roi["happiness_roi"] < 40:
            insights.append(
                f"{worst_cat} spending (₹{worst_roi['spend']:,.0f}) generated the fewest positive memories. "
                f"This is not a call to deprive yourself — Ken Honda never moralises about spending. "
                f"It is simply an invitation to ask: does this money flow with joy, or with habit?"
            )

    # Insight 4: Arigato practice
    if top_happy:
        tx = top_happy[0]
        insights.append(
            f"Your highest Arigato moment this month was {tx.merchant_clean} "
            f"(score: {tx.arigato_score}/100). "
            f"Before your next big purchase, pause and ask: will I remember this with warmth? "
            f"That question, held lightly, is the essence of Happy Money. Arigato."
        )
    else:
        insights.append(
            "Ken Honda's core practice is simple: say Arigato — thank you — "
            "to money as it arrives, and again as it leaves. "
            "Even one mindful moment with each transaction begins to shift the energy of your financial life."
        )

    return insights
