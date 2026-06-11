"""
ethical_insights_generator.py

Generates narrative insights combining Happy Money scores and ethical lens analysis.
LLM is used only for explanation text. All scoring is deterministic (ontology-based).

Two output modes:
  1. Per-transaction lens explanation (short, tradition-specific)
  2. Monthly ethical summary (longer, cross-tradition synthesis)
"""

import json
import re
import requests

from perspective_engine import EthicalProfile, EthicalScore, LENS_DESCRIPTIONS, get_lens_summary

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


def generate_monthly_ethical_summary(
    lens: str,
    lens_summary: dict,
    virtue_dashboard: dict,
    top_positive: list,
    top_concern: list,
    use_llm: bool = True
) -> str:
    """
    Generate a paragraph-length narrative summary for a given ethical lens.
    """
    if not use_llm:
        return _template_summary(lens, lens_summary, virtue_dashboard)

    lens_full = LENS_DESCRIPTIONS.get(lens, lens.title())

    lens_prompts = {
        "buddhist": (
            "You are a Buddhist monk and ethics teacher. "
            "You speak with warmth, precision, and without judgment. "
            "You reference specific Pali/Sanskrit terms naturally."
        ),
        "christian": (
            "You are a Christian pastoral counsellor with deep knowledge of "
            "scripture and Christian social ethics. You are affirming, not condemning."
        ),
        "jewish": (
            "You are a rabbi with knowledge of Talmud and contemporary Jewish ethics. "
            "You speak with warmth and the tradition's characteristic embrace of questioning."
        ),
        "islamic": (
            "You are an Islamic scholar familiar with fiqh, maqasid al-shariah, and "
            "contemporary Islamic finance. You are clear about principles without being harsh."
        ),
        "stoic": (
            "You are a Stoic philosopher in the tradition of Marcus Aurelius and Epictetus. "
            "You speak with directness and intellectual rigour. You quote Stoic texts naturally."
        ),
        "universal": (
            "You are a cross-cultural ethicist drawing on positive psychology and "
            "human values research. You are evidence-informed and non-dogmatic."
        ),
    }

    persona = lens_prompts.get(lens, "You are an ethical philosophy scholar.")

    positive_cats = ", ".join([p.category for p in top_positive[:3]]) if top_positive else "none identified"
    concern_cats = ", ".join([p.category for p in top_concern[:3]]) if top_concern else "none identified"

    virtue_str = ", ".join([
        f"{v}: {pct}%" for v, pct in
        sorted(virtue_dashboard.get("by_pct", {}).items(), key=lambda x: x[1], reverse=True)[:5]
    ])

    prompt = f"""{persona}

Provide a {lens_full} perspective on a person's monthly spending.

Summary data:
  Overall {lens_full} score: {lens_summary.get('score_avg', 0):.2f} ({lens_summary.get('score_label', 'Neutral')})
  Most ethically positive categories: {positive_cats}
  Categories with concerns: {concern_cats}
  Best category by this lens: {lens_summary.get('best_category', 'unknown')}
  Virtue distribution of spending: {virtue_str}

Write 2-3 paragraphs of reflective commentary from your tradition's perspective.
Be specific about this tradition's ethical concepts. Be warm and non-judgmental.
Acknowledge complexity where it exists. End with one concrete suggestion or practice.
Do not start with "I" and do not use the person's name."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.35, "num_predict": 400}
            },
            timeout=45
        )
        text = response.json().get("response", "").strip()
        text = re.sub(r"^(Sure|Certainly|Of course)[,!.]?\s*", "", text)
        return text if text else _template_summary(lens, lens_summary, virtue_dashboard)
    except Exception as e:
        print(f"[ethical_insights] LLM error ({lens}): {e}")
        return _template_summary(lens, lens_summary, virtue_dashboard)


def _template_summary(lens: str, lens_summary: dict, virtue_dashboard: dict) -> str:
    """Static template fallback."""
    lens_full = LENS_DESCRIPTIONS.get(lens, lens.title())
    score_avg = lens_summary.get("score_avg", 0)
    best_cat = lens_summary.get("best_category", "Learning")
    worst_cat = lens_summary.get("worst_category", "Shopping")
    score_label = lens_summary.get("score_label", "Neutral")

    templates = {
        "buddhist": (
            f"From a Buddhist perspective, this month's spending shows a {score_label.lower()} relationship "
            f"with the principle of right livelihood and dana (generosity). "
            f"{best_cat} spending aligns well with the cultivation of wisdom and wholesome states. "
            f"{worst_cat} spending warrants reflection: the Buddha identified craving (tanha) as the root of dissatisfaction, "
            f"and it is worth asking whether these purchases arose from genuine need or from restlessness. "
            f"A simple practice: before the next discretionary purchase, pause for three breaths and ask: "
            f"'Does this arise from contentment or from craving?'"
        ),
        "christian": (
            f"From a Christian stewardship perspective, this month's spending is {score_label.lower()} overall. "
            f"{best_cat} reflects the call to invest in what endures — relationships, growth, and service. "
            f"{worst_cat} spending invites the question Jesus posed: 'What does it profit a person to gain the world and lose their soul?' "
            f"The practice of tithing — setting aside a portion for charity before spending on desire — is worth considering."
        ),
        "jewish": (
            f"Through a Jewish ethical lens, this month shows {score_label.lower()} alignment with the tradition's values. "
            f"{best_cat} reflects the mitzvah of talmud (learning) or tzedakah (generosity) at its best. "
            f"{worst_cat} spending raises bal tashchit considerations — the principle that unnecessary waste or excess "
            f"is a form of damage to the world. The tradition suggests reviewing one's 'accounting of the soul' (cheshbon ha-nefesh) monthly."
        ),
        "islamic": (
            f"From an Islamic ethics perspective, this month's spending merits {score_label.lower()} evaluation. "
            f"{best_cat} aligns with the maqasid al-shariah — the objectives of Islamic law in preserving life, intellect, and progeny. "
            f"{worst_cat} requires reflection on israf (excess) — the Quran cautions, 'Do not be extravagant; indeed He does not like the extravagant.' "
            f"The practice of writing intentions (niyyah) before major purchases can help align spending with values."
        ),
        "stoic": (
            f"Viewed through a Stoic lens, this month's spending scores {score_label.lower()} overall. "
            f"{best_cat} reflects what Epictetus called kathêkon — fulfilling one's proper duties and roles. "
            f"For {worst_cat}, Marcus Aurelius offers a useful question: 'Is this impression something in my control or not?' "
            f"Much consumer spending arises from impressions (phantasiai) we have not examined. "
            f"The Stoic practice: write one sentence before each significant purchase describing which virtue it serves."
        ),
        "universal": (
            f"Drawing on cross-cultural values research, this month's spending shows {score_label.lower()} alignment "
            f"with widely-shared human values. {best_cat} consistently ranks among the highest-wellbeing spending categories "
            f"across cultures and empirical studies. {worst_cat} tends to score lower on lasting satisfaction metrics. "
            f"Research from positive psychology (Dunn, Norton; Kahneman) consistently shows that spending on experiences, "
            f"relationships, and others generates more durable well-being than spending on material goods."
        ),
    }

    return templates.get(lens, f"This month's spending scores {score_label.lower()} under {lens_full} principles.")


def generate_cross_tradition_comparison(
    category: str,
    scores: dict,
    use_llm: bool = True
) -> str:
    """
    For a single category, generate a brief cross-tradition comparison.
    Shows where traditions agree and where they diverge.
    """
    if not use_llm:
        return _template_cross_comparison(category, scores)

    score_lines = "\n".join([
        f"  {LENS_DESCRIPTIONS.get(lens, lens)}: {data.score_label} ({data.score:.2f}) — {data.reason[:80]}"
        for lens, data in scores.items()
        if isinstance(data, EthicalScore)
    ])

    prompt = f"""You are a comparative religion and ethics scholar.

Briefly compare how different ethical traditions assess spending on: {category}

Tradition scores:
{score_lines}

Write 2-3 sentences that:
1. Note where traditions converge
2. Note the most interesting point of divergence
3. Identify what the divergence reveals about each tradition's core values

Be concise and intellectually precise. No moralising."""

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
        return text if text else _template_cross_comparison(category, scores)
    except Exception:
        return _template_cross_comparison(category, scores)


def _template_cross_comparison(category: str, scores: dict) -> str:
    """Template fallback for cross-tradition comparison."""
    high_lenses = [lens for lens, data in scores.items()
                   if isinstance(data, EthicalScore) and data.score >= 0.60]
    low_lenses = [lens for lens, data in scores.items()
                  if isinstance(data, EthicalScore) and data.score < 0.10]

    if high_lenses and not low_lenses:
        return (f"Spending on {category} receives broadly positive assessment across traditions. "
                f"All six lenses find it consistent with their core values of "
                f"{', '.join([data.score_label for data in list(scores.values())[:2]])}.")
    elif low_lenses:
        return (f"Spending on {category} raises concerns across multiple traditions "
                f"({', '.join(low_lenses)}), though for different reasons. "
                f"The convergence of concern itself is worth noting.")
    else:
        return (f"Traditions offer mixed perspectives on {category} spending. "
                f"The divergence reflects different core values: "
                f"some traditions emphasise intention, others emphasise outcome.")
