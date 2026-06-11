"""
journal_parser.py
Parses happiness journal entries across three input levels:

Level 1 - No journal: user selects happy purchases from a list
Level 2 - Monthly reflection: user types 3 moments, LLM extracts events
Level 3 - Journal upload: CSV or free text diary

Outputs a list of JournalEvent with date, description, sentiment, and keywords.
"""

import json
import re
import requests
from dataclasses import dataclass
from typing import Optional

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "qwen2.5:7b"


@dataclass
class JournalEvent:
    date: str
    raw_text: str
    sentiment: str        # "positive" / "negative" / "neutral"
    intensity: int        # 1-10
    keywords: list[str]   # ["family", "travel", "meditation", ...]
    categories: list[str] # mapped to spending categories


def _extract_events_llm(text: str, date: str = "") -> list[dict]:
    """Ask Qwen3 to extract emotional events from a journal entry."""
    prompt = f"""Analyse this journal entry and extract happiness/emotion signals.

Entry (date: {date or 'unknown'}):
"{text}"

Return ONLY valid JSON array. Each element:
{{
  "sentiment": "positive" | "negative" | "neutral",
  "intensity": <1-10>,
  "keywords": ["<keyword1>", "<keyword2>"],
  "categories": ["<from: Travel, Learning, Food, Family, Health, Entertainment, Shopping, Charity>"]
}}

Extract up to 3 events. If only one mood, return a single-element array.
No markdown, no explanation."""

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.1}
            },
            timeout=20
        )
        raw = response.json().get("response", "")
        raw = re.sub(r"```json|```", "", raw).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
    except Exception as e:
        print(f"[journal_parser] LLM error: {e}")

    # Simple fallback: detect valence from keywords
    pos_words = {"wonderful", "great", "happy", "joy", "love", "amazing", "peaceful",
                 "meaningful", "warm", "fulfilled", "recharged", "magical"}
    neg_words = {"lonely", "sad", "stressed", "anxious", "hollow", "regret", "tired", "empty"}
    lower_text = text.lower()
    if any(w in lower_text for w in pos_words):
        sentiment = "positive"
        intensity = 7
    elif any(w in lower_text for w in neg_words):
        sentiment = "negative"
        intensity = 6
    else:
        sentiment = "neutral"
        intensity = 4

    return [{"sentiment": sentiment, "intensity": intensity, "keywords": [], "categories": []}]


# ---------------------------------------------------------------------------
# Level 1: No journal — user picks from purchase list
# ---------------------------------------------------------------------------

def from_user_selections(selected_descriptions: list[str], date_range: str = "") -> list[JournalEvent]:
    """
    User selected these transactions as their happy memories.
    Each gets a synthetic positive journal event.
    """
    events = []
    for desc in selected_descriptions:
        events.append(JournalEvent(
            date=date_range,
            raw_text=f"User marked as happy: {desc}",
            sentiment="positive",
            intensity=8,
            keywords=[desc.lower()[:30]],
            categories=[]
        ))
    return events


# ---------------------------------------------------------------------------
# Level 2: Monthly reflection text
# ---------------------------------------------------------------------------

def from_reflection(reflection_text: str, date: str = "") -> list[JournalEvent]:
    """
    User typed their top 3 moments. Extract events via LLM.
    """
    extracted = _extract_events_llm(reflection_text, date)
    events = []
    for e in extracted:
        events.append(JournalEvent(
            date=date,
            raw_text=reflection_text,
            sentiment=e.get("sentiment", "neutral"),
            intensity=e.get("intensity", 5),
            keywords=e.get("keywords", []),
            categories=e.get("categories", [])
        ))
    return events


# ---------------------------------------------------------------------------
# Level 3: CSV journal upload
# ---------------------------------------------------------------------------

def from_csv(file_obj) -> list[JournalEvent]:
    """
    Parse a journal CSV with columns: Date, Entry
    Each row processed through LLM.
    """
    import pandas as pd
    try:
        df = pd.read_csv(file_obj)
    except Exception as e:
        raise ValueError(f"Could not read journal CSV: {e}")

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns or "entry" not in df.columns:
        raise ValueError("Journal CSV must have 'Date' and 'Entry' columns.")

    events = []
    for _, row in df.iterrows():
        date = str(row["date"]).strip()
        entry = str(row["entry"]).strip()
        if not entry or entry.lower() == "nan":
            continue
        extracted = _extract_events_llm(entry, date)
        for e in extracted:
            events.append(JournalEvent(
                date=date,
                raw_text=entry,
                sentiment=e.get("sentiment", "neutral"),
                intensity=e.get("intensity", 5),
                keywords=e.get("keywords", []),
                categories=e.get("categories", [])
            ))

    return events


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def positive_category_counts(events: list[JournalEvent]) -> dict[str, int]:
    """
    Returns {category: count_of_positive_mentions}
    Used for Happiness ROI correlation.
    """
    counts: dict[str, int] = {}
    for e in events:
        if e.sentiment == "positive":
            for cat in e.categories:
                counts[cat] = counts.get(cat, 0) + 1
    return counts


def keyword_frequency(events: list[JournalEvent], sentiment: str = "positive") -> dict[str, int]:
    """Return keyword frequency for a given sentiment."""
    freq: dict[str, int] = {}
    for e in events:
        if e.sentiment == sentiment:
            for kw in e.keywords:
                freq[kw] = freq.get(kw, 0) + 1
    return dict(sorted(freq.items(), key=lambda x: x[1], reverse=True))
