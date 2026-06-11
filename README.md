# 🙏 Happy Money Coach

> *"The key to becoming rich in the truest sense is not working harder, but working on your relationship with money."*
> — Ken Honda

An open-source Streamlit app that analyses your bank statement through two lenses:

1. **Ken Honda's Happy Money philosophy** — every transaction tagged with an emotional signature (Gratitude / Joy / Meaning / Love / Stress / Regret) and an Arigato Score (0–100)
2. **Comparative ethical analysis** — spending scored across six traditions: Universal Human Values, Buddhist, Christian, Jewish (halacha), Islamic (shariah), and Stoic philosophy

Unlike budgeting apps that optimise net worth, this asks: *does your money flow with joy?*

---

## Quickstart

```bash
git clone https://github.com/joyboseroy/happy-money-coach
cd happy-money-coach
pip install -r requirements.txt
streamlit run app.py
```

Loads with demo data immediately. No uploads or Ollama required to explore.

---

## What Works

**CSV and Excel uploads work reliably.** Download your statement from your bank's NetBanking portal (most Indian banks offer CSV/Excel under Account → Download Statement). The parser handles multiple column naming conventions and Indian number formats.

Once loaded, all features work regardless of source:
- Honda tagging and Arigato scoring
- Ethical ontology scoring across 6 traditions
- Virtue Dashboard (Generosity / Wisdom / Temperance / Prudence / Vitality)
- Happiness ROI correlation (requires journal input)
- AI narrative generation (requires Ollama)
- Feedback loop and SQLite storage

**The merchant KB** handles common Indian narration patterns: UPI payments, NEFT/IMPS transfers, ACH dividends, Zerodha/Groww share purchases, salary credits, SIPs, utility bills.

---

## PDF Parsing: Current Limitations

PDF parsing works for some statement formats and not others. Be aware of the following before relying on it.

**What works:** Statements where each transaction spans 2-3 lines (merchant name, UPI narration, then date+amount+balance on a separate line). The preprocessor reassembles these and resolves debit/credit by tracking the running balance.

**Known failure modes:**

| Issue | Symptom | Fix |
|-------|---------|-----|
| Image-based PDF | Zero transactions extracted | Download as CSV/Excel instead |
| Transaction split across page boundary | Missing transactions at page edges | Known limitation, not yet fixed |
| Non-standard date format | Some transactions missed | Open an issue with debug output |
| LLM timeout on slow hardware | Descriptions less clean | App falls back to rule-based cleaning automatically |

**If extraction looks wrong:** run the included debug script:
```bash
python3 debug_pdf.py your_statement.pdf
```
It reports what pdfplumber extracts, whether transactions are being found, and whether Ollama is responding. The output is usually enough to identify the problem.

**Recommended:** use CSV. It is faster, more reliable, and supported by every major Indian bank.

---

## Optional: AI Narrative Features

Install Ollama and pull a model for narrative summaries and description enhancement:

```bash
ollama pull qwen2.5:7b    # ~4.7 GB
# or for faster responses on slow hardware:
ollama pull tinyllama      # ~637 MB
```

Enable the toggle in the sidebar. On CPU-only machines, inference is slow (1-3 minutes per page). All extraction and scoring works without Ollama.

If using tinyllama, change `OLLAMA_MODEL = "qwen2.5:7b"` to `OLLAMA_MODEL = "tinyllama"` in all `.py` files.

---

## Project Structure

```
happy-money-coach/
├── app.py                      # Streamlit dashboard (6 tabs)
├── transaction_parser.py       # CSV / Excel / PDF ingestion
├── categorizer.py              # Merchant KB + category classification
├── happy_money_tagger.py       # Ken Honda emotional tagging
├── journal_parser.py           # 3-mode journal ingestion
├── happiness_scorer.py         # Happiness ROI correlation
├── perspective_engine.py       # Ethical scoring engine
├── ethical_insights_generator.py  # Narrative synthesis
├── insights_generator.py       # Honda narrative synthesis
├── visualizations.py           # Plotly charts
├── feedback_store.py           # SQLite feedback loop
├── debug_pdf.py                # PDF extraction diagnostics
├── data/
│   ├── merchant_kb.json        # 106-entry merchant knowledge base
│   ├── ethical_ontology.json   # Ethical scoring knowledge graph
│   └── examples/               # Demo CSV files
├── tests/
│   └── test_core.py            # 25 tests
└── requirements.txt
```

---

## The Ethical Ontology

The ethics layer scores each spending category across six traditions using a structured JSON knowledge graph — not LLM generation. Each entry contains a score (-1.0 to 1.0), reasoning grounded in that tradition, specific flags, and tradition-specific concepts and textual references.

Examples of what the ontology encodes:
- Learning scores 0.85–0.95 across all six traditions (Talmud Torah, iqra, samma ditthi, sophia)
- Gambling scores -0.7 to -1.0 across all traditions, for different reasons (maysir in Islam, tanha in Buddhism, epithumia in Stoicism)
- Charity scores 0.95–0.97 universally (tzedakah, zakat, dana, dikaiosyne)

The ontology is in `data/ethical_ontology.json` and carries no restrictions on reuse.

---

## The Feedback Dataset

The feedback tab collects human ratings on AI-assigned Honda tags. Stored in SQLite:
```
transaction description + category + predicted tag + human correction
```
This is training data for a future fine-tuned Happy Money classifier. Export via `feedback_store.export_training_data()`.

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Frontend | Streamlit |
| Visualisation | Plotly |
| PDF extraction | pdfplumber |
| LLM (optional) | Qwen2.5:7b via Ollama |
| Storage | SQLite |
| Tests | pytest (25 passing) |

Runs entirely offline. No cloud, no API keys.

---

## Contributing

PRs welcome, particularly for:
- Additional merchant KB entries for non-Indian banks
- Bank-specific PDF format fixes (run debug_pdf.py, open issue with output)
- PERMA psychology layer (Version 2 roadmap)
- Multi-month trend view

---

*See the [Medium article](https://medium.com/@joyboseroy) for a fuller discussion of the design philosophy, the ethical ontology, and the research directions.*
