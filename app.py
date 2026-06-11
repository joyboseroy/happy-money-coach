"""
app.py
Happy Money Coach — Streamlit dashboard.
Merges Ken Honda's Happy Money philosophy with data-driven Happiness ROI.

Run: streamlit run app.py
"""

import io
import streamlit as st
import pandas as pd
from pathlib import Path

# ---- Page config ----
st.set_page_config(
    page_title="Happy Money Coach",
    page_icon="🙏",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Imports (lazy so app loads even without Ollama) ----
from transaction_parser import parse_statement, Transaction
from categorizer import categorize_batch, get_category_summary
from happy_money_tagger import tag_batch, get_energy_flow, get_arigato_leaderboard, get_regret_list
from journal_parser import from_csv as journal_from_csv, from_reflection, from_user_selections
from happiness_scorer import compute_happiness_roi, compute_wellbeing_score, spending_reallocation_suggestion
from insights_generator import generate_insights
from perspective_engine import (
    score_batch, get_lens_summary, get_virtue_dashboard,
    get_ethical_leaderboard, get_concern_transactions,
    LENSES, LENS_DESCRIPTIONS
)
from ethical_insights_generator import generate_monthly_ethical_summary, generate_cross_tradition_comparison
from visualizations import (
    energy_flow_chart, happiness_roi_chart, honda_tag_pie,
    arigato_score_table, wellbeing_gauge, spending_treemap
)
from feedback_store import init_db, record_feedback, record_session, get_feedback_stats

init_db()

# ---- Custom CSS ----
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a237e 0%, #4CAF50 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .honda-quote {
        background: #f3e5f5;
        border-left: 4px solid #9C27B0;
        padding: 0.8rem 1.2rem;
        border-radius: 0 8px 8px 0;
        font-style: italic;
        margin: 0.5rem 0 1.5rem 0;
        color: #4a148c;
    }
    .metric-card {
        background: white;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
    }
    .arigato-high { color: #2E7D32; font-weight: bold; }
    .arigato-mid  { color: #F57F17; font-weight: bold; }
    .arigato-low  { color: #C62828; font-weight: bold; }
    .insight-card {
        background: #E8F5E9;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #4CAF50;
    }
    .regret-card {
        background: #FFEBEE;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.5rem 0;
        border-left: 4px solid #F44336;
    }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="main-header">
    <h1>🙏 Happy Money Coach</h1>
    <p style="margin:0; opacity:0.9;">Inspired by Ken Honda's Happy Money philosophy — because money is energy, and every transaction tells a story.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="honda-quote">
"The key to becoming rich in the truest sense is not working harder, but working on your relationship with money."
<br><small>— Ken Honda</small>
</div>
""", unsafe_allow_html=True)


# ====================================================================
# SIDEBAR
# ====================================================================
with st.sidebar:
    st.header("📂 Upload Your Data")

    # ---- Transaction statement ----
    st.subheader("1. Bank / Credit Card Statement")
    statement_file = st.file_uploader(
        "Upload CSV, Excel, or PDF",
        type=["csv", "xlsx", "xls", "pdf"],
        key="statement"
    )

    use_demo = st.checkbox("Use demo data instead", value=True)

    st.divider()

    # ---- Journal input ----
    st.subheader("2. Happiness Journal (optional)")
    journal_mode = st.radio(
        "Input mode",
        ["None (I'll select later)", "Monthly Reflection", "Upload Journal CSV"],
        index=0
    )

    journal_text = ""
    journal_file = None

    if journal_mode == "Monthly Reflection":
        journal_text = st.text_area(
            "What were your 3 happiest moments this month?",
            placeholder="e.g. Weekend trip to Coorg, finished a course, family dinner...",
            height=120
        )
    elif journal_mode == "Upload Journal CSV":
        journal_file = st.file_uploader(
            "Journal CSV (Date, Entry columns)",
            type=["csv"],
            key="journal"
        )

    st.divider()

    # ---- Ollama settings ----
    st.subheader("⚙️ Settings")
    use_llm = st.checkbox("Use Qwen3 via Ollama", value=False,
                          help="Requires Ollama running locally with qwen2.5:7b model")
    if not use_llm:
        st.info("Running in offline mode — using rule-based analysis. Enable Qwen3 for deeper insights.")

    st.divider()
    stats = get_feedback_stats()
    st.caption(f"Sessions: {stats['sessions']} | Feedback: {stats['total_feedback']} | AI accuracy: {stats['ai_accuracy_pct']}%")


# ====================================================================
# LOAD DATA
# ====================================================================

@st.cache_data
def load_demo_transactions():
    demo_path = Path(__file__).resolve().parent / "data" / "examples" / "sample_transactions.csv"
    with open(demo_path) as f:
        txs = parse_statement(f, "sample_transactions.csv")
    return txs


@st.cache_data
def load_demo_journal():
    demo_path = Path(__file__).resolve().parent / "data" / "examples" / "sample_journal.csv"
    with open(demo_path) as f:
        return journal_from_csv(f)


def process_transactions(txs: list[Transaction], run_llm: bool) -> list[Transaction]:
    """Categorize and Honda-tag all transactions."""
    if run_llm:
        txs = categorize_batch(txs)
        txs = tag_batch(txs)
    else:
        # Offline: use merchant KB + category fallbacks
        import json, re
        from pathlib import Path as P
        from happy_money_tagger import _category_fallback
        from categorizer import _load_kb, _normalize, _kb_lookup

        kb = _load_kb()
        for tx in txs:
            if tx.tx_type == "Credit":
                tx.category = "Income"
                tx.merchant_clean = tx.description[:40]
                tx.honda_tag = "Gratitude"
                tx.arigato_score = 85
                tx.happy_money = True
                tx.honda_reason = "Money received — welcome it with Arigato."
                continue
            kb_result = _kb_lookup(tx.description, kb)
            if kb_result:
                tx.category = kb_result["category"]
                tx.merchant_clean = tx.description[:40]
            else:
                tx.category = "Shopping"
                tx.merchant_clean = tx.description[:40]
            result = _category_fallback(tx)
            tx.honda_tag = result["honda_tag"]
            tx.arigato_score = result["arigato_score"]
            tx.happy_money = result["happy_money"]
            tx.honda_reason = result["honda_reason"]
    return txs


# ---- Load ----
if statement_file is not None:
    raw_txs = parse_statement(statement_file, statement_file.name, use_llm=use_llm)
elif use_demo:
    raw_txs = load_demo_transactions()
else:
    st.info("Upload a bank statement or enable demo data to begin.")
    st.stop()

with st.spinner("Analysing your money energy..."):
    transactions = process_transactions(raw_txs, use_llm)

# ---- Journal events ----
journal_events = []
if journal_mode == "Monthly Reflection" and journal_text.strip():
    journal_events = from_reflection(journal_text)
elif journal_mode == "Upload Journal CSV" and journal_file:
    journal_events = journal_from_csv(journal_file)
elif use_demo:
    journal_events = load_demo_journal()


# ====================================================================
# COMPUTE SUMMARIES
# ====================================================================
category_summary = get_category_summary(transactions)
energy_flow = get_energy_flow(transactions)
roi_data = compute_happiness_roi(category_summary, journal_events)
wellbeing = compute_wellbeing_score(transactions)
top_happy = get_arigato_leaderboard(transactions, top_n=5)
top_regret = get_regret_list(transactions, top_n=5)

best_cat = max(roi_data, key=lambda c: roi_data[c]["happiness_roi"]) if roi_data else ""
reallocation = spending_reallocation_suggestion(roi_data, 0, best_cat)

# ---- Ethical scoring (rules-based, no LLM needed) ----
ethical_profiles = score_batch(transactions)
virtue_dashboard_data = get_virtue_dashboard(ethical_profiles, category_summary)

# ---- Save session ----
record_session(len(transactions), wellbeing["score"])


# ====================================================================
# TABS
# ====================================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🌊 Energy Flow",
    "📊 Happiness ROI",
    "🧾 Transactions",
    "💡 AI Insights",
    "⚖️ Ethics Lens",
    "📈 Overview",
])


# ====================================================================
# TAB 1: Ken Honda Energy Flow
# ====================================================================
with tab1:
    st.subheader("Money Energy Flow — Ken Honda Style")
    st.caption("Ken Honda teaches that money is energy. This view shows how your money's emotional energy moved this month.")

    col1, col2, col3 = st.columns(3)
    total_income = sum(tx.amount for tx in transactions if tx.tx_type == "Credit")
    happy_spend = energy_flow.get("spent_with_joy", 0) + energy_flow.get("creating_meaning", 0)
    stress_spend = energy_flow.get("spent_with_stress", 0)

    with col1:
        st.metric("Received with Gratitude", f"₹{total_income:,.0f}")
    with col2:
        st.metric("Spent with Joy + Meaning", f"₹{happy_spend:,.0f}",
                  delta=f"{(happy_spend/total_income*100):.0f}% of income" if total_income else "")
    with col3:
        st.metric("Spent with Stress", f"₹{stress_spend:,.0f}",
                  delta=f"-₹{stress_spend:,.0f}" if stress_spend > 0 else "None",
                  delta_color="inverse")

    st.plotly_chart(energy_flow_chart(energy_flow), use_container_width=True, key="energy_flow")
    st.plotly_chart(honda_tag_pie(transactions), use_container_width=True, key="honda_pie")

    st.markdown("### 🙏 Arigato Moments — Highest Scoring Transactions")
    for tx in top_happy:
        score = tx.arigato_score
        emoji = "🟢" if score >= 75 else "🟡"
        st.markdown(f"""
<div class="insight-card">
    {emoji} <b>{tx.merchant_clean}</b> — ₹{tx.amount:,.0f}
    <br><small>{tx.honda_tag} &nbsp;|&nbsp; Arigato Score: <b>{score}/100</b></small>
    <br><small style="color:#555">{tx.honda_reason}</small>
</div>
""", unsafe_allow_html=True)

    if top_regret:
        st.markdown("### 🔴 Regret Patterns — Lowest Scoring Transactions")
        for tx in top_regret:
            st.markdown(f"""
<div class="regret-card">
    🔴 <b>{tx.merchant_clean}</b> — ₹{tx.amount:,.0f}
    <br><small>{tx.honda_tag} &nbsp;|&nbsp; Arigato Score: <b>{tx.arigato_score}/100</b></small>
    <br><small style="color:#555">{tx.honda_reason}</small>
</div>
""", unsafe_allow_html=True)


# ====================================================================
# TAB 2: Happiness ROI
# ====================================================================
with tab2:
    st.subheader("Happiness ROI by Spending Category")
    st.caption("Correlates where your money went with what appeared in your journal as positive memories.")

    if not journal_events:
        st.info("Add a journal (sidebar) to see correlation-based ROI. Currently showing Honda tag scores only.")

    st.plotly_chart(happiness_roi_chart(roi_data), use_container_width=True, key="happiness_roi")

    st.markdown("### Category Breakdown")
    for cat, data in sorted(roi_data.items(), key=lambda x: x[1]["happiness_roi"], reverse=True):
        score = data["happiness_roi"]
        spend = data["spend"]
        bar_pct = score
        color = "#4CAF50" if score >= 55 else ("#FFC107" if score >= 35 else "#F44336")

        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"**{cat}** — ₹{spend:,.0f}")
            st.progress(bar_pct / 100, text=f"{score}/100 Happiness Points")
        with col2:
            st.markdown(f"<br><span style='color:{color};font-size:1.1em'>{score}</span>", unsafe_allow_html=True)

        st.caption(data["interpretation"])
        st.divider()

    if reallocation:
        st.info(f"💡 {reallocation}")

    st.plotly_chart(spending_treemap(category_summary), use_container_width=True, key="treemap_roi")


# ====================================================================
# TAB 3: Transaction Detail + Feedback
# ====================================================================
with tab3:
    st.subheader("Transaction Arigato Scores")
    st.caption("Each transaction tagged with Ken Honda's emotional framework. Click 👍/👎 to correct the AI.")

    # Filter controls
    col1, col2 = st.columns(2)
    with col1:
        filter_type = st.selectbox("Show", ["All", "Happy Money Only", "Unhappy Money Only"], key="filter_type")
    with col2:
        filter_cat = st.selectbox("Category", ["All"] + sorted(set(tx.category for tx in transactions)), key="filter_cat")

    filtered = transactions
    if filter_type == "Happy Money Only":
        filtered = [tx for tx in filtered if tx.happy_money and tx.tx_type == "Debit"]
    elif filter_type == "Unhappy Money Only":
        filtered = [tx for tx in filtered if not tx.happy_money and tx.tx_type == "Debit"]
    if filter_cat != "All":
        filtered = [tx for tx in filtered if tx.category == filter_cat]

    st.plotly_chart(arigato_score_table(filtered), use_container_width=True, key="arigato_table")

    # Feedback section
    st.markdown("### 🎯 Correct the AI")
    st.caption("Help train the Happy Money model. Select a transaction and rate the AI's tag.")

    tx_options = {
        f"{tx.date} | {(tx.merchant_clean or tx.description or 'Unknown')[:25]} | ₹{tx.amount:,.0f} | {tx.honda_tag or 'Neutral'}": tx
        for tx in transactions if tx.tx_type == "Debit"
    }

    if not tx_options:
        st.info("No debit transactions found to review.")
    else:
        selected_label = st.selectbox("Select transaction", list(tx_options.keys()), key="select_tx")
        selected_tx = tx_options[selected_label]

        col1, col2, col3 = st.columns(3)
        with col1:
            vote = st.radio("Was the AI tag correct?", ["👍 Yes", "👎 No"], horizontal=True, key="vote_radio")
        with col2:
            corrected_tag = st.selectbox(
                "Correct tag (if wrong)",
                ["(no change)", "Gratitude", "Joy", "Meaning", "Love", "Neutral", "Stress", "Regret"],
                key="correct_tag"
            )
        with col3:
            corrected_score = st.slider("Correct Arigato score", 0, 100, selected_tx.arigato_score, key="correct_score")

        if st.button("Submit Feedback", key="submit_feedback"):
            record_feedback(
                tx_description=selected_tx.description,
                tx_amount=selected_tx.amount,
                tx_date=selected_tx.date,
                category=selected_tx.category,
                honda_tag=selected_tx.honda_tag,
                arigato_score=selected_tx.arigato_score,
                happy_money=selected_tx.happy_money,
                user_vote="up" if "Yes" in vote else "down",
                user_corrected_tag=corrected_tag if corrected_tag != "(no change)" else None,
                user_corrected_score=corrected_score if corrected_score != selected_tx.arigato_score else None
            )
            st.success("Feedback saved. Arigato for helping train the model.")


# ====================================================================
# TAB 4: AI Insights
# ====================================================================
with tab4:
    st.subheader("💡 Ken Honda AI Insights")
    st.caption("Synthesised narrative from your spending patterns and journal entries.")

    if st.button("Generate Insights", key="gen_insights" + (" (using Qwen3)" if use_llm else " (template mode)")):
        with st.spinner("Reflecting on your money energy..."):
            insights = generate_insights(
                energy_flow, roi_data, wellbeing,
                top_happy, top_regret, reallocation
            )
        st.session_state["insights"] = insights

    if "insights" in st.session_state:
        for i, insight in enumerate(st.session_state["insights"], 1):
            st.markdown(f"""
<div class="insight-card">
    <b>Insight {i}</b><br>{insight}
</div>
""", unsafe_allow_html=True)
    else:
        st.info("Click 'Generate Insights' to see your personalised Happy Money narrative.")

    # Journal patterns
    if journal_events:
        st.markdown("### 📖 Journal Pattern Summary")
        from journal_parser import keyword_frequency
        pos_keywords = keyword_frequency(journal_events, "positive")
        neg_keywords = keyword_frequency(journal_events, "negative")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Positive keywords in your journal:**")
            for kw, count in list(pos_keywords.items())[:8]:
                st.markdown(f"- {kw} ({count}x)")
        with col2:
            st.markdown("**Negative keywords:**")
            for kw, count in list(neg_keywords.items())[:8]:
                st.markdown(f"- {kw} ({count}x)")

    # Arigato practice
    st.markdown("---")
    st.markdown("""
### 🙏 Daily Arigato Practice

Ken Honda's core teaching is simple: **greet money with gratitude**.

- When salary arrives: *"Arigato for arriving. I will use you with care."*
- When paying bills: *"Arigato for providing light, water, and connection."*
- Before shopping: *"Will I remember this purchase with warmth?"*
- After an impulse buy: *"What need was I really trying to meet?"*

The goal is not to spend less. It is to spend **awake**.
""")


# ====================================================================
# TAB 5: Overview
# ====================================================================
with tab5:
    st.subheader("Monthly Overview")

    col1, col2, col3, col4 = st.columns(4)
    total_debit = sum(tx.amount for tx in transactions if tx.tx_type == "Debit")
    total_credit = sum(tx.amount for tx in transactions if tx.tx_type == "Credit")
    happy_count = sum(1 for tx in transactions if tx.happy_money and tx.tx_type == "Debit")
    tx_count = sum(1 for tx in transactions if tx.tx_type == "Debit")

    with col1:
        st.metric("Total Income", f"₹{total_credit:,.0f}")
    with col2:
        st.metric("Total Spend", f"₹{total_debit:,.0f}")
    with col3:
        pct = int(happy_count/tx_count*100) if tx_count else 0
        st.metric("Happy Money Transactions", f"{happy_count}/{tx_count}", delta=f"{pct}%")
    with col4:
        st.metric("Net", f"₹{total_credit - total_debit:,.0f}",
                  delta="positive" if total_credit > total_debit else "negative",
                  delta_color="normal" if total_credit > total_debit else "inverse")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(
            wellbeing_gauge(wellbeing["score"], wellbeing["label"]),
            use_container_width=True, key="gauge_tab5"
        )
        st.caption(wellbeing["insight"])
    with col2:
        st.plotly_chart(spending_treemap(category_summary), use_container_width=True, key="treemap_overview")

    # Level 1 journal: user selects happy purchases
    if journal_mode == "None (I'll select later)":
        st.markdown("### Which purchases made you happiest this month?")
        st.caption("Select your top happy purchases. This trains your personal Happy Money model.")
        debit_txs = [tx for tx in transactions if tx.tx_type == "Debit"]
        options = [f"{tx.date} | {tx.merchant_clean} | ₹{tx.amount:,.0f}" for tx in debit_txs]
        selected = st.multiselect("Select your happiest purchases:", options, key="happy_picks_tab5")
        if selected and st.button("Save Selections", key="save_sel_tab5"):
            events = from_user_selections(selected)
            st.success(f"Saved {len(events)} happy moments. These will improve your ROI scores.")
            st.session_state["user_events"] = events


# ====================================================================
# TAB 5 (new): Ethics Lens
# ====================================================================
with tab5:
    st.subheader("⚖️ Ethical Lens Analysis")
    st.caption(
        "Your spending viewed through six ethical traditions: Universal Human Values, "
        "Buddhist, Christian, Jewish, Islamic, and Stoic. "
        "Scores are computed from a structured ethical ontology — not generated by AI."
    )

    # --- Lens selector ---
    col1, col2 = st.columns([2, 1])
    with col1:
        selected_lens = st.selectbox(
            "Select ethical lens",
            options=LENSES,
            format_func=lambda x: LENS_DESCRIPTIONS.get(x, x.title()),
            index=0
        )
    with col2:
        show_all_lenses = st.checkbox("Compare all lenses", value=False)

    # --- Virtue Dashboard ---
    st.markdown("### 🌿 Virtue Dashboard")
    st.caption("Your spending mapped to virtues, not just categories.")

    vd = virtue_dashboard_data.get("by_pct", {})
    if vd:
        virtue_items = sorted(vd.items(), key=lambda x: x[1], reverse=True)
        v_labels = [v for v, _ in virtue_items]
        v_values = [p for _, p in virtue_items]

        VIRTUE_COLORS = {
            "Generosity":        "#4CAF50",
            "Wisdom":            "#2196F3",
            "Temperance":        "#FF9800",
            "Prudence":          "#9C27B0",
            "Vitality":          "#00BCD4",
            "Right Livelihood":  "#8BC34A",
            "Consumption":       "#F44336",
        }
        bar_colors = [VIRTUE_COLORS.get(v, "#9E9E9E") for v in v_labels]

        import plotly.graph_objects as go
        fig_virtue = go.Figure(go.Bar(
            x=v_values,
            y=v_labels,
            orientation="h",
            marker_color=bar_colors,
            text=[f"{p}%" for p in v_values],
            textposition="outside",
        ))
        fig_virtue.update_layout(
            title="Spending by Virtue",
            xaxis_title="% of total spend",
            xaxis=dict(range=[0, max(v_values) * 1.2 if v_values else 50]),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(size=13),
            margin=dict(t=50, b=20, l=130),
            height=350,
        )
        st.plotly_chart(fig_virtue, use_container_width=True, key="virtue_dashboard")

    st.divider()

    # --- Per-lens summary ---
    if show_all_lenses:
        st.markdown("### Cross-Tradition Score Comparison")
        cols = st.columns(len(LENSES))
        for i, lens in enumerate(LENSES):
            summary = get_lens_summary(ethical_profiles, lens)
            score = summary.get("score_avg", 0)
            label = summary.get("score_label", "Neutral")
            color = "#4CAF50" if score >= 0.40 else ("#FF9800" if score >= 0.0 else "#F44336")
            with cols[i]:
                st.markdown(f"""
<div style="text-align:center; padding:0.5rem; background:#f9f9f9; border-radius:8px; border-top: 4px solid {color}">
    <b style="font-size:0.85em">{LENS_DESCRIPTIONS.get(lens, lens)}</b><br>
    <span style="font-size:1.5em; color:{color}">{score:.2f}</span><br>
    <small>{label}</small>
</div>
""", unsafe_allow_html=True)
    else:
        lens_summary = get_lens_summary(ethical_profiles, selected_lens)
        score_avg = lens_summary.get("score_avg", 0)
        score_label = lens_summary.get("score_label", "Neutral")

        col1, col2, col3 = st.columns(3)
        color = "#4CAF50" if score_avg >= 0.40 else ("#FF9800" if score_avg >= 0.0 else "#F44336")
        with col1:
            st.metric(f"{LENS_DESCRIPTIONS[selected_lens]} Score",
                      f"{score_avg:.2f}", delta=score_label)
        with col2:
            st.metric("Best Category", lens_summary.get("best_category", "—"))
        with col3:
            st.metric("Concerns", lens_summary.get("concern_count", 0),
                      delta="transactions", delta_color="off")

    st.divider()

    # --- Category-by-lens breakdown ---
    st.markdown(f"### Spending Categories — {LENS_DESCRIPTIONS.get(selected_lens, selected_lens)} Perspective")

    for profile in sorted(ethical_profiles, key=lambda p: p.scores.get(selected_lens, type('', (), {'score': 0})()).score, reverse=True):
        if selected_lens not in profile.scores:
            continue
        score_data = profile.scores[selected_lens]
        score = score_data.score
        label = score_data.score_label
        amount = profile.tx_amount

        if score >= 0.60:
            bar_color = "#E8F5E9"
            icon = "🟢"
        elif score >= 0.10:
            bar_color = "#FFF8E1"
            icon = "🟡"
        else:
            bar_color = "#FFEBEE"
            icon = "🔴"

        with st.expander(f"{icon} {profile.tx_description[:45]} — ₹{amount:,.0f} | {label} ({score:.2f})"):
            col1, col2 = st.columns([2, 1])
            with col1:
                st.markdown(f"**Category:** {profile.category}  |  **Human Need:** {profile.human_need}")
                st.markdown(f"**Virtue:** {profile.virtue}  |  **Value:** {profile.value}")
                st.markdown(f"**{LENS_DESCRIPTIONS[selected_lens]} view:** {score_data.reason}")
                if score_data.flags:
                    st.caption(f"Flags: {' · '.join(score_data.flags[:3])}")
                if score_data.concepts:
                    st.caption(f"Concepts: {', '.join(score_data.concepts[:4])}")
            with col2:
                if profile.virtue_alignment:
                    st.markdown("**Virtues aligned:**")
                    for v in profile.virtue_alignment:
                        st.markdown(f"  · {v}")

    st.divider()

    # --- Monthly narrative ---
    st.markdown(f"### {LENS_DESCRIPTIONS.get(selected_lens, selected_lens)} Monthly Narrative")
    if st.button(f"Generate {LENS_DESCRIPTIONS[selected_lens]} Analysis"):
        with st.spinner(f"Reflecting through {LENS_DESCRIPTIONS[selected_lens]} lens..."):
            lens_sum = get_lens_summary(ethical_profiles, selected_lens)
            top_pos = get_ethical_leaderboard(ethical_profiles, selected_lens, top_n=3)
            top_con = get_concern_transactions(ethical_profiles, selected_lens)[:3]
            narrative = generate_monthly_ethical_summary(
                selected_lens, lens_sum, virtue_dashboard_data,
                top_pos, top_con, use_llm=use_llm
            )
        st.session_state[f"narrative_{selected_lens}"] = narrative

    if f"narrative_{selected_lens}" in st.session_state:
        st.markdown(f"""
<div style="background:#f3e5f5; border-left:4px solid #9C27B0; padding:1rem 1.2rem; border-radius:0 8px 8px 0; margin:0.5rem 0">
{st.session_state[f'narrative_{selected_lens}']}
</div>
""", unsafe_allow_html=True)

    # --- Concern highlight ---
    concern_txs = get_concern_transactions(ethical_profiles, selected_lens)
    if concern_txs:
        st.markdown(f"### ⚠️ Transactions Flagged by {LENS_DESCRIPTIONS[selected_lens]}")
        for p in concern_txs[:5]:
            score = p.scores[selected_lens].score
            reason = p.scores[selected_lens].reason
            flags = p.scores[selected_lens].flags
            st.markdown(f"""
<div class="regret-card">
    🔴 <b>{p.tx_description[:50]}</b> — ₹{p.tx_amount:,.0f} | Score: {score:.2f}<br>
    <small>{reason}</small><br>
    <small style="color:#888">{' · '.join(flags[:2]) if flags else ''}</small>
</div>
""", unsafe_allow_html=True)


# ====================================================================
# TAB 6 (was 5): Overview — now uses tab6
# ====================================================================
with tab6:
    st.subheader("Monthly Overview")

    col1, col2, col3, col4 = st.columns(4)
    total_debit = sum(tx.amount for tx in transactions if tx.tx_type == "Debit")
    total_credit = sum(tx.amount for tx in transactions if tx.tx_type == "Credit")
    happy_count = sum(1 for tx in transactions if tx.happy_money and tx.tx_type == "Debit")
    tx_count = sum(1 for tx in transactions if tx.tx_type == "Debit")

    with col1:
        st.metric("Total Income", f"₹{total_credit:,.0f}")
    with col2:
        st.metric("Total Spend", f"₹{total_debit:,.0f}")
    with col3:
        pct = int(happy_count/tx_count*100) if tx_count else 0
        st.metric("Happy Money Transactions", f"{happy_count}/{tx_count}", delta=f"{pct}%")
    with col4:
        st.metric("Net", f"₹{total_credit - total_debit:,.0f}",
                  delta="positive" if total_credit > total_debit else "negative",
                  delta_color="normal" if total_credit > total_debit else "inverse")

    col1, col2 = st.columns([1, 2])
    with col1:
        st.plotly_chart(
            wellbeing_gauge(wellbeing["score"], wellbeing["label"]),
            use_container_width=True, key="gauge_overview"
        )
        st.caption(wellbeing["insight"])
    with col2:
        st.plotly_chart(spending_treemap(category_summary), use_container_width=True, key="treemap_overview2")

    # Universal ethics quick summary
    univ_summary = get_lens_summary(ethical_profiles, "universal")
    if univ_summary:
        univ_score = univ_summary.get("score_avg", 0)
        univ_label = univ_summary.get("score_label", "Neutral")
        color = "#4CAF50" if univ_score >= 0.40 else "#FF9800"
        st.markdown(f"""
<div style="background:#f9f9f9; border-radius:10px; padding:1rem; margin-top:1rem; border-left:4px solid {color}">
    <b>Universal Ethics Score: {univ_score:.2f} ({univ_label})</b><br>
    <small>Best category: {univ_summary.get('best_category', '—')} &nbsp;|&nbsp;
    Encouraged transactions: {univ_summary.get('encouraged_count', 0)} &nbsp;|&nbsp;
    Flagged: {univ_summary.get('concern_count', 0)}</small>
</div>
""", unsafe_allow_html=True)

    # Level 1 journal: user selects happy purchases
    if journal_mode == "None (I'll select later)":
        st.markdown("### Which purchases made you happiest this month?")
        st.caption("Select your top happy purchases. This trains your personal Happy Money model.")
        debit_txs = [tx for tx in transactions if tx.tx_type == "Debit"]
        options = [f"{tx.date} | {tx.merchant_clean} | ₹{tx.amount:,.0f}" for tx in debit_txs]
        selected = st.multiselect("Select your happiest purchases:", options, key="happy_picks_tab6")
        if selected and st.button("Save Selections", key="save_sel_tab6"):
            events = from_user_selections(selected)
            st.success(f"Saved {len(events)} happy moments. These will improve your ROI scores.")
            st.session_state["user_events"] = events
