"""
visualizations.py
All Plotly charts for the Happy Money Coach dashboard.
"""

import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

# Color palette
HAPPY_COLOR = "#4CAF50"
NEUTRAL_COLOR = "#FFC107"
UNHAPPY_COLOR = "#F44336"
MEANING_COLOR = "#9C27B0"
JOY_COLOR = "#2196F3"
ACCENT = "#FF9800"

HONDA_TAG_COLORS = {
    "Gratitude":  "#4CAF50",
    "Joy":        "#2196F3",
    "Meaning":    "#9C27B0",
    "Love":       "#E91E63",
    "Neutral":    "#9E9E9E",
    "Stress":     "#FF9800",
    "Regret":     "#F44336",
}


# ---------------------------------------------------------------------------
# 1. Money Energy Flow (Ken Honda)
# ---------------------------------------------------------------------------

def energy_flow_chart(energy_flow: dict) -> go.Figure:
    """Sankey-style energy flow or a clean bar chart of money energy."""
    labels = {
        "received_with_gratitude": "Received with Gratitude",
        "spent_with_joy":          "Spent with Joy",
        "creating_meaning":        "Creating Meaning",
        "neutral_spend":           "Neutral Spend",
        "spent_with_stress":       "Spent with Stress",
    }
    colors = {
        "received_with_gratitude": HAPPY_COLOR,
        "spent_with_joy":          JOY_COLOR,
        "creating_meaning":        MEANING_COLOR,
        "neutral_spend":           NEUTRAL_COLOR,
        "spent_with_stress":       UNHAPPY_COLOR,
    }

    keys = [k for k in labels if energy_flow.get(k, 0) > 0]
    values = [energy_flow[k] for k in keys]
    bar_colors = [colors[k] for k in keys]
    bar_labels = [labels[k] for k in keys]

    fig = go.Figure(go.Bar(
        x=bar_labels,
        y=values,
        marker_color=bar_colors,
        text=[f"₹{v:,.0f}" for v in values],
        textposition="outside",
    ))

    fig.update_layout(
        title="Money Energy Flow",
        yaxis_title="Amount (₹)",
        xaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        showlegend=False,
        margin=dict(t=50, b=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 2. Happiness ROI Bar Chart
# ---------------------------------------------------------------------------

def happiness_roi_chart(roi_data: dict) -> go.Figure:
    """Horizontal bar chart of Happiness ROI per category."""
    if not roi_data:
        return go.Figure()

    cats = list(roi_data.keys())
    scores = [roi_data[c]["happiness_roi"] for c in cats]
    spends = [roi_data[c]["spend"] for c in cats]

    # Sort by ROI descending
    sorted_pairs = sorted(zip(cats, scores, spends), key=lambda x: x[1], reverse=True)
    cats, scores, spends = zip(*sorted_pairs) if sorted_pairs else ([], [], [])

    bar_colors = [HAPPY_COLOR if s >= 55 else (NEUTRAL_COLOR if s >= 35 else UNHAPPY_COLOR)
                  for s in scores]

    fig = go.Figure(go.Bar(
        x=list(scores),
        y=list(cats),
        orientation="h",
        marker_color=bar_colors,
        text=[f"{s} pts | ₹{sp:,.0f}" for s, sp in zip(scores, spends)],
        textposition="outside",
    ))

    fig.update_layout(
        title="Happiness ROI by Category",
        xaxis_title="Happiness Score (0-100)",
        xaxis=dict(range=[0, 110]),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=50, b=20, l=110),
    )
    return fig


# ---------------------------------------------------------------------------
# 3. Honda Tag Distribution (pie / donut)
# ---------------------------------------------------------------------------

def honda_tag_pie(transactions) -> go.Figure:
    """Donut chart of Honda tag distribution by spend amount."""
    from collections import defaultdict
    tag_spend: dict[str, float] = defaultdict(float)
    for tx in transactions:
        if tx.tx_type == "Debit" and tx.honda_tag:
            tag_spend[tx.honda_tag] += tx.amount

    if not tag_spend:
        return go.Figure()

    tags = list(tag_spend.keys())
    values = [tag_spend[t] for t in tags]
    colors = [HONDA_TAG_COLORS.get(t, "#9E9E9E") for t in tags]

    fig = go.Figure(go.Pie(
        labels=tags,
        values=values,
        hole=0.5,
        marker_colors=colors,
        textinfo="label+percent",
        hovertemplate="%{label}<br>₹%{value:,.0f}<extra></extra>",
    ))

    fig.update_layout(
        title="Money Emotional Signature (Ken Honda)",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        margin=dict(t=50, b=10),
    )
    return fig


# ---------------------------------------------------------------------------
# 4. Arigato Score Transaction Table
# ---------------------------------------------------------------------------

def arigato_score_table(transactions) -> go.Figure:
    """Color-coded table of transactions with Arigato scores."""
    debits = [tx for tx in transactions if tx.tx_type == "Debit"]
    debits_sorted = sorted(debits, key=lambda t: t.arigato_score, reverse=True)

    rows = []
    for tx in debits_sorted[:30]:  # Top 30
        score = tx.arigato_score
        if score >= 65:
            color = "#E8F5E9"  # light green
        elif score >= 40:
            color = "#FFF8E1"  # light amber
        else:
            color = "#FFEBEE"  # light red

        rows.append({
            "Date": tx.date,
            "Merchant": tx.merchant_clean or tx.description[:30],
            "Category": tx.category,
            "Amount": f"₹{tx.amount:,.0f}",
            "Tag": tx.honda_tag,
            "Arigato": f"{score}/100",
            "_color": color,
        })

    if not rows:
        return go.Figure()

    df = pd.DataFrame(rows)
    fill_colors = [[r["_color"] for r in rows]] * (len(df.columns) - 1)
    cols_display = ["Date", "Merchant", "Category", "Amount", "Tag", "Arigato"]

    fig = go.Figure(go.Table(
        header=dict(
            values=cols_display,
            fill_color="#37474F",
            font=dict(color="white", size=13),
            align="left",
        ),
        cells=dict(
            values=[df[c].tolist() for c in cols_display],
            fill_color=fill_colors,
            font=dict(size=12),
            align="left",
            height=28,
        )
    ))

    fig.update_layout(
        title="Transaction Arigato Scores",
        margin=dict(t=50, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------------
# 5. Well-being gauge
# ---------------------------------------------------------------------------

def wellbeing_gauge(score: int, label: str) -> go.Figure:
    color = HAPPY_COLOR if score >= 65 else (NEUTRAL_COLOR if score >= 45 else UNHAPPY_COLOR)

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"Monthly Well-being Score<br><sub>{label}</sub>", "font": {"size": 16}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40],  "color": "#FFCDD2"},
                {"range": [40, 65], "color": "#FFF9C4"},
                {"range": [65, 100], "color": "#C8E6C9"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.75,
                "value": score,
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=13),
        height=250,
        margin=dict(t=60, b=10, l=20, r=20),
    )
    return fig


# ---------------------------------------------------------------------------
# 6. Spending category treemap
# ---------------------------------------------------------------------------

def spending_treemap(category_summary: dict) -> go.Figure:
    """Treemap of total spend by category."""
    cats = [c for c in category_summary if c != "Income"]
    values = [category_summary[c]["total_spend"] for c in cats]

    if not cats:
        return go.Figure()

    fig = px.treemap(
        names=cats,
        parents=["" for _ in cats],
        values=values,
        title="Spending Distribution",
        color=values,
        color_continuous_scale=["#FFCDD2", "#FFF9C4", "#C8E6C9"],
    )
    fig.update_traces(texttemplate="%{label}<br>₹%{value:,.0f}", textfont_size=13)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(t=50, b=10),
    )
    return fig
