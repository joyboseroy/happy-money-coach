"""
tests/test_core.py
Basic unit tests for Happy Money Coach core modules.
Run: python -m pytest tests/ -v
"""

import sys
import io
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from transaction_parser import parse_csv, Transaction
from categorizer import _kb_lookup, _load_kb, get_category_summary
from happy_money_tagger import _quick_tag, _category_fallback, get_energy_flow, get_arigato_leaderboard
from happiness_scorer import compute_wellbeing_score, compute_happiness_roi
from feedback_store import init_db, record_feedback, get_feedback_stats


# ---- Fixtures ----

def sample_csv_content():
    return """Date,Description,Amount,Type
2026-05-01,IRCTC WEB BOOKING,-3200,Debit
2026-05-02,SWIGGY LTD,-480,Debit
2026-05-04,SALARY CREDIT,120000,Credit
2026-05-05,UDEMY COURSE,-1299,Debit
2026-05-22,RESTAURANT FAMILY DINNER,-3200,Debit
"""


def make_transactions():
    f = io.StringIO(sample_csv_content())
    txs = parse_csv(f)
    kb = _load_kb()
    from categorizer import _kb_lookup
    for tx in txs:
        if tx.tx_type == "Credit":
            tx.category = "Income"
            tx.merchant_clean = tx.description[:40]
            continue
        result = _kb_lookup(tx.description, kb)
        if result:
            tx.category = result["category"]
        else:
            tx.category = "Shopping"
        tx.merchant_clean = tx.description[:40]
        r = _category_fallback(tx)
        tx.honda_tag = r["honda_tag"]
        tx.arigato_score = r["arigato_score"]
        tx.happy_money = r["happy_money"]
        tx.honda_reason = r["honda_reason"]
    return txs


# ---- Tests ----

class TestTransactionParser:
    def test_parse_csv_basic(self):
        f = io.StringIO(sample_csv_content())
        txs = parse_csv(f)
        assert len(txs) == 5

    def test_credit_debit_split(self):
        f = io.StringIO(sample_csv_content())
        txs = parse_csv(f)
        credits = [t for t in txs if t.tx_type == "Credit"]
        debits = [t for t in txs if t.tx_type == "Debit"]
        assert len(credits) == 1
        assert len(debits) == 4

    def test_amount_positive(self):
        f = io.StringIO(sample_csv_content())
        txs = parse_csv(f)
        for tx in txs:
            assert tx.amount >= 0, f"Negative amount: {tx}"

    def test_irctc_parsed(self):
        f = io.StringIO(sample_csv_content())
        txs = parse_csv(f)
        descs = [tx.description for tx in txs]
        assert any("IRCTC" in d for d in descs)


class TestMerchantKB:
    def test_kb_loads(self):
        kb = _load_kb()
        assert "merchants" in kb
        assert len(kb["merchants"]) > 0

    def test_irctc_lookup(self):
        kb = _load_kb()
        tx = Transaction(date="2026-05-01", description="IRCTC WEB BOOKING", amount=3200, tx_type="Debit")
        result = _kb_lookup(tx.description, kb)
        assert result is not None
        assert result["category"] == "Travel"

    def test_unknown_returns_none(self):
        kb = _load_kb()
        tx = Transaction(date="2026-05-01", description="XYZUNKNOWN123", amount=100, tx_type="Debit")
        result = _kb_lookup(tx.description, kb)
        assert result is None


class TestHondaTagger:
    def test_credit_gets_gratitude(self):
        tx = Transaction(date="2026-05-01", description="SALARY CREDIT", amount=120000, tx_type="Credit")
        result = _quick_tag(tx)
        assert result is not None
        assert result["honda_tag"] == "Gratitude"
        assert result["happy_money"] is True

    def test_charity_gets_meaning(self):
        tx = Transaction(date="2026-05-01", description="GIVE INDIA DONATION", amount=2000, tx_type="Debit")
        result = _quick_tag(tx)
        assert result is not None
        assert result["honda_tag"] == "Meaning"
        assert result["arigato_score"] >= 90

    def test_category_fallback_travel(self):
        tx = Transaction(date="2026-05-01", description="IRCTC", amount=3200,
                         tx_type="Debit", category="Travel")
        result = _category_fallback(tx)
        assert result["honda_tag"] == "Joy"
        assert result["happy_money"] is True

    def test_category_fallback_bills(self):
        tx = Transaction(date="2026-05-01", description="ELECTRICITY BILL", amount=1500,
                         tx_type="Debit", category="Bills")
        result = _category_fallback(tx)
        assert result["honda_tag"] == "Stress"
        assert result["happy_money"] is False

    def test_arigato_score_range(self):
        txs = make_transactions()
        for tx in txs:
            assert 0 <= tx.arigato_score <= 100, f"Out-of-range score: {tx.arigato_score}"


class TestEnergyFlow:
    def test_energy_flow_keys(self):
        txs = make_transactions()
        flow = get_energy_flow(txs)
        assert "received_with_gratitude" in flow
        assert "spent_with_joy" in flow
        assert "spent_with_stress" in flow
        assert "creating_meaning" in flow

    def test_energy_flow_non_negative(self):
        txs = make_transactions()
        flow = get_energy_flow(txs)
        for k, v in flow.items():
            assert v >= 0, f"Negative flow: {k}={v}"

    def test_leaderboard_sorted(self):
        txs = make_transactions()
        top = get_arigato_leaderboard(txs, top_n=3)
        scores = [tx.arigato_score for tx in top]
        assert scores == sorted(scores, reverse=True)


class TestHappinessScorer:
    def test_wellbeing_score_range(self):
        txs = make_transactions()
        wb = compute_wellbeing_score(txs)
        assert 0 <= wb["score"] <= 100
        assert wb["label"] in ("Flourishing", "Growing", "Neutral", "Draining")

    def test_roi_categories_present(self):
        txs = make_transactions()
        cat_summary = get_category_summary(txs)
        roi = compute_happiness_roi(cat_summary, [])
        for cat, data in roi.items():
            assert "happiness_roi" in data
            assert 0 <= data["happiness_roi"] <= 100


class TestFeedbackStore:
    def test_init_and_record(self):
        init_db()
        record_feedback(
            tx_description="TEST MERCHANT",
            tx_amount=1000.0,
            tx_date="2026-05-01",
            category="Shopping",
            honda_tag="Neutral",
            arigato_score=45,
            happy_money=False,
            user_vote="down",
            user_corrected_tag="Regret",
            user_corrected_score=20
        )
        stats = get_feedback_stats()
        assert stats["total_feedback"] >= 1

    def test_stats_structure(self):
        stats = get_feedback_stats()
        assert "total_feedback" in stats
        assert "ai_accuracy_pct" in stats


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


class TestPerspectiveEngine:
    def test_score_learning(self):
        from perspective_engine import score_transaction
        from transaction_parser import Transaction
        tx = Transaction(date='2026-05-01', description='UDEMY COURSE', amount=1299, tx_type='Debit', category='Learning')
        profile = score_transaction(tx)
        # All traditions should rate learning positively
        for lens in ['universal', 'buddhist', 'christian', 'jewish', 'islamic', 'stoic']:
            assert lens in profile.scores
            assert profile.scores[lens].score >= 0.80, f'{lens} score for Learning should be >= 0.80, got {profile.scores[lens].score}'

    def test_score_charity(self):
        from perspective_engine import score_transaction
        from transaction_parser import Transaction
        tx = Transaction(date='2026-05-01', description='GIVE INDIA DONATION', amount=2000, tx_type='Debit', category='Charity')
        profile = score_transaction(tx)
        assert profile.composite_score >= 0.85

    def test_score_shopping_caution(self):
        from perspective_engine import score_transaction
        from transaction_parser import Transaction
        tx = Transaction(date='2026-05-01', description='AMAZON IMPULSE BUY', amount=5000, tx_type='Debit', category='Shopping')
        profile = score_transaction(tx)
        # Buddhist and Islamic should flag shopping
        assert profile.scores['buddhist'].score < 0.30
        assert profile.scores['islamic'].score < 0.30

    def test_composite_score_range(self):
        from perspective_engine import score_transaction
        from transaction_parser import Transaction
        txs = [
            Transaction(date='2026-05-01', description='IRCTC', amount=3200, tx_type='Debit', category='Travel'),
            Transaction(date='2026-05-02', description='SALARY', amount=120000, tx_type='Credit', category='Income'),
        ]
        for tx in txs:
            p = score_transaction(tx)
            assert -1.0 <= p.composite_score <= 1.0

    def test_virtue_dashboard(self):
        from perspective_engine import score_transaction, get_virtue_dashboard
        from transaction_parser import Transaction
        txs = [
            Transaction(date='2026-05-01', description='UDEMY', amount=1299, tx_type='Debit', category='Learning'),
            Transaction(date='2026-05-02', description='AMAZON', amount=2400, tx_type='Debit', category='Shopping'),
        ]
        profiles = [score_transaction(tx) for tx in txs]
        cat_summary = {
            'Learning': {'total_spend': 1299},
            'Shopping': {'total_spend': 2400},
        }
        vd = get_virtue_dashboard(profiles, cat_summary)
        assert 'by_pct' in vd
        assert 'by_spend' in vd

    def test_gambling_special_category(self):
        from perspective_engine import score_transaction
        from transaction_parser import Transaction
        tx = Transaction(date='2026-05-01', description='DREAM11 FANTASY', amount=500, tx_type='Debit', category='Entertainment')
        profile = score_transaction(tx)
        # Should hit gambling special category
        assert profile.scores['islamic'].score <= -0.90
        assert profile.scores['buddhist'].score <= -0.70
