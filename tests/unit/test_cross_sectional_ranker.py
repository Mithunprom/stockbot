"""Unit tests for CrossSectionalRanker (H1 hypothesis).

All tests are self-contained: no numpy, pandas, yfinance, or broker deps.
The ranker is pure-Python datetime logic + sorting — fast and deterministic.
"""

from __future__ import annotations

from datetime import date, datetime
from zoneinfo import ZoneInfo

import pytest

from src.agents.cross_sectional_ranker import (
    CS_DIR_PROB_MIN,
    CS_MAX_SECTOR_SLOTS,
    CS_PRED_RETURN_FLOOR,
    DEFAULT_SNAPSHOT_HOUR,
    DEFAULT_SNAPSHOT_MINUTE,
    CrossSectionalRanker,
    RankCandidate,
)

ET = ZoneInfo("America/New_York")


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _ts(hour: int, minute: int, day: int = 1) -> datetime:
    """Return an ET-aware datetime for 2026-07-{day} at hour:minute."""
    return datetime(2026, 7, day, hour, minute, tzinfo=ET)


def _ranker(**kwargs) -> CrossSectionalRanker:
    return CrossSectionalRanker(**kwargs)


def _feed(ranker: CrossSectionalRanker, tickers: list[tuple]) -> None:
    """Feed (ticker, pred_return, dir_prob) tuples into ranker.update()."""
    for ticker, pred, prob in tickers:
        ranker.update(ticker, pred, prob, price=100.0, atr_pct=0.001,
                      daily_vol=0.02)


# ─── Snapshot timing ─────────────────────────────────────────────────────────

class TestSnapshotTiming:
    def test_returns_none_before_snapshot_time(self):
        r = _ranker()
        _feed(r, [("AAPL", 0.01, 0.70)])
        assert r.get_ranked_candidates(_ts(13, 59)) is None

    def test_returns_none_after_snapshot_time(self):
        r = _ranker()
        _feed(r, [("AAPL", 0.01, 0.70)])
        assert r.get_ranked_candidates(_ts(14, 1)) is None

    def test_returns_list_exactly_at_snapshot_time(self):
        r = _ranker()
        _feed(r, [("AAPL", 0.01, 0.70)])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert result is not None

    def test_fires_only_once_per_trading_day(self):
        r = _ranker()
        _feed(r, [("AAPL", 0.01, 0.70)])
        # First call at 14:00 → fires
        first = r.get_ranked_candidates(_ts(14, 0, day=1))
        assert first is not None
        # Second call same day at 14:00 (same timestamp — should not re-fire)
        second = r.get_ranked_candidates(_ts(14, 0, day=1))
        assert second is None

    def test_fires_again_next_day(self):
        r = _ranker()
        _feed(r, [("AAPL", 0.01, 0.70)])
        r.get_ranked_candidates(_ts(14, 0, day=1))   # day 1 fires
        _feed(r, [("AAPL", 0.01, 0.70)])
        result = r.get_ranked_candidates(_ts(14, 0, day=2))  # day 2 should fire
        assert result is not None

    def test_custom_snapshot_time_respected(self):
        r = _ranker(snapshot_hour=13, snapshot_minute=30)
        _feed(r, [("AAPL", 0.01, 0.70)])
        assert r.get_ranked_candidates(_ts(14, 0)) is None
        assert r.get_ranked_candidates(_ts(13, 30)) is not None


# ─── Qualification filters ────────────────────────────────────────────────────

class TestQualificationFilters:
    def test_negative_pred_return_excluded(self):
        r = _ranker()
        _feed(r, [("AAPL", -0.01, 0.80)])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert result == []

    def test_zero_pred_return_excluded(self):
        r = _ranker()
        _feed(r, [("AAPL", 0.0, 0.80)])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert result == []

    def test_below_floor_excluded(self):
        r = _ranker()
        # CS_PRED_RETURN_FLOOR = 0.001; exactly at floor is NOT > floor
        _feed(r, [("AAPL", CS_PRED_RETURN_FLOOR, 0.80)])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert result == []

    def test_above_floor_included(self):
        r = _ranker()
        _feed(r, [("AAPL", CS_PRED_RETURN_FLOOR + 0.0001, 0.80)])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert len(result) == 1

    def test_low_dir_prob_excluded(self):
        r = _ranker()
        _feed(r, [("AAPL", 0.01, CS_DIR_PROB_MIN - 0.01)])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert result == []

    def test_at_dir_prob_min_included(self):
        r = _ranker()
        _feed(r, [("AAPL", 0.01, CS_DIR_PROB_MIN)])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert len(result) == 1

    def test_no_tickers_returns_empty_list_not_none(self):
        r = _ranker()
        result = r.get_ranked_candidates(_ts(14, 0))
        assert result == []


# ─── Ranking order ────────────────────────────────────────────────────────────

class TestRankingOrder:
    def test_highest_pred_return_first(self):
        r = _ranker(k=5)
        _feed(r, [
            ("AAPL", 0.02, 0.70),
            ("MSFT", 0.05, 0.70),
            ("GOOGL", 0.01, 0.70),
        ])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert [c.ticker for c in result] == ["MSFT", "AAPL", "GOOGL"]

    def test_tie_in_pred_broken_by_dir_prob(self):
        r = _ranker(k=5)
        _feed(r, [
            ("AAPL", 0.02, 0.65),
            ("MSFT", 0.02, 0.80),
        ])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert result[0].ticker == "MSFT"

    def test_tie_in_pred_and_prob_broken_alphabetically(self):
        r = _ranker(k=5)
        _feed(r, [
            ("TSLA", 0.02, 0.70),
            ("AAPL", 0.02, 0.70),
            ("MSFT", 0.02, 0.70),
        ])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert [c.ticker for c in result] == ["AAPL", "MSFT", "TSLA"]

    def test_k_limits_output_length(self):
        r = _ranker(k=3)
        _feed(r, [
            ("AAPL", 0.01 * i, 0.70)
            for i, ticker in enumerate(
                ["A", "B", "C", "D", "E"], start=1
            )
        ])
        # feed 5 tickers, expect only 3
        r2 = _ranker(k=3)
        for i, t in enumerate(["A", "B", "C", "D", "E"], start=1):
            r2.update(t, 0.01 * i, 0.70, 100.0, 0.001, 0.02)
        result = r2.get_ranked_candidates(_ts(14, 0))
        assert len(result) == 3

    def test_k_equals_one(self):
        r = _ranker(k=1)
        _feed(r, [("AAPL", 0.02, 0.70), ("MSFT", 0.05, 0.70)])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert len(result) == 1
        assert result[0].ticker == "MSFT"

    def test_fewer_qualifiers_than_k_returns_all(self):
        r = _ranker(k=10)
        _feed(r, [("AAPL", 0.02, 0.70)])
        result = r.get_ranked_candidates(_ts(14, 0))
        assert len(result) == 1


# ─── Already-open exclusion ────────────────────────────────────────────────────

class TestAlreadyOpenExclusion:
    def test_open_ticker_excluded(self):
        r = _ranker(k=5)
        _feed(r, [("AAPL", 0.05, 0.80), ("MSFT", 0.03, 0.70)])
        result = r.get_ranked_candidates(_ts(14, 0), already_open={"AAPL"})
        tickers = [c.ticker for c in result]
        assert "AAPL" not in tickers
        assert "MSFT" in tickers

    def test_already_open_empty_set_no_exclusion(self):
        r = _ranker(k=5)
        _feed(r, [("AAPL", 0.05, 0.80)])
        result = r.get_ranked_candidates(_ts(14, 0), already_open=set())
        assert len(result) == 1

    def test_already_open_none_no_exclusion(self):
        r = _ranker(k=5)
        _feed(r, [("AAPL", 0.05, 0.80)])
        result = r.get_ranked_candidates(_ts(14, 0), already_open=None)
        assert len(result) == 1


# ─── Sector diversity ─────────────────────────────────────────────────────────

class TestSectorDiversity:
    SECTOR_MAP = {
        "NVDA": "semis", "AMD": "semis", "AVGO": "semis",
        "AAPL": "tech", "MSFT": "tech",
        "JPM": "financials",
    }

    def test_sector_cap_limits_same_sector(self):
        r = _ranker(k=4, max_sector_slots=2)
        _feed(r, [
            ("NVDA", 0.09, 0.90),
            ("AMD", 0.08, 0.85),
            ("AVGO", 0.07, 0.80),   # 3rd semi — should be excluded
            ("JPM", 0.06, 0.75),
        ])
        result = r.get_ranked_candidates(_ts(14, 0), sector_map=self.SECTOR_MAP)
        tickers = [c.ticker for c in result]
        semis = [t for t in tickers if self.SECTOR_MAP.get(t) == "semis"]
        assert len(semis) <= 2

    def test_sector_cap_fills_remaining_from_other_sectors(self):
        r = _ranker(k=4, max_sector_slots=2)
        _feed(r, [
            ("NVDA", 0.09, 0.90),
            ("AMD", 0.08, 0.85),
            ("AVGO", 0.07, 0.80),   # 3rd semi — skipped
            ("JPM", 0.06, 0.75),    # financials — should fill slot 3
            ("AAPL", 0.05, 0.70),   # tech — should fill slot 4
        ])
        result = r.get_ranked_candidates(_ts(14, 0), sector_map=self.SECTOR_MAP)
        tickers = [c.ticker for c in result]
        assert len(result) == 4
        assert "AVGO" not in tickers
        assert "JPM" in tickers
        assert "AAPL" in tickers

    def test_no_sector_map_no_diversity_enforcement(self):
        r = _ranker(k=4, max_sector_slots=1)
        _feed(r, [
            ("NVDA", 0.09, 0.90),
            ("AMD", 0.08, 0.85),
            ("AVGO", 0.07, 0.80),
            ("MU", 0.06, 0.75),
        ])
        # Without sector_map, all 4 should be returned even if they're all semis
        result = r.get_ranked_candidates(_ts(14, 0), sector_map=None)
        assert len(result) == 4

    def test_unknown_ticker_sector_uses_other(self):
        r = _ranker(k=4, max_sector_slots=2)
        _feed(r, [
            ("UNKNOWN1", 0.09, 0.90),
            ("UNKNOWN2", 0.08, 0.85),
            ("UNKNOWN3", 0.07, 0.80),   # third "other" — should be capped
            ("JPM", 0.06, 0.75),
        ])
        result = r.get_ranked_candidates(
            _ts(14, 0), sector_map={"JPM": "financials"}
        )
        tickers = [c.ticker for c in result]
        # At most 2 "other" (UNKNOWN1, UNKNOWN2) + JPM
        others = [t for t in tickers if t.startswith("UNKNOWN")]
        assert len(others) <= 2


# ─── RankCandidate data integrity ─────────────────────────────────────────────

class TestRankCandidateIntegrity:
    def test_update_overwrites_stale_prediction(self):
        r = _ranker(k=5)
        r.update("AAPL", 0.01, 0.60, 100.0, 0.001, 0.02)
        r.update("AAPL", 0.05, 0.80, 101.0, 0.001, 0.02)  # fresher bar
        result = r.get_ranked_candidates(_ts(14, 0))
        assert result[0].pred_return == pytest.approx(0.05)
        assert result[0].dir_prob == pytest.approx(0.80)

    def test_candidate_fields_match_update_args(self):
        r = _ranker(k=5)
        r.update("AAPL", 0.03, 0.72, 150.0, 0.0015, 0.025, regime=2)
        result = r.get_ranked_candidates(_ts(14, 0))
        assert len(result) == 1
        c = result[0]
        assert c.ticker == "AAPL"
        assert c.pred_return == pytest.approx(0.03)
        assert c.dir_prob == pytest.approx(0.72)
        assert c.price == pytest.approx(150.0)
        assert c.atr_pct == pytest.approx(0.0015)
        assert c.daily_vol == pytest.approx(0.025)
        assert c.regime == 2


# ─── Diagnostics ─────────────────────────────────────────────────────────────

class TestDiagnostics:
    def test_snapshot_summary_keys(self):
        r = _ranker(k=3)
        s = r.snapshot_summary()
        assert "k" in s
        assert "snapshot_time" in s
        assert "n_tickers_tracked" in s
        assert s["k"] == 3

    def test_snapshot_summary_tracks_count(self):
        r = _ranker()
        _feed(r, [("AAPL", 0.01, 0.70), ("MSFT", 0.02, 0.75)])
        assert r.snapshot_summary()["n_tickers_tracked"] == 2
