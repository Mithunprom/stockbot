"""Unit tests for the News Risk Agent's macro severity scorer."""

from __future__ import annotations

from src.agents.news_risk_agent import (
    current_news_risk_level,
    score_articles,
    score_headline,
)


def art(title: str, desc: str = "") -> dict:
    return {"title": title, "description": desc,
            "source": {"name": "test"}, "publishedAt": "", "url": ""}


class TestScoreHeadline:
    def test_severe_phrases(self):
        assert score_headline("Russia declares war on NATO member")[0] == 3
        assert score_headline("Trading halted as market crash deepens")[0] == 3

    def test_high_phrases(self):
        assert score_headline("Fed calls emergency meeting after bank stress")[0] == 2

    def test_elevated_phrases(self):
        assert score_headline("New tariff round hits chipmakers")[0] == 1

    def test_benign_headline_scores_zero(self):
        lvl, matched = score_headline("Apple unveils new MacBook lineup")
        assert lvl == 0 and matched == []

    def test_bidding_war_not_flagged_as_war(self):
        # "war " / " war," phrase forms shouldn't match "bidding war on X"
        lvl, _ = score_headline("Streaming bidding war heats up over sports rights")
        assert lvl <= 1  # never high/severe from a bidding war


class TestScoreArticles:
    def test_single_severe_headline_demotes_to_high(self):
        # One lone wire report can't declare a market dislocation by itself
        out = score_articles([art("Trading halted after flash crash")])
        assert out["level"] == 2

    def test_two_severe_headlines_stick(self):
        out = score_articles([
            art("Trading halted after flash crash"),
            art("Nuclear threat escalates in region"),
        ])
        assert out["level"] == 3 and out["level_name"] == "severe"

    def test_quiet_tape(self):
        out = score_articles([art("Earnings beat expectations"), art("New phone released")])
        assert out["level"] == 0 and out["n_flagged"] == 0

    def test_empty_feed_is_level_zero(self):
        assert score_articles([])["level"] == 0


class TestGateFailsOpen:
    def test_missing_report_means_no_block(self, tmp_path, monkeypatch):
        import src.agents.news_risk_agent as nra
        monkeypatch.setattr(nra, "REPORT_PATH", tmp_path / "missing.json")
        nra._gate_cache["at"] = None
        assert current_news_risk_level() == 0
