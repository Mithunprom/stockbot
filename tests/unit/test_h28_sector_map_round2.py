"""H28 — SECTOR_MAP Round 2 completeness.

Tickers confirmed in the live trading universe (Oct 2026 diagnostics or open
positions) that were resolving to `unmapped`.  Each is now mapped to its
correct correlation bucket so the per-sector caps apply correctly.

Evidence for each addition:
  ON      — ON Semiconductor; in Oct-7 diagnostics (blocked by kelly_probation)
  SNPS    — Synopsys EDA; in Oct-7 diagnostics (blocked by kelly_probation)
  FICO    — Fair Isaac; in Oct-7 diagnostics (blocked by kelly_probation)
  CRM     — Salesforce; in live universe, H27 PR #62 gap
  COIN    — Coinbase; H27 PR #62 gap; financial-services classification
  MNST    — Monster Beverage; H22 PR #48 gap; confirmed consumer
  NOC     — Northrop Grumman; open position W41 Oct 5 2026 while unmapped
  RTX     — Raytheon; H27 PR #62 gap; defense/industrials
  PCG     — PG&E; confirmed in Sep-17 diagnostics; H27 PR #62 gap; utilities
  CTVA    — Corteva; in Oct-7 diagnostics; agriculture/materials

Freeze classification: FREEZE-EXEMPT — structural correctness fix identical in
nature to H19 (PR #42), H22 (PR #48), and H27 (PR #62).  Adding a ticker
LOOSENS the unmapped cap (1 position, 12.5%) into the correct sector cap
(2 positions, 40%), which is the deliberate, conservative direction per the
SECTOR_MAP design comment in position_sizer.py.
"""

from __future__ import annotations

import pytest

from src.execution.position_sizer import (
    SECTOR_MAP,
    _SECTOR_CAP_PCT,
    _UNMAPPED_SECTOR_CAP_PCT,
    sector_of,
    max_positions_for_sector,
    sector_cap_pct,
)

UNMAPPED_SECTOR = "unmapped"


# ─── Sector resolution ──────────────────────────────────────────────────────

def test_on_semiconductor_maps_to_semis():
    """ON Semi shares the same demand cycle as AMD/NVDA/QCOM — one bucket."""
    assert sector_of("ON") == "semis"


def test_synopsys_maps_to_tech():
    """SNPS makes EDA software (not chips) — correlation bucket is tech."""
    assert sector_of("SNPS") == "tech"


def test_fico_maps_to_tech():
    """Fair Isaac is analytics/decisioning software — correlation bucket is tech."""
    assert sector_of("FICO") == "tech"


def test_crm_maps_to_tech():
    """Salesforce is enterprise cloud software — correlation bucket is tech."""
    assert sector_of("CRM") == "tech"


def test_coin_maps_to_financials():
    """Coinbase is a financial-services exchange — correlation bucket is financials."""
    assert sector_of("COIN") == "financials"


def test_mnst_maps_to_consumer():
    """Monster Beverage is consumer staples — correlation bucket is consumer."""
    assert sector_of("MNST") == "consumer"


def test_noc_maps_to_industrials():
    """Northrop Grumman was an open position Oct 5 2026 while unmapped."""
    assert sector_of("NOC") == "industrials"


def test_rtx_maps_to_industrials():
    """Raytheon is defense/industrials alongside LMT."""
    assert sector_of("RTX") == "industrials"


def test_pcg_maps_to_utilities():
    """PG&E is a regulated utility — needs its own sector, not energy."""
    assert sector_of("PCG") == "utilities"


def test_ctva_maps_to_materials():
    """Corteva is agriculture/crop-protection chemicals — materials sector."""
    assert sector_of("CTVA") == "materials"


# ─── New sectors get standard caps (not unmapped-strictness) ────────────────

def test_utilities_sector_uses_standard_cap():
    """PCG's new sector must not inherit the unmapped per-position strictness."""
    assert max_positions_for_sector("utilities") > max_positions_for_sector(UNMAPPED_SECTOR)
    assert sector_cap_pct("utilities") == pytest.approx(_SECTOR_CAP_PCT)


def test_materials_sector_uses_standard_cap():
    """CTVA's new sector must not inherit the unmapped per-position strictness."""
    assert max_positions_for_sector("materials") > max_positions_for_sector(UNMAPPED_SECTOR)
    assert sector_cap_pct("materials") == pytest.approx(_SECTOR_CAP_PCT)


# ─── M3 live-universe coverage check ────────────────────────────────────────

# Tickers confirmed active in M3 (Sep 23 – Oct 7 2026): appear in the live
# signal loop's diagnostics, open positions, or closed-trade ledger.
M3_LIVE_TICKERS = [
    # M3 closed trades
    "MSTR", "XOM", "COHR", "CVX", "WDC", "ORCL", "LITE",
    "STX", "ARM", "LITE", "NFLX", "KLAC",
    # W41 open positions (Oct 5)
    "WFC", "TER", "NOC",
    # Oct-7 diagnostics (passed both entry gates)
    "CTVA", "FICO", "ON", "SNPS", "JPM", "DELL", "V", "ANET", "LLY", "AMAT",
]


def test_every_m3_live_ticker_is_mapped():
    """No ticker confirmed in the M3 live universe resolves to unmapped."""
    unmapped = [t for t in M3_LIVE_TICKERS if sector_of(t) == UNMAPPED_SECTOR]
    assert unmapped == [], f"Still unmapped after H28: {unmapped}"


# ─── Regression: unmapped bucket still fails closed ─────────────────────────

def test_novel_ticker_still_resolves_to_unmapped():
    """The fail-closed behaviour introduced in 2026-09-15 is intact."""
    assert sector_of("ZZZZ") == UNMAPPED_SECTOR
    assert "ZZZZ" not in SECTOR_MAP
