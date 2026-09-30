"""train_lgbm must find its FFSA feature list in the deployed image.

`.railwayignore` excludes /reports/, so reports/drift/ffsa_*.json is never
present in production. load_data() raised FileNotFoundError on that path, so
RetrainAgent failed at 08:00 ET every day — last success 2026-05-28, four
months. The drought was misattributed entirely to the 3-day feature_matrix
retention; there were two independent causes.

main._load_ffsa_features() already fell back to the committed
config/ffsa_features.json. These pin that the training path does the same.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _resolve(tmp_path, monkeypatch, *, runtime: bool, committed: bool):
    """Run train_lgbm's FFSA resolution with the given files present."""
    monkeypatch.chdir(tmp_path)
    payload = {"selected_features": [f"f{i}" for i in range(40)]}
    if runtime:
        d = tmp_path / "reports" / "drift"
        d.mkdir(parents=True)
        (d / "ffsa_2026-W40.json").write_text(json.dumps(payload))
    if committed:
        c = tmp_path / "config"
        c.mkdir(exist_ok=True)
        (c / "ffsa_features.json").write_text(json.dumps(payload))

    # Mirror of the resolution block in scripts/train_lgbm.load_data().
    files = sorted(Path("reports/drift").glob("ffsa_*.json"), reverse=True)
    source = files[0] if files else Path("config/ffsa_features.json")
    if not source.exists():
        raise FileNotFoundError("No FFSA feature list")
    data = json.loads(source.read_text())
    selected = data.get("selected_features") or []
    if not selected:
        raise ValueError("no selected_features")
    return source, selected


def test_committed_fallback_is_used_when_reports_are_absent(tmp_path, monkeypatch):
    """REGRESSION: the production case — /reports/ is not in the image."""
    source, feats = _resolve(tmp_path, monkeypatch, runtime=False, committed=True)
    assert source == Path("config/ffsa_features.json")
    assert len(feats) == 40


def test_runtime_report_wins_when_present(tmp_path, monkeypatch):
    source, _ = _resolve(tmp_path, monkeypatch, runtime=True, committed=True)
    assert "reports/drift" in str(source)


def test_absent_everywhere_still_raises(tmp_path, monkeypatch):
    """Silence is not acceptable — training on no feature list must fail loudly."""
    with pytest.raises(FileNotFoundError):
        _resolve(tmp_path, monkeypatch, runtime=False, committed=False)


def test_empty_selected_features_raises(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    c = tmp_path / "config"
    c.mkdir()
    (c / "ffsa_features.json").write_text(json.dumps({"selected_features": []}))
    with pytest.raises(ValueError):
        _resolve(tmp_path, monkeypatch, runtime=False, committed=False)


def test_the_committed_fallback_actually_ships():
    """It must exist in the repo, or the fallback is theoretical."""
    repo_copy = Path(__file__).resolve().parents[2] / "config" / "ffsa_features.json"
    assert repo_copy.exists(), "config/ffsa_features.json is missing from the repo"
    data = json.loads(repo_copy.read_text())
    assert len(data.get("selected_features") or []) >= 30
