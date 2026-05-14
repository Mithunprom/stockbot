"""PSI — Population Stability Index for feature drift detection.

Measures how much a feature's distribution has shifted between a reference
period (training data) and a current period (live data). The standard metric
for detecting feature drift in deployed ML models.

Formula:
    PSI = sum( (actual_pct_i - expected_pct_i) * ln(actual_pct_i / expected_pct_i) )

Interpretation:
    PSI < 0.10  : No significant shift (stable)
    0.10 <= PSI < 0.25 : Moderate shift (warning)
    PSI >= 0.25 : Significant shift (critical — escalation per CLAUDE.md)

Usage:
    from src.features.psi import compute_psi, generate_drift_report

    psi = compute_psi(reference_array, current_array, bins=10)

    report = generate_drift_report(
        reference_path=Path("models/lgbm/reference_distributions.json"),
        current_df=live_feature_df,
        feature_cols=feature_cols,
    )
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import structlog

logger = structlog.get_logger(__name__)

# ─── Thresholds ──────────────────────────────────────────────────────────────

PSI_STABLE = 0.10
PSI_WARNING = 0.25

# Small epsilon to avoid log(0) and division by zero in bin proportions
_EPSILON = 1e-6

# Default storage path for reference distributions
DEFAULT_REFERENCE_PATH = Path("models/lgbm/reference_distributions.json")


# ─── Core PSI computation ────────────────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int = 10,
) -> float:
    """Compute the Population Stability Index between two distributions.

    Supports both numeric (continuous) and categorical features.
    For numeric features, bin edges are derived from the reference distribution
    so that comparisons are stable across time.

    Args:
        reference: 1-D array of reference (training) values.
        current: 1-D array of current (live/test) values.
        bins: Number of quantile bins for numeric features.

    Returns:
        PSI score (non-negative float). 0.0 means identical distributions.

    Raises:
        ValueError: If reference or current array is empty after dropping NaN.
    """
    # Clean inputs — drop NaN/inf
    reference = np.asarray(reference, dtype=float)
    current = np.asarray(current, dtype=float)
    reference = reference[np.isfinite(reference)]
    current = current[np.isfinite(current)]

    if len(reference) == 0:
        raise ValueError("Reference array is empty after removing NaN/inf")
    if len(current) == 0:
        raise ValueError("Current array is empty after removing NaN/inf")

    # Detect categorical: if the reference has <= bins unique values, treat
    # each unique value as its own bin (no quantile bucketing).
    unique_ref = np.unique(reference)
    if len(unique_ref) <= bins:
        return _psi_categorical(reference, current, unique_ref)

    return _psi_numeric(reference, current, bins)


def _psi_numeric(
    reference: np.ndarray,
    current: np.ndarray,
    bins: int,
) -> float:
    """PSI for continuous features using quantile-based binning."""
    # Compute bin edges from reference quantiles so bins are stable
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(reference, quantiles)

    # Ensure unique edges (can collapse if reference has many identical values)
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        # All reference values identical — PSI is 0 if current is the same
        return 0.0

    # Extend edges to capture any out-of-range current values
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=bin_edges)[0].astype(float)
    cur_counts = np.histogram(current, bins=bin_edges)[0].astype(float)

    return _psi_from_counts(ref_counts, cur_counts)


def _psi_categorical(
    reference: np.ndarray,
    current: np.ndarray,
    categories: np.ndarray,
) -> float:
    """PSI for categorical (or low-cardinality) features."""
    ref_counts = np.array([np.sum(reference == cat) for cat in categories], dtype=float)
    cur_counts = np.array([np.sum(current == cat) for cat in categories], dtype=float)

    # Add a catch-all bin for values in current that don't appear in reference
    novel_count = float(len(current) - cur_counts.sum())
    if novel_count > 0:
        ref_counts = np.append(ref_counts, 0.0)
        cur_counts = np.append(cur_counts, novel_count)

    return _psi_from_counts(ref_counts, cur_counts)


def _psi_from_counts(ref_counts: np.ndarray, cur_counts: np.ndarray) -> float:
    """Compute PSI from raw bin counts, adding epsilon to prevent log(0)."""
    # Convert to proportions
    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    # Add epsilon to avoid division by zero and log(0)
    ref_pct = np.clip(ref_pct, _EPSILON, None)
    cur_pct = np.clip(cur_pct, _EPSILON, None)

    # Re-normalize after clipping so proportions sum to 1
    ref_pct = ref_pct / ref_pct.sum()
    cur_pct = cur_pct / cur_pct.sum()

    psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
    return float(psi)


# ─── Batch computation ───────────────────────────────────────────────────────

def compute_feature_psi(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    feature_cols: list[str],
    bins: int = 10,
) -> dict[str, float]:
    """Compute PSI for each feature column between reference and current data.

    Args:
        reference_df: DataFrame with training-period feature values.
        current_df: DataFrame with current-period feature values.
        feature_cols: List of column names to compute PSI for.
        bins: Number of quantile bins for numeric features.

    Returns:
        Dict mapping feature_name -> PSI score.
    """
    results: dict[str, float] = {}

    for col in feature_cols:
        if col not in reference_df.columns:
            logger.warning("psi_skip_missing_ref", feature=col)
            continue
        if col not in current_df.columns:
            logger.warning("psi_skip_missing_cur", feature=col)
            continue

        try:
            ref_vals = reference_df[col].values
            cur_vals = current_df[col].values
            results[col] = compute_psi(ref_vals, cur_vals, bins=bins)
        except ValueError as exc:
            logger.warning("psi_compute_error", feature=col, error=str(exc))
            results[col] = float("nan")

    return results


# ─── Reference distribution storage ─────────────────────────────────────────

def save_reference_distribution(
    df: pd.DataFrame,
    feature_cols: list[str],
    path: Path | None = None,
    bins: int = 10,
) -> None:
    """Compute and save reference bin edges + proportions for each feature.

    Stores the quantile bin edges and expected proportions as JSON so that
    future drift checks can compare against them without needing the full
    reference dataset.

    Args:
        df: Reference DataFrame (typically the training data).
        feature_cols: Ordered list of feature column names.
        path: Output JSON path. Defaults to models/lgbm/reference_distributions.json.
        bins: Number of quantile bins.
    """
    path = Path(path or DEFAULT_REFERENCE_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)

    distributions: dict[str, dict[str, Any]] = {}

    for col in feature_cols:
        if col not in df.columns:
            logger.warning("save_ref_skip_missing", feature=col)
            continue

        values = df[col].dropna().values.astype(float)
        values = values[np.isfinite(values)]

        if len(values) == 0:
            logger.warning("save_ref_skip_empty", feature=col)
            continue

        unique_vals = np.unique(values)
        is_categorical = len(unique_vals) <= bins

        if is_categorical:
            # Store category counts and proportions
            categories = unique_vals.tolist()
            counts = [int(np.sum(values == cat)) for cat in categories]
            total = sum(counts)
            proportions = [c / total for c in counts]
            distributions[col] = {
                "type": "categorical",
                "categories": categories,
                "proportions": proportions,
                "n_samples": len(values),
            }
        else:
            # Store quantile bin edges and proportions
            quantiles = np.linspace(0, 100, bins + 1)
            bin_edges = np.percentile(values, quantiles)
            bin_edges_unique = np.unique(bin_edges)

            if len(bin_edges_unique) < 2:
                # Degenerate: all values the same
                distributions[col] = {
                    "type": "constant",
                    "value": float(bin_edges_unique[0]),
                    "n_samples": len(values),
                }
                continue

            # Extend edges for histogramming
            edges_for_hist = bin_edges_unique.copy()
            edges_for_hist[0] = -np.inf
            edges_for_hist[-1] = np.inf

            counts = np.histogram(values, bins=edges_for_hist)[0].astype(float)
            proportions = (counts / counts.sum()).tolist()

            # Store finite edges for JSON (replace inf with the actual boundary)
            json_edges = bin_edges_unique.tolist()

            distributions[col] = {
                "type": "numeric",
                "bin_edges": json_edges,
                "proportions": proportions,
                "n_samples": len(values),
            }

    payload = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_features": len(distributions),
        "bins": bins,
        "distributions": distributions,
    }

    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    logger.info(
        "reference_distributions_saved",
        path=str(path),
        n_features=len(distributions),
    )


def load_reference_distribution(path: Path | None = None) -> dict:
    """Load a previously saved reference distribution from JSON.

    Args:
        path: Path to the reference distribution JSON file.

    Returns:
        Full payload dict with keys: generated_at, n_features, bins, distributions.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path or DEFAULT_REFERENCE_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Reference distribution not found: {path}")

    with open(path) as f:
        data = json.load(f)

    logger.info(
        "reference_distributions_loaded",
        path=str(path),
        n_features=data.get("n_features", 0),
        generated_at=data.get("generated_at", "unknown"),
    )
    return data


# ─── PSI from saved reference (no full dataset needed) ──────────────────────

def _psi_from_reference(
    ref_dist: dict[str, Any],
    current_values: np.ndarray,
) -> float:
    """Compute PSI using stored reference distribution against current values.

    This avoids needing the full reference dataset at comparison time — only
    the saved bin edges and proportions are required.

    Args:
        ref_dist: Per-feature distribution dict from saved reference.
        current_values: 1-D array of current values for this feature.

    Returns:
        PSI score.
    """
    current_values = np.asarray(current_values, dtype=float)
    current_values = current_values[np.isfinite(current_values)]

    if len(current_values) == 0:
        raise ValueError("Current values empty after removing NaN/inf")

    dist_type = ref_dist["type"]

    if dist_type == "constant":
        # Reference was a constant — any deviation is drift
        ref_val = ref_dist["value"]
        if np.all(current_values == ref_val):
            return 0.0
        # Treat as 2-bin: "equal to constant" vs "not equal"
        match_pct = np.mean(current_values == ref_val)
        ref_pct = np.array([1.0, 0.0])
        cur_pct = np.array([match_pct, 1.0 - match_pct])
        return float(_psi_from_counts(
            ref_pct * 1000,  # scale to pseudo-counts
            cur_pct * 1000,
        ))

    if dist_type == "categorical":
        categories = np.array(ref_dist["categories"])
        ref_proportions = np.array(ref_dist["proportions"])

        cur_counts = np.array([np.sum(current_values == cat) for cat in categories], dtype=float)
        novel_count = float(len(current_values) - cur_counts.sum())
        if novel_count > 0:
            ref_proportions = np.append(ref_proportions, 0.0)
            cur_counts = np.append(cur_counts, novel_count)

        # Convert reference proportions to pseudo-counts for consistent calculation
        ref_counts = ref_proportions * float(ref_dist["n_samples"])
        return _psi_from_counts(ref_counts, cur_counts)

    # dist_type == "numeric"
    bin_edges = np.array(ref_dist["bin_edges"])
    ref_proportions = np.array(ref_dist["proportions"])

    # Extend edges to capture out-of-range values
    edges_for_hist = bin_edges.copy()
    edges_for_hist[0] = -np.inf
    edges_for_hist[-1] = np.inf

    cur_counts = np.histogram(current_values, bins=edges_for_hist)[0].astype(float)

    # Convert reference proportions to pseudo-counts
    ref_counts = ref_proportions * float(ref_dist["n_samples"])
    return _psi_from_counts(ref_counts, cur_counts)


# ─── Drift report generation ────────────────────────────────────────────────

def classify_psi(psi_value: float) -> str:
    """Classify a PSI value into a severity level.

    Returns:
        "stable", "warning", or "critical"
    """
    if np.isnan(psi_value):
        return "unknown"
    if psi_value < PSI_STABLE:
        return "stable"
    if psi_value < PSI_WARNING:
        return "warning"
    return "critical"


def generate_drift_report(
    reference_path: Path | None,
    current_df: pd.DataFrame,
    feature_cols: list[str],
) -> dict[str, Any]:
    """Generate a drift report comparing current features against stored reference.

    Uses the saved reference distributions (bin edges + proportions) so the
    full training dataset is not needed at report time.

    Args:
        reference_path: Path to the reference distribution JSON.
                        Defaults to models/lgbm/reference_distributions.json.
        current_df: DataFrame with current feature values.
        feature_cols: List of feature column names to check.

    Returns:
        Dict with keys:
            generated_at: ISO timestamp
            reference_generated_at: when the reference was created
            n_features_checked: number of features evaluated
            overall_severity: worst severity across all features
            features: dict of {feature_name: {psi, severity}}
            stable_features: list of feature names with PSI < 0.10
            warning_features: list of feature names with 0.10 <= PSI < 0.25
            critical_features: list of feature names with PSI >= 0.25
    """
    ref_data = load_reference_distribution(reference_path)
    ref_distributions = ref_data["distributions"]

    feature_results: dict[str, dict[str, Any]] = {}
    stable: list[str] = []
    warning: list[str] = []
    critical: list[str] = []

    for col in feature_cols:
        if col not in ref_distributions:
            logger.warning("drift_skip_no_reference", feature=col)
            continue
        if col not in current_df.columns:
            logger.warning("drift_skip_no_current", feature=col)
            continue

        try:
            current_values = current_df[col].values
            psi_value = _psi_from_reference(ref_distributions[col], current_values)
        except (ValueError, KeyError) as exc:
            logger.warning("drift_compute_error", feature=col, error=str(exc))
            psi_value = float("nan")

        severity = classify_psi(psi_value)
        feature_results[col] = {
            "psi": round(psi_value, 6) if np.isfinite(psi_value) else None,
            "severity": severity,
        }

        if severity == "stable":
            stable.append(col)
        elif severity == "warning":
            warning.append(col)
        elif severity == "critical":
            critical.append(col)

    # Overall severity: worst across all features
    if critical:
        overall = "critical"
    elif warning:
        overall = "warning"
    else:
        overall = "stable"

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "reference_generated_at": ref_data.get("generated_at", "unknown"),
        "n_features_checked": len(feature_results),
        "overall_severity": overall,
        "features": feature_results,
        "stable_features": stable,
        "warning_features": warning,
        "critical_features": critical,
    }

    logger.info(
        "drift_report_generated",
        n_features=len(feature_results),
        overall=overall,
        n_stable=len(stable),
        n_warning=len(warning),
        n_critical=len(critical),
    )

    return report
