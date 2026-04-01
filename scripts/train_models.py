"""Train Transformer and TCN signal models (v3.3 — Warmup + IC Loss).

Hybrid dual-head architecture:
  Head A (Regression): Huber warmup on z-scored targets → IC loss fine-tune
  Head B (Direction):  BCEWithLogitsLoss on binary sign(return)

v3.0-3.1: Huber on raw returns → collapsed to predicting zero (target std≈0.003).
v3.2: IC loss only → flat gradient landscape at init, IC stuck at 0.0005.
v3.3: Warmup fix — z-score targets (std=1), Huber for N epochs (strong gradients,
  can't collapse), then switch to IC loss (refines correlation).

Walk-forward split: most recent 20% of bars per ticker used for validation.
MPS (Apple Silicon) acceleration enabled automatically.

Usage:
    python scripts/train_models.py --warmup-epochs 5
    python scripts/train_models.py --epochs 20 --warmup-epochs 5 --accum-steps 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dataset import FORWARD_N, STRIDE, TRADING_THRESHOLD, StockSequenceDataset
from src.models.tcn import TCNSignalModel, save_checkpoint as save_tcn
from src.models.transformer import TransformerSignalModel, save_checkpoint as save_transformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_models")

# ─── Device ───────────────────────────────────────────────────────────────────

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ─── Data loading ─────────────────────────────────────────────────────────────

SP500_TOP50: list[str] = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "AVGO", "AMD", "ORCL", "ADBE",
    "JPM", "V", "MA", "GS", "BAC", "WFC", "BLK", "AXP", "SCHW", "MS",
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "ISRG", "AMGN", "PFE",
    "COST", "WMT", "MCD", "HD", "NKE", "SBUX", "TGT", "LOW", "PG", "KO",
    "XOM", "CVX", "CAT", "RTX", "LMT", "GE", "NEE", "SO", "DUK", "CL",
]


async def load_training_data(top_n: int, tickers: list[str] | None = None, max_rows: int = 100_000) -> tuple[pd.DataFrame, list[str]]:
    """Load feature matrix + close prices, compute forward return, merge."""
    from sqlalchemy import select
    from src.data.db import FeatureMatrix, OHLCV1m, get_session_factory, init_db

    await init_db()
    session_factory = get_session_factory()

    ffsa_files = sorted(Path("reports/drift").glob("ffsa_*.json"), reverse=True)
    if not ffsa_files:
        raise FileNotFoundError("No FFSA report found — run scripts/run_ffsa.py first")
    with open(ffsa_files[0]) as f:
        ffsa = json.load(f)
    feature_cols: list[str] = ffsa["selected_features"][:top_n]
    logger.info("  Using %d FFSA features from %s", len(feature_cols), ffsa_files[0].name)

    from main import _DEFAULT_UNIVERSE
    if tickers is None:
        tickers = _DEFAULT_UNIVERSE
    tickers = [t for t in tickers if "/" not in t]
    logger.info("  Training universe: %d tickers | max %d rows each", len(tickers), max_rows)

    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        async with session_factory() as session:
            result = await session.execute(
                select(FeatureMatrix.time, FeatureMatrix.features)
                .where(FeatureMatrix.ticker == ticker)
                .order_by(FeatureMatrix.time.desc())
                .limit(max_rows)
            )
            feat_rows = result.all()

        if not feat_rows:
            logger.warning("  %s: no feature rows — skipping", ticker)
            continue

        feat_rows = list(reversed(feat_rows))

        records = []
        for ts, feat_dict in feat_rows:
            if not isinstance(feat_dict, dict):
                continue
            rec: dict = {"time": ts}
            for f in feature_cols:
                rec[f] = float(feat_dict.get(f) or 0.0)
            records.append(rec)
        del feat_rows

        ticker_df = pd.DataFrame(records)
        ticker_df["time"] = pd.to_datetime(ticker_df["time"], utc=True)
        for col in feature_cols:
            if col in ticker_df.columns:
                ticker_df[col] = ticker_df[col].astype(np.float32)
        del records

        async with session_factory() as session:
            result = await session.execute(
                select(OHLCV1m.time, OHLCV1m.close)
                .where(OHLCV1m.ticker == ticker)
                .order_by(OHLCV1m.time)
            )
            price_rows = result.all()

        close_s = pd.Series(
            {r.time: float(r.close) for r in price_rows}, name="close"
        )
        close_s.index = pd.to_datetime(close_s.index, utc=True)

        fwd_15 = close_s.pct_change(FORWARD_N).shift(-FORWARD_N).rename("forward_return")
        fwd_5  = close_s.pct_change(5).shift(-5).rename("forward_return_5m")
        fwd_30 = close_s.pct_change(30).shift(-30).rename("forward_return_30m")
        del price_rows, close_s

        ticker_df = ticker_df.set_index("time").sort_index()
        ticker_df = ticker_df.join(fwd_15, how="inner")
        ticker_df = ticker_df.join(fwd_5,  how="left")
        ticker_df = ticker_df.join(fwd_30, how="left")
        ticker_df = ticker_df.dropna(subset=["forward_return"])
        ticker_df = ticker_df.reset_index()
        ticker_df["ticker"] = ticker
        for col in ["forward_return", "forward_return_5m", "forward_return_30m"]:
            if col in ticker_df.columns:
                ticker_df[col] = ticker_df[col].astype(np.float32)
        del fwd_15, fwd_5, fwd_30

        all_frames.append(ticker_df)
        logger.info("  %s: %d rows", ticker, len(ticker_df))

    merged = pd.concat(all_frames, ignore_index=True)
    del all_frames

    feature_cols = [c for c in feature_cols if c in merged.columns]
    logger.info("  Total merged: %d rows, %d features", len(merged), len(feature_cols))
    return merged, feature_cols


# ─── Walk-forward split ───────────────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame, val_frac: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_frames, val_frames = [], []
    time_col = "time" if "time" in df.columns else "index"
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values(time_col)
        cutoff = int(len(grp) * (1 - val_frac))
        train_frames.append(grp.iloc[:cutoff])
        val_frames.append(grp.iloc[cutoff:])
    return pd.concat(train_frames, ignore_index=True), pd.concat(val_frames, ignore_index=True)


# ─── Hybrid loss ─────────────────────────────────────────────────────────────

def _ic_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Negative Pearson correlation loss. Directly optimizes IC.

    Key property: CANNOT be minimized by predicting zero/constant.
    Constant predictions → IC=0 → loss=1.0. The model MUST spread its
    predictions to correlate with targets.

    Returns scalar in [0, 2]: 0=perfect correlation, 1=uncorrelated, 2=anti-correlated.
    """
    pred_c = pred - pred.mean()
    tgt_c = target - target.mean()
    num = (pred_c * tgt_c).sum()
    den = torch.sqrt((pred_c ** 2).sum() * (tgt_c ** 2).sum() + 1e-8)
    return 1.0 - num / den


def _zscore_huber_loss(
    pred: torch.Tensor, target: torch.Tensor,
    target_mean: float, target_std: float,
) -> torch.Tensor:
    """Huber loss on z-scored targets. Fixes the collapse-to-zero problem.

    Raw returns (std≈0.003): Huber rewards predicting zero → model collapses.
    Z-scored returns (std=1): predicting zero gives loss≈0.5 → model must learn.
    """
    target_z = (target - target_mean) / (target_std + 1e-8)
    return nn.functional.huber_loss(pred, target_z, delta=1.0)


def _hybrid_loss(
    bce_fn: nn.Module,
    pred_return: torch.Tensor,   # (B, 1)
    dir_logit: torch.Tensor,     # (B, 1)
    tgt_return: torch.Tensor,    # (B,)
    dir_label: torch.Tensor,     # (B,) — 0.0 or 1.0
    lambda_reg: float = 1.0,
    lambda_dir: float = 1.0,
    warmup: bool = False,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hybrid loss with warm-up schedule.

    Warmup: Huber on z-scored targets (strong gradients, no collapse).
    Fine-tune: IC loss (directly optimizes correlation).
    Direction BCE active in both phases.
    """
    if warmup:
        reg_loss = _zscore_huber_loss(pred_return.squeeze(-1), tgt_return, target_mean, target_std)
    else:
        reg_loss = _ic_loss(pred_return.squeeze(-1), tgt_return)
    dir_loss = bce_fn(dir_logit.squeeze(-1), dir_label)
    total = lambda_reg * reg_loss + lambda_dir * dir_loss
    return total, reg_loss, dir_loss


# ─── Metrics ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    bce_fn: nn.Module,
    device: torch.device,
    model_type: str,
    lambda_reg: float,
    lambda_dir: float,
    warmup: bool = False,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> dict[str, float]:
    """Evaluate and return metrics dict.

    Primary metric for early stopping: binary_dir_acc (Head B accuracy).
    Always computes IC as a tracking metric regardless of loss type.
    """
    model.eval()
    total_loss, total_reg, total_dir = 0.0, 0.0, 0.0
    total = 0
    all_pred_returns: list[float] = []
    all_targets: list[float] = []
    all_dir_probs: list[float] = []
    all_dir_labels: list[float] = []

    for batch in loader:
        x_1m, x_5m, tgt_15, tgt_5, tgt_30, dir_lbl, ticker_ids = batch
        x_1m       = x_1m.to(device)
        x_5m       = x_5m.to(device)
        tgt_15     = tgt_15.float().to(device)
        dir_lbl    = dir_lbl.float().to(device)
        ticker_ids = ticker_ids.to(device)

        if model_type == "transformer":
            pred_ret, dir_logit, _ = model(x_1m, ticker_ids=ticker_ids)
        else:
            pred_ret, dir_logit, _ = model(
                x_1m.transpose(1, 2), x_5m.transpose(1, 2), ticker_ids=ticker_ids
            )

        loss, rl, dl = _hybrid_loss(bce_fn, pred_ret, dir_logit, tgt_15, dir_lbl, lambda_reg, lambda_dir, warmup, target_mean, target_std)
        total_loss += loss.item() * len(tgt_15)
        total_reg  += rl.item() * len(tgt_15)
        total_dir  += dl.item() * len(tgt_15)
        total += len(tgt_15)

        all_pred_returns.extend(pred_ret.squeeze(-1).cpu().tolist())
        all_targets.extend(tgt_15.cpu().tolist())
        all_dir_probs.extend(torch.sigmoid(dir_logit).squeeze(-1).cpu().tolist())
        all_dir_labels.extend(dir_lbl.cpu().tolist())

    preds_arr = np.array(all_pred_returns)
    tgts_arr = np.array(all_targets)
    dir_probs = np.array(all_dir_probs)
    dir_labels = np.array(all_dir_labels)

    # IC (regression quality)
    ic = float(np.corrcoef(preds_arr, tgts_arr)[0, 1]) if len(preds_arr) > 10 else 0.0
    if np.isnan(ic):
        ic = 0.0

    # Binary direction accuracy (Head B — PRIMARY metric for early stopping)
    dir_preds = (dir_probs > 0.5).astype(float)
    binary_dir_acc = float((dir_preds == dir_labels).mean())

    # Non-flat directional accuracy (only samples where regression |pred| > threshold)
    nf_mask = np.abs(preds_arr) > TRADING_THRESHOLD
    nf_count = int(nf_mask.sum())
    if nf_count > 0:
        nf_dir_preds = (dir_probs[nf_mask] > 0.5).astype(float)
        nf_dir_acc = float((nf_dir_preds == dir_labels[nf_mask]).mean())
    else:
        nf_dir_acc = 0.0

    # Regression pred stats
    pred_std = preds_arr.std()
    pct_above = (np.abs(preds_arr) > TRADING_THRESHOLD).mean() * 100

    # Direction head confidence distribution
    dir_mean_prob = dir_probs.mean()

    logger.info(
        "  reg: pred_std=%.6f | %.1f%% above threshold | IC=%.4f",
        pred_std, pct_above, ic,
    )
    logger.info(
        "  dir: binary_acc=%.3f | nf_acc=%.3f (%d signals) | mean_prob=%.3f",
        binary_dir_acc, nf_dir_acc, nf_count, dir_mean_prob,
    )

    return {
        "total_loss": total_loss / max(total, 1),
        "reg_loss": total_reg / max(total, 1),
        "dir_loss": total_dir / max(total, 1),
        "ic": ic,
        "binary_dir_acc": binary_dir_acc,
        "nf_dir_acc": nf_dir_acc,
        "nf_count": nf_count,
        "pred_std": pred_std,
    }


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_type: str,
    epochs: int,
    lr: float,
    save_fn,
    lambda_reg: float = 1.0,
    lambda_dir: float = 1.0,
    patience: int = 5,
    accum_steps: int = 1,
    thermal_sleep: int = 30,
    warmup_epochs: int = 5,
    target_mean: float = 0.0,
    target_std: float = 1.0,
) -> None:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    bce_fn = nn.BCEWithLogitsLoss()
    logger.info("Warmup: %d epochs (Huber on z-scored targets), then IC loss", warmup_epochs)
    logger.info("Target stats: mean=%.6f, std=%.6f", target_mean, target_std)

    best_dir_acc = 0.0
    best_ic = 0.0
    epochs_without_improvement = 0
    step = 0

    for epoch in range(1, epochs + 1):
        is_warmup = epoch <= warmup_epochs
        if epoch == warmup_epochs + 1:
            logger.info("━" * 50)
            logger.info("WARMUP COMPLETE — switching from Huber to IC loss")
            logger.info("━" * 50)
        model.train()
        epoch_loss, epoch_reg, epoch_dir, n_batches = 0.0, 0.0, 0.0, 0
        t0 = time.time()
        optimizer.zero_grad()

        for batch_idx, batch in enumerate(train_loader):
            x_1m, x_5m, tgt_15, tgt_5, tgt_30, dir_lbl, ticker_ids = batch
            x_1m       = x_1m.to(device)
            x_5m       = x_5m.to(device)
            tgt_15     = tgt_15.float().to(device)
            dir_lbl    = dir_lbl.float().to(device)
            ticker_ids = ticker_ids.to(device)

            if model_type == "transformer":
                pred_ret, dir_logit, _ = model(x_1m, ticker_ids=ticker_ids)
            else:
                pred_ret, dir_logit, _ = model(
                    x_1m.transpose(1, 2), x_5m.transpose(1, 2), ticker_ids=ticker_ids
                )

            loss, rl, dl = _hybrid_loss(
                bce_fn, pred_ret, dir_logit, tgt_15, dir_lbl,
                lambda_reg, lambda_dir, is_warmup, target_mean, target_std,
            )

            # Gradient accumulation
            loss_scaled = loss / accum_steps
            loss_scaled.backward()

            if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(train_loader):
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1

            epoch_loss += loss.item()
            epoch_reg  += rl.item()
            epoch_dir  += dl.item()
            n_batches  += 1

        # Validation
        metrics = eval_epoch(model, val_loader, bce_fn, device, model_type, lambda_reg, lambda_dir, is_warmup, target_mean, target_std)
        val_dir_acc = metrics["binary_dir_acc"]
        scheduler.step(val_dir_acc)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        phase = "WARMUP" if is_warmup else "IC"
        reg_label = "huber" if is_warmup else "ic_loss"
        logger.info(
            "[%s][%s] Epoch %d/%d | loss=%.4f (%s=%.4f dir=%.4f) | val_dir_acc=%.3f | IC=%.4f | pred_std=%.6f | lr=%.2e | %.1fs",
            model_type.upper(), phase, epoch, epochs,
            epoch_loss / max(n_batches, 1),
            reg_label,
            epoch_reg / max(n_batches, 1),
            epoch_dir / max(n_batches, 1),
            val_dir_acc, metrics["ic"], metrics["pred_std"], current_lr, elapsed,
        )

        # Early stopping on binary_dir_acc (the whole point of the hybrid head)
        if val_dir_acc > best_dir_acc:
            best_dir_acc = val_dir_acc
            best_ic = metrics["ic"]
            epochs_without_improvement = 0
            # Encode dir_acc × 10 in checkpoint name for easy comparison
            ckpt = save_fn(model, step, round(val_dir_acc * 10, 3))
            logger.info(
                "  ✓ New best: %s (dir_acc=%.3f, IC=%.4f, nf_acc=%.3f)",
                ckpt.name, val_dir_acc, metrics["ic"], metrics["nf_dir_acc"],
            )
        else:
            epochs_without_improvement += 1
            logger.info(
                "  No improvement for %d epoch(s) (best dir_acc=%.3f)",
                epochs_without_improvement, best_dir_acc,
            )

        if epochs_without_improvement >= patience:
            logger.info(
                "[%s] Early stopping at epoch %d. Best dir_acc=%.3f, IC=%.4f",
                model_type.upper(), epoch, best_dir_acc, best_ic,
            )
            break

        # ── Thermal cooldown for MPS ──────────────────────────────────────────
        if thermal_sleep > 0 and epoch < epochs:
            logger.info("  Thermal cooldown: sleeping %ds ...", thermal_sleep)
            time.sleep(thermal_sleep)

    logger.info(
        "[%s] Done. Best dir_acc=%.3f, IC=%.4f",
        model_type.upper(), best_dir_acc, best_ic,
    )


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(
    epochs: int,
    batch_size: int,
    top_n: int,
    lr: float,
    tickers: list[str] | None,
    max_rows: int,
    d_model: int,
    n_layers: int,
    n_channels: int,
    patience: int,
    accum_steps: int,
    lambda_reg: float,
    lambda_dir: float,
    thermal_sleep: int,
    warmup_epochs: int,
) -> None:
    device = _get_device()
    logger.info("Device: %s", device)
    logger.info(
        "Config: d_model=%d | n_layers=%d | n_channels=%d | patience=%d | accum=%d | λ_reg=%.1f | λ_dir=%.1f | warmup=%d",
        d_model, n_layers, n_channels, patience, accum_steps, lambda_reg, lambda_dir, warmup_epochs,
    )
    logger.info("Effective batch size: %d (batch=%d × accum=%d)", batch_size * accum_steps, batch_size, accum_steps)

    merged, feature_cols = await load_training_data(top_n, tickers=tickers, max_rows=max_rows)
    n_features = len(feature_cols)
    logger.info("Features (%d): %s", n_features, feature_cols[:5])

    train_df, val_df = walk_forward_split(merged, val_frac=0.20)
    logger.info("Split: train=%d rows, val=%d rows", len(train_df), len(val_df))

    # ── Target normalization stats (for z-scored Huber warmup) ────────────
    target_mean = float(train_df["forward_return"].mean())
    target_std = float(train_df["forward_return"].std())
    logger.info("Target stats: mean=%.6f, std=%.6f", target_mean, target_std)

    # ── Linear baseline (sanity check: do features have gradient signal?) ─
    try:
        X_tr = np.nan_to_num(train_df[feature_cols].values.astype(np.float32))
        y_tr = train_df["forward_return"].values.astype(np.float32)
        X_va = np.nan_to_num(val_df[feature_cols].values.astype(np.float32))
        y_va = val_df["forward_return"].values.astype(np.float32)
        X_b = np.column_stack([X_tr, np.ones(len(X_tr), dtype=np.float32)])
        XtX = X_b.T @ X_b + np.eye(X_b.shape[1], dtype=np.float32)
        beta = np.linalg.solve(XtX, X_b.T @ y_tr)
        X_vb = np.column_stack([X_va, np.ones(len(X_va), dtype=np.float32)])
        y_pred_lin = X_vb @ beta
        linear_ic = float(np.corrcoef(y_pred_lin, y_va)[0, 1])
        if np.isnan(linear_ic):
            linear_ic = 0.0
        logger.info("━" * 50)
        logger.info("LINEAR BASELINE — Ridge IC: %.4f (%d val samples)", linear_ic, len(y_va))
        logger.info("━" * 50)
        del X_tr, y_tr, X_va, y_va, X_b, XtX, beta, X_vb, y_pred_lin
    except Exception as e:
        logger.warning("Linear baseline failed: %s", e)

    all_tickers = sorted(merged["ticker"].unique())
    ticker_to_id = {t: i for i, t in enumerate(all_tickers)}
    n_tickers = len(all_tickers)
    logger.info("Ticker vocabulary: %d tickers", n_tickers)

    train_ds = StockSequenceDataset(train_df, feature_cols, ticker_to_id=ticker_to_id)
    val_ds   = StockSequenceDataset(val_df, feature_cols, stride=5, ticker_to_id=ticker_to_id)

    if len(train_ds) < 100:
        logger.error("Too few training sequences (%d). Need more data.", len(train_ds))
        return

    # WeightedRandomSampler for ticker balance (full dataset, no curriculum filtering)
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(train_ds),
        replacement=True,
    )

    # Enable augmentation for training dataset
    train_ds.set_augmentation(True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ── Transformer ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Training TRANSFORMER (v3 — Hybrid: Regression + Direction)")
    print("═" * 60)
    n_heads = 4 if d_model <= 64 else 8
    d_ff = d_model * 4
    transformer = TransformerSignalModel(
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        n_tickers=max(n_tickers + 1, 64),
    )
    n_params = sum(p.numel() for p in transformer.parameters())
    logger.info("Transformer params: %s", f"{n_params:,}")

    train_one_model(
        transformer, train_loader, val_loader, device,
        model_type="transformer",
        epochs=epochs,
        lr=lr,
        save_fn=save_transformer,
        lambda_reg=lambda_reg,
        lambda_dir=lambda_dir,
        patience=patience,
        accum_steps=accum_steps,
        thermal_sleep=thermal_sleep,
        warmup_epochs=warmup_epochs,
        target_mean=target_mean,
        target_std=target_std,
    )

    # ── TCN ───────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Training TCN (v3 — Hybrid: Regression + Direction)")
    print("═" * 60)
    tcn = TCNSignalModel(
        n_features_1m=n_features,
        n_features_5m=n_features,
        n_channels=n_channels,
        n_tickers=max(n_tickers + 1, 64),
    )
    n_params_tcn = sum(p.numel() for p in tcn.parameters())
    logger.info("TCN params: %s", f"{n_params_tcn:,}")

    train_one_model(
        tcn, train_loader, val_loader, device,
        model_type="tcn",
        epochs=epochs,
        lr=lr,
        save_fn=save_tcn,
        lambda_reg=lambda_reg,
        lambda_dir=lambda_dir,
        patience=patience,
        accum_steps=accum_steps,
        thermal_sleep=thermal_sleep,
        warmup_epochs=warmup_epochs,
        target_mean=target_mean,
        target_std=target_std,
    )

    print("\n" + "═" * 60)
    print("  v3.3 training complete — checkpoints saved to models/")
    print("  Huber warmup → IC loss | z-scored targets | Dir-acc early stopping")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train models (v3 — Hybrid Loss)")
    parser.add_argument("--epochs",        type=int,   default=20)
    parser.add_argument("--batch-size",    type=int,   default=256)
    parser.add_argument("--top-n",         type=int,   default=30)
    parser.add_argument("--lr",            type=float, default=3e-4)
    parser.add_argument("--max-rows",      type=int,   default=100_000)
    parser.add_argument("--tickers",       nargs="*",  default=None)
    parser.add_argument("--d-model",       type=int,   default=128)
    parser.add_argument("--n-layers",      type=int,   default=5)
    parser.add_argument("--n-channels",    type=int,   default=128)
    parser.add_argument("--patience",      type=int,   default=3)
    parser.add_argument("--accum-steps",   type=int,   default=2,
                        help="Gradient accumulation steps (eff. batch = batch × accum)")
    parser.add_argument("--lambda-reg",    type=float, default=1.0,
                        help="Weight of IC (correlation) regression loss. Both IC and BCE are ~[0,1].")
    parser.add_argument("--lambda-dir",    type=float, default=1.0,
                        help="Weight of BCE direction loss (λ). Higher = more direction forcing.")
    parser.add_argument("--thermal-sleep", type=int,   default=30,
                        help="Seconds to sleep between epochs for MPS thermal management (0=disabled)")
    parser.add_argument("--warmup-epochs", type=int,   default=5,
                        help="Epochs of Huber warmup on z-scored targets before switching to IC loss (0=IC only)")
    args = parser.parse_args()

    ticker_list: list[str] | None = None
    if args.tickers:
        if args.tickers == ["sp500-top50"]:
            ticker_list = SP500_TOP50
            logger.info("Using SP500_TOP50 training universe (%d tickers)", len(ticker_list))
        else:
            ticker_list = args.tickers

    asyncio.run(main(
        epochs=args.epochs,
        batch_size=args.batch_size,
        top_n=args.top_n,
        lr=args.lr,
        tickers=ticker_list,
        max_rows=args.max_rows,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_channels=args.n_channels,
        patience=args.patience,
        accum_steps=args.accum_steps,
        lambda_reg=args.lambda_reg,
        lambda_dir=args.lambda_dir,
        thermal_sleep=args.thermal_sleep,
        warmup_epochs=args.warmup_epochs,
    ))
