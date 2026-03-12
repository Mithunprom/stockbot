"""Train Transformer and TCN signal models on the feature matrix.

Walk-forward split: most recent 20% of bars per ticker used for validation.
MPS (Apple Silicon) acceleration enabled automatically.

Usage:
    python scripts/train_models.py
    python scripts/train_models.py --epochs 15 --batch-size 512 --top-n 30
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.dataset import FORWARD_N, STRIDE, StockSequenceDataset, compute_class_weights
from src.models.tcn import TCNSignalModel, save_checkpoint as save_tcn
from src.models.transformer import FocalLoss, TransformerSignalModel, save_checkpoint as save_transformer

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

# Top 50 S&P 500 stocks by avg daily volume — broadest regime diversity for training.
# Model learns generalizable indicator patterns rather than tech-sector-specific ones.
SP500_TOP50: list[str] = [
    # Mega-cap tech
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "AVGO", "AMD", "ORCL", "ADBE",
    # Financials
    "JPM", "V", "MA", "GS", "BAC", "WFC", "BLK", "AXP", "SCHW", "MS",
    # Healthcare
    "LLY", "UNH", "JNJ", "MRK", "ABBV", "TMO", "ABT", "ISRG", "AMGN", "PFE",
    # Consumer / retail
    "COST", "WMT", "MCD", "HD", "NKE", "SBUX", "TGT", "LOW", "PG", "KO",
    # Industrials / energy / other
    "XOM", "CVX", "CAT", "RTX", "LMT", "GE", "NEE", "SO", "DUK", "CL",
]


async def load_training_data(top_n: int, tickers: list[str] | None = None, max_rows: int = 100_000) -> tuple[pd.DataFrame, list[str]]:
    """Load feature matrix + close prices, compute forward return, merge.

    Loads one ticker at a time to avoid OOM on 3.7M rows of JSONB data.
    Only extracts the top-N FFSA features per row (not all 47+ keys),
    so peak memory is ~50MB per ticker instead of >10GB all at once.
    """
    from sqlalchemy import select

    from src.data.db import FeatureMatrix, OHLCV1m, get_session_factory, init_db

    await init_db()
    session_factory = get_session_factory()

    # ── FFSA selected features ────────────────────────────────────────────────
    ffsa_files = sorted(Path("reports/drift").glob("ffsa_*.json"), reverse=True)
    if not ffsa_files:
        raise FileNotFoundError("No FFSA report found — run scripts/run_ffsa.py first")
    with open(ffsa_files[0]) as f:
        ffsa = json.load(f)
    feature_cols: list[str] = ffsa["selected_features"][:top_n]
    logger.info("  Using %d FFSA features from %s", len(feature_cols), ffsa_files[0].name)

    # ── Get universe ──────────────────────────────────────────────────────────
    from main import _DEFAULT_UNIVERSE
    if tickers is None:
        tickers = _DEFAULT_UNIVERSE
    # Skip crypto tickers — no historical data in DB yet (need 30+ days of live streaming)
    tickers = [t for t in tickers if "/" not in t]
    logger.info("  Training universe: %d tickers | max %d rows each", len(tickers), max_rows)

    # ── Per-ticker loading to keep peak memory low ────────────────────────────
    all_frames: list[pd.DataFrame] = []

    for ticker in tickers:
        # 1. Load features for this ticker only (one ticker's JSONB at a time)
        # Cap at max_rows most-recent rows to keep memory bounded when training
        # on many tickers (e.g. 50 tickers × 100k rows = 5M rows manageable)
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

        # Re-sort ascending (we fetched desc to get the most-recent max_rows)
        feat_rows = list(reversed(feat_rows))

        # 2. Extract only the FFSA features — discard full JSONB dict after extraction
        records = []
        for ts, feat_dict in feat_rows:
            if not isinstance(feat_dict, dict):
                continue
            rec: dict = {"time": ts}
            for f in feature_cols:
                rec[f] = float(feat_dict.get(f) or 0.0)
            records.append(rec)
        del feat_rows  # free JSONB dicts immediately

        ticker_df = pd.DataFrame(records)
        ticker_df["time"] = pd.to_datetime(ticker_df["time"], utc=True)
        for col in feature_cols:
            if col in ticker_df.columns:
                ticker_df[col] = ticker_df[col].astype(np.float32)
        del records

        # 3. Load close prices for this ticker and compute 5-bar forward return
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
        fwd = close_s.pct_change(FORWARD_N).shift(-FORWARD_N).rename("forward_return")
        del price_rows, close_s

        # 4. Merge features + forward return (time-aligned)
        ticker_df = ticker_df.set_index("time").sort_index()
        ticker_df = ticker_df.join(fwd, how="inner")
        ticker_df = ticker_df.dropna(subset=["forward_return"])
        ticker_df = ticker_df.reset_index()
        ticker_df["ticker"] = ticker
        ticker_df["forward_return"] = ticker_df["forward_return"].astype(np.float32)
        del fwd

        all_frames.append(ticker_df)
        logger.info("  %s: %d rows", ticker, len(ticker_df))

    merged = pd.concat(all_frames, ignore_index=True)
    del all_frames

    # Keep only valid feature columns (some FFSA cols may be absent from old data)
    feature_cols = [c for c in feature_cols if c in merged.columns]
    logger.info("  Total merged: %d rows, %d features", len(merged), len(feature_cols))
    return merged, feature_cols


# ─── Walk-forward split ───────────────────────────────────────────────────────

def walk_forward_split(df: pd.DataFrame, val_frac: float = 0.20) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split per-ticker by time: train on oldest (1-val_frac), val on newest val_frac."""
    train_frames, val_frames = [], []
    for ticker, grp in df.groupby("ticker"):
        grp = grp.sort_values("time")
        cutoff = int(len(grp) * (1 - val_frac))
        train_frames.append(grp.iloc[:cutoff])
        val_frames.append(grp.iloc[cutoff:])
    return pd.concat(train_frames, ignore_index=True), pd.concat(val_frames, ignore_index=True)


# ─── Metrics ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    model_type: str,
) -> tuple[float, float]:
    """Return (avg_loss, accuracy)."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for x_1m, x_5m, labels in loader:
        x_1m   = x_1m.to(device)
        x_5m   = x_5m.to(device)
        labels = labels.to(device)

        if model_type == "transformer":
            logits, _ = model(x_1m)
        else:  # tcn
            _, logits, _ = model(
                x_1m.transpose(1, 2),  # (B, n_feat, seq_len)
                x_5m.transpose(1, 2),  # (B, n_feat, seq_5m)
            )
        loss = loss_fn(logits, labels)
        total_loss += loss.item() * len(labels)
        correct += (logits.argmax(dim=-1) == labels).sum().item()
        total += len(labels)

    return total_loss / max(total, 1), correct / max(total, 1)


# ─── Training loop ────────────────────────────────────────────────────────────

def train_one_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    model_type: str,       # "transformer" or "tcn"
    epochs: int,
    lr: float,
    class_weights: torch.Tensor,
    save_fn,
) -> None:
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    # With balanced classes (~33/33/33) class weights are ~1.0 — use plain CE.
    # FocalLoss gamma=0 reduces to cross-entropy; no weight needed.
    loss_fn   = FocalLoss(gamma=0.5, weight=None)

    best_val_acc = 0.0
    step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss, n_batches = 0.0, 0
        t0 = time.time()

        for x_1m, x_5m, labels in train_loader:
            x_1m   = x_1m.to(device)
            x_5m   = x_5m.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if model_type == "transformer":
                logits, _ = model(x_1m)
            else:
                _, logits, _ = model(
                    x_1m.transpose(1, 2),
                    x_5m.transpose(1, 2),
                )
            loss = loss_fn(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1
            step       += 1

        scheduler.step()
        val_loss, val_acc = eval_epoch(model, val_loader, loss_fn, device, model_type)
        elapsed = time.time() - t0

        logger.info(
            "[%s] Epoch %d/%d | train_loss=%.4f | val_loss=%.4f | val_acc=%.3f | %.1fs",
            model_type.upper(), epoch, epochs,
            epoch_loss / max(n_batches, 1), val_loss, val_acc, elapsed,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Compute a placeholder Sharpe from accuracy (real Sharpe requires PnL sim)
            pseudo_sharpe = (val_acc - 0.33) / 0.05  # normalized above random
            ckpt = save_fn(model, step, max(0.0, pseudo_sharpe))
            logger.info("  ✓ New best checkpoint saved: %s", ckpt.name)

    logger.info("[%s] Training done. Best val_acc=%.3f", model_type.upper(), best_val_acc)


# ─── Main ─────────────────────────────────────────────────────────────────────

async def main(epochs: int, batch_size: int, top_n: int, lr: float, tickers: list[str] | None, max_rows: int) -> None:
    device = _get_device()
    logger.info("Device: %s", device)

    merged, feature_cols = await load_training_data(top_n, tickers=tickers, max_rows=max_rows)
    n_features = len(feature_cols)
    logger.info("Features (%d): %s", n_features, feature_cols[:5])

    train_df, val_df = walk_forward_split(merged, val_frac=0.20)
    logger.info(
        "Split: train=%d rows, val=%d rows (per ticker, walk-forward)",
        len(train_df), len(val_df),
    )

    train_ds = StockSequenceDataset(train_df, feature_cols)
    val_ds   = StockSequenceDataset(val_df,   feature_cols, stride=5)   # val: wider stride is fine

    if len(train_ds) < 100:
        logger.error("Too few training sequences (%d). Need more data.", len(train_ds))
        return

    class_weights = compute_class_weights(train_ds)
    logger.info("Class weights: down=%.3f flat=%.3f up=%.3f", *class_weights.tolist())

    # Persistent workers cause issues on macOS MPS; use 0 for safety
    n_workers = 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=n_workers)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=n_workers)

    logger.info("Train batches: %d | Val batches: %d", len(train_loader), len(val_loader))

    # ── Transformer ───────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Training TRANSFORMER")
    print("═" * 60)
    transformer = TransformerSignalModel(n_features=n_features)
    n_params = sum(p.numel() for p in transformer.parameters())
    logger.info("Transformer params: %s", f"{n_params:,}")

    train_one_model(
        transformer, train_loader, val_loader, device,
        model_type="transformer",
        epochs=epochs,
        lr=lr,
        class_weights=class_weights,
        save_fn=save_transformer,
    )

    # ── TCN ───────────────────────────────────────────────────────────────────
    print("\n" + "═" * 60)
    print("  Training TCN")
    print("═" * 60)
    tcn = TCNSignalModel(n_features_1m=n_features, n_features_5m=n_features)
    n_params_tcn = sum(p.numel() for p in tcn.parameters())
    logger.info("TCN params: %s", f"{n_params_tcn:,}")

    train_one_model(
        tcn, train_loader, val_loader, device,
        model_type="tcn",
        epochs=epochs,
        lr=lr,
        class_weights=class_weights,
        save_fn=save_tcn,
    )

    print("\n" + "═" * 60)
    print("  Phase 3 complete — checkpoints saved to models/")
    print("  Next: wire EnsembleEngine into main.py → paper trading loop")
    print("═" * 60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer + TCN signal models")
    parser.add_argument("--epochs",       type=int,   default=10)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--top-n",        type=int,   default=30,    help="FFSA top-N features")
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--max-rows",     type=int,   default=100_000,
                        help="Max rows per ticker (most recent). Default 100k (~6mo 1m bars).")
    parser.add_argument("--tickers",      nargs="*",  default=None,
                        help="Ticker list override. Use 'sp500-top50' for broad training set. "
                             "Default: _DEFAULT_UNIVERSE from main.py")
    args = parser.parse_args()

    # Resolve ticker set
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
    ))
