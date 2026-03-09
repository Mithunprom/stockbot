"""FinBERT sentiment model for financial news.

Uses HuggingFace ProsusAI/finbert (pre-trained on financial text).
Computes a Rolling Sentiment Index per ticker using recency decay.
Detects sudden sentiment reversals as high-conviction signals.

Usage:
    scorer = SentimentScorer()
    await scorer.load()
    si = await scorer.rolling_sentiment_index("AAPL", lookback_hours=24)
"""

from __future__ import annotations

import asyncio
import logging
import structlog
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
from sqlalchemy import select, text
from sqlalchemy.orm import Session

from src.data.db import NewsRaw, get_session_factory

logger = structlog.get_logger(__name__)

# Lazy import — transformers is optional until FinBERT is actually used
try:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline as hf_pipeline  # noqa: F401
    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False
    logger.warning("transformers not installed — FinBERT sentiment disabled")

MODEL_NAME = "ProsusAI/finbert"
CHECKPOINT_DIR = Path("models/sentiment")

# Sentiment label mapping from FinBERT output
_LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
_SCORE_TO_LABEL = {1.0: "positive", 0.0: "neutral", -1.0: "negative"}

RECENCY_HALF_LIFE_HOURS = 12   # score decays to 50% after 12 hours


# ─── Relevance scoring ─────────────────────────────────────────────────────────

def compute_relevance(headline: str, body: str | None, ticker: str) -> float:
    """Estimate how relevant an article is specifically to ticker [0, 1].

    Simple heuristic: count direct ticker mentions vs. total words.
    A future version can use a trained relevance classifier.
    """
    full_text = f"{headline} {body or ''}".upper()
    mentions = full_text.count(ticker.upper())
    words = max(len(full_text.split()), 1)
    # Cap at 1.0; diminishing returns after 3 mentions per 100 words
    return min(mentions * 100 / (words * 3), 1.0)


# ─── Recency decay ────────────────────────────────────────────────────────────

def recency_weight(published_at: datetime, half_life_hours: float = RECENCY_HALF_LIFE_HOURS) -> float:
    """Exponential decay weight: 1.0 for now, 0.5 at half_life_hours."""
    age_hours = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600
    return 2 ** (-age_hours / half_life_hours)


# ─── Sentiment scorer ────────────────────────────────────────────────────────

class SentimentScorer:
    """Runs FinBERT inference and maintains the rolling sentiment index."""

    def __init__(self, model_name: str = MODEL_NAME, device: str | None = None) -> None:
        self._model_name = model_name
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipe = None   # loaded lazily

    async def load(self) -> None:
        """Load model in thread pool to avoid blocking the event loop."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self._load_sync)

    def _load_sync(self) -> None:
        if not _TRANSFORMERS_AVAILABLE:
            logger.warning("transformers_unavailable_skipping_finbert_load")
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
        tokenizer = AutoTokenizer.from_pretrained(self._model_name)
        model = AutoModelForSequenceClassification.from_pretrained(self._model_name)
        self._pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            device=0 if self._device == "cuda" else -1,
            top_k=None,   # return all labels
        )
        logger.info("finbert_loaded", device=self._device)

    def score_text(self, text: str) -> tuple[float, str]:
        """Score a single piece of text.

        Returns: (sentiment_score in [-1, 1], label)
        """
        if self._pipe is None:
            raise RuntimeError("SentimentScorer not loaded — call await scorer.load() first")

        results = self._pipe(text[:512], truncation=True)[0]
        # results: [{"label": "positive", "score": 0.9}, ...]
        best = max(results, key=lambda x: x["score"])
        label = best["label"].lower()
        score = _LABEL_TO_SCORE.get(label, 0.0) * best["score"]
        return score, label

    async def score_text_async(self, text: str) -> tuple[float, str]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.score_text, text)

    async def score_unscored_articles(self, batch_size: int = 32) -> int:
        """Pull unscored articles from DB, score them, update sentiment fields.

        Returns the number of articles scored.
        """
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(NewsRaw)
                .where(NewsRaw.sentiment_score.is_(None))
                .order_by(NewsRaw.published_at.desc())
                .limit(batch_size)
            )
            articles = result.scalars().all()

            if not articles:
                return 0

            for article in articles:
                text = f"{article.headline} {article.body or ''}"
                score, label = await self.score_text_async(text)
                relevance = compute_relevance(
                    article.headline, article.body, article.ticker
                )
                article.sentiment_score = score
                article.sentiment_label = label
                article.relevance_score = relevance

            await session.commit()
            logger.info("articles_scored", count=len(articles))
            return len(articles)

    async def rolling_sentiment_index(
        self, ticker: str, lookback_hours: int = 24
    ) -> float:
        """Compute the Rolling Sentiment Index for a ticker.

        SI(t) = weighted_avg(sentiment × relevance × recency_decay)
        Range: approximately [-1, +1]
        """
        since = datetime.now(timezone.utc) - timedelta(hours=lookback_hours)
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(NewsRaw)
                .where(NewsRaw.ticker == ticker.upper())
                .where(NewsRaw.published_at >= since)
                .where(NewsRaw.sentiment_score.is_not(None))
                .order_by(NewsRaw.published_at.desc())
            )
            articles = result.scalars().all()

        if not articles:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for art in articles:
            rw = recency_weight(art.published_at)
            rel = art.relevance_score or 0.5
            w = rw * rel
            weighted_sum += (art.sentiment_score or 0.0) * w
            total_weight += w

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    async def detect_sentiment_reversal(
        self, ticker: str, window_hours: int = 2, threshold: float = 0.5
    ) -> bool:
        """Detect sudden sentiment reversal (high-conviction signal).

        Returns True if the most recent 2h SI differs from the prior 2h by > threshold.
        """
        now = datetime.now(timezone.utc)
        recent_si = await self.rolling_sentiment_index(ticker, lookback_hours=window_hours)

        # Shift window back by window_hours
        session_factory = get_session_factory()
        async with session_factory() as session:
            result = await session.execute(
                select(NewsRaw)
                .where(NewsRaw.ticker == ticker.upper())
                .where(
                    NewsRaw.published_at.between(
                        now - timedelta(hours=2 * window_hours),
                        now - timedelta(hours=window_hours),
                    )
                )
                .where(NewsRaw.sentiment_score.is_not(None))
            )
            prior_articles = result.scalars().all()

        if not prior_articles:
            return False

        prior_scores = [a.sentiment_score or 0.0 for a in prior_articles]
        prior_si = sum(prior_scores) / len(prior_scores)

        reversal = abs(recent_si - prior_si) > threshold
        if reversal:
            logger.info(
                "sentiment_reversal_detected",
                ticker=ticker,
                recent_si=recent_si,
                prior_si=prior_si,
                delta=abs(recent_si - prior_si),
            )
        return reversal


# ─── Checkpoint helpers ───────────────────────────────────────────────────────

def save_finetuned(model: Any, tokenizer: Any, step: int) -> Path:
    """Save a fine-tuned FinBERT checkpoint."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECKPOINT_DIR / f"finbert_step_{step:06d}"
    model.save_pretrained(out_path)
    tokenizer.save_pretrained(out_path)
    logger.info("finbert_checkpoint_saved", path=str(out_path))
    return out_path
