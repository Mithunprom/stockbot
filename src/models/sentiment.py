"""FinBERT sentiment via HuggingFace Inference API.

Calls the hosted ProsusAI/finbert endpoint — no local model download needed.
Requires HUGGINGFACE_API_TOKEN env var (free HF account).

Scoring pipeline:
  1. NewsPoller fetches articles → stored in news_raw DB table
  2. score_unscored_articles() calls HF API for each unscored article
  3. rolling_sentiment_index() reads scored articles → computes weighted SI
  4. SI feeds into ensemble: ensemble += 0.20 * sentiment_index

API rate limits (free tier): ~300 requests/min — more than enough for our
news volume (~50-200 articles/day across 24 tickers).
"""

from __future__ import annotations

import asyncio
import os
import structlog
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy import select

from src.data.db import NewsRaw, get_session_factory

logger = structlog.get_logger(__name__)

HF_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HF_TOKEN = os.environ.get("HUGGINGFACE_API_TOKEN", "")

# Sentiment label → numeric score
_LABEL_TO_SCORE = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}

RECENCY_HALF_LIFE_HOURS = 12   # score decays to 50% after 12 hours


# ─── Relevance scoring ────────────────────────────────────────────────────────

def compute_relevance(headline: str, body: str | None, ticker: str) -> float:
    """Estimate how relevant an article is to a specific ticker [0, 1]."""
    full_text = f"{headline} {body or ''}".upper()
    mentions = full_text.count(ticker.upper())
    words = max(len(full_text.split()), 1)
    return min(mentions * 100 / (words * 3), 1.0)


# ─── Recency decay ────────────────────────────────────────────────────────────

def recency_weight(published_at: datetime, half_life_hours: float = RECENCY_HALF_LIFE_HOURS) -> float:
    """Exponential decay: 1.0 now, 0.5 at half_life_hours."""
    age_hours = (datetime.now(timezone.utc) - published_at).total_seconds() / 3600
    return 2 ** (-age_hours / half_life_hours)


# ─── HF Inference API call ────────────────────────────────────────────────────

async def _call_finbert_api(texts: list[str]) -> list[tuple[float, str]]:
    """Call HF Inference API for a batch of texts.

    Returns list of (score, label) tuples. Falls back to (0.0, 'neutral')
    on error so the pipeline never crashes.
    """
    if not HF_TOKEN:
        logger.warning("hf_token_missing_sentiment_disabled")
        return [(0.0, "neutral")] * len(texts)

    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    results = []

    async with httpx.AsyncClient(timeout=30.0) as client:
        for text in texts:
            try:
                resp = await client.post(
                    HF_API_URL,
                    headers=headers,
                    json={"inputs": text[:512]},
                )
                if resp.status_code == 503:
                    # Model loading — wait and retry once
                    await asyncio.sleep(5)
                    resp = await client.post(
                        HF_API_URL,
                        headers=headers,
                        json={"inputs": text[:512]},
                    )
                if resp.status_code != 200:
                    logger.warning("hf_api_error", status=resp.status_code, text=text[:60])
                    results.append((0.0, "neutral"))
                    continue

                data = resp.json()
                # Response: [[{"label": "positive", "score": 0.98}, ...]]
                labels = data[0] if isinstance(data[0], list) else data
                best = max(labels, key=lambda x: x["score"])
                label = best["label"].lower()
                score = _LABEL_TO_SCORE.get(label, 0.0) * best["score"]
                results.append((score, label))

            except Exception as exc:
                logger.warning("hf_api_exception", error=str(exc), text=text[:60])
                results.append((0.0, "neutral"))

    return results


# ─── Sentiment scorer ─────────────────────────────────────────────────────────

class SentimentScorer:
    """Scores financial news via HF Inference API and maintains rolling SI."""

    def __init__(self, model_name: str = "ProsusAI/finbert", device: str | None = None) -> None:
        self._model_name = model_name
        self._pipeline = HF_TOKEN or None  # non-None = active

    async def load(self) -> None:
        """No local model to load — just verify the API token is present."""
        if HF_TOKEN:
            logger.info("finbert_hf_api_ready", model=self._model_name)
        else:
            logger.warning("finbert_disabled_no_hf_token")

    async def score_unscored_articles(self, batch_size: int = 20) -> int:
        """Pull unscored articles from DB, score via HF API, write results back.

        Returns the number of articles scored.
        """
        if not HF_TOKEN:
            return 0

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

        texts = [f"{a.headline} {a.body or ''}" for a in articles]
        scored = await _call_finbert_api(texts)

        async with session_factory() as session:
            for article, (score, label) in zip(articles, scored):
                article.sentiment_score = score
                article.sentiment_label = label
                article.relevance_score = compute_relevance(
                    article.headline, article.body, article.ticker
                )
                session.add(article)
            await session.commit()

        logger.info("articles_scored_via_hf_api", count=len(articles))
        return len(articles)

    async def rolling_sentiment_index(
        self, ticker: str, lookback_hours: int = 24
    ) -> float:
        """Compute Rolling Sentiment Index for a ticker.

        SI = weighted_avg(sentiment × relevance × recency_decay), range ≈ [-1, +1]
        Returns 0.0 if no scored articles found.
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
        """Detect sudden sentiment reversal (high-conviction signal)."""
        now = datetime.now(timezone.utc)
        recent_si = await self.rolling_sentiment_index(ticker, lookback_hours=window_hours)

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

        prior_si = sum(a.sentiment_score or 0.0 for a in prior_articles) / len(prior_articles)
        reversal = abs(recent_si - prior_si) > threshold
        if reversal:
            logger.info(
                "sentiment_reversal_detected",
                ticker=ticker,
                recent_si=round(recent_si, 3),
                prior_si=round(prior_si, 3),
            )
        return reversal
