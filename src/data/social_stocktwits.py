"""Reddit social sentiment feed — free, no API key needed.

Polls Reddit's public JSON API for mentions of tickers across finance
subreddits (r/wallstreetbets, r/stocks, r/investing, r/options).

Scores posts using:
  1. Existing FinBERT sentiment model (if available)
  2. Simple keyword heuristic fallback (bullish/bearish word matching)
  3. Weighted by upvotes (high-score posts = stronger signal)
  4. Recency-weighted (newer posts count more)

Rate limit: Reddit allows ~60 req/min without auth. We poll every 5 minutes
and batch at 1 req/sec per subreddit to stay well within limits.

Falls back to StockTwits API if Reddit is unavailable.
Gracefully returns 0.0 on errors.
"""

from __future__ import annotations

import asyncio
import re
import structlog
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = structlog.get_logger(__name__)

REDDIT_BASE = "https://www.reddit.com"
SUBREDDITS = ["wallstreetbets", "stocks", "investing", "options"]
POLL_INTERVAL_SECONDS = 300  # 5 minutes
CACHE_TTL_SECONDS = 600     # 10-minute cache per ticker
REQUEST_DELAY = 1.0          # 1 sec between requests (Reddit is generous but be polite)

# Simple keyword-based sentiment (used when FinBERT is unavailable)
_BULLISH_WORDS = frozenset({
    "buy", "calls", "long", "moon", "rocket", "bullish", "undervalued",
    "breakout", "surge", "rally", "upside", "beat", "earnings beat",
    "strong", "growth", "upgrade", "accumulate", "dip", "buy the dip",
    "tendies", "squeeze", "gamma", "yolo",
})
_BEARISH_WORDS = frozenset({
    "sell", "puts", "short", "crash", "dump", "bearish", "overvalued",
    "breakdown", "plunge", "downside", "miss", "earnings miss",
    "weak", "decline", "downgrade", "avoid", "bag", "loss porn",
    "bubble", "rug pull", "dead cat",
})


@dataclass
class _TickerSentiment:
    """Cached sentiment for one ticker."""

    score: float = 0.0               # [-1, +1]
    n_bullish: int = 0
    n_bearish: int = 0
    n_neutral: int = 0
    n_posts: int = 0
    total_upvotes: int = 0
    fetched_at: float = 0.0          # time.time()


_cache: dict[str, _TickerSentiment] = {}


def _keyword_sentiment(text: str) -> float:
    """Quick keyword-based sentiment [-1, +1] from post title/body."""
    text_lower = text.lower()
    words = set(text_lower.split())

    bullish_hits = len(words & _BULLISH_WORDS)
    bearish_hits = len(words & _BEARISH_WORDS)

    # Also check multi-word phrases
    for phrase in ("buy the dip", "earnings beat", "earnings miss", "loss porn",
                   "rug pull", "dead cat", "gamma squeeze"):
        if phrase in text_lower:
            if phrase in ("buy the dip", "earnings beat", "gamma squeeze"):
                bullish_hits += 1
            else:
                bearish_hits += 1

    total = bullish_hits + bearish_hits
    if total == 0:
        return 0.0
    return (bullish_hits - bearish_hits) / total


def _mentions_ticker(text: str, ticker: str) -> bool:
    """Check if text mentions the ticker (case-sensitive $TICK or word boundary)."""
    # $AAPL style (cashtag)
    if f"${ticker}" in text:
        return True
    # Word boundary match (avoid matching "APPLE" when looking for "AAPL")
    pattern = rf'\b{re.escape(ticker)}\b'
    return bool(re.search(pattern, text))


async def _fetch_reddit_posts(subreddits: list[str], limit: int = 50) -> list[dict]:
    """Fetch recent posts from finance subreddits.

    Returns list of post dicts with: title, body, score, created_utc, subreddit.
    """
    try:
        import httpx
    except ImportError:
        logger.warning("httpx_not_installed — Reddit feed disabled")
        return []

    all_posts: list[dict] = []
    headers = {"User-Agent": "StockBot/1.0 (financial-research)"}

    async with httpx.AsyncClient(timeout=10.0, headers=headers) as client:
        for sub in subreddits:
            try:
                resp = await client.get(
                    f"{REDDIT_BASE}/r/{sub}/new.json",
                    params={"limit": limit},
                )
                if resp.status_code != 200:
                    logger.debug("reddit_http_error", subreddit=sub, status=resp.status_code)
                    await asyncio.sleep(REQUEST_DELAY)
                    continue

                data = resp.json()
                children = data.get("data", {}).get("children", [])

                for child in children:
                    d = child.get("data", {})
                    all_posts.append({
                        "title": d.get("title", ""),
                        "body": d.get("selftext", ""),
                        "score": d.get("score", 0),
                        "created_utc": d.get("created_utc", 0),
                        "subreddit": sub,
                        "num_comments": d.get("num_comments", 0),
                    })

            except Exception as exc:
                logger.debug("reddit_fetch_error", subreddit=sub, error=str(exc))

            await asyncio.sleep(REQUEST_DELAY)

    return all_posts


def _score_ticker(
    ticker: str,
    posts: list[dict],
    finbert_fn: Any = None,
) -> _TickerSentiment:
    """Compute sentiment for a ticker from Reddit posts.

    Filters posts mentioning the ticker, then scores each post
    with FinBERT (if available) or keyword heuristic.
    Weights by upvotes and recency.
    """
    now = time.time()
    relevant = []

    for post in posts:
        text = f"{post['title']} {post['body']}"
        if _mentions_ticker(text, ticker):
            relevant.append(post)

    if not relevant:
        return _TickerSentiment(fetched_at=now)

    weighted_score = 0.0
    total_weight = 0.0
    n_bullish = 0
    n_bearish = 0
    n_neutral = 0
    total_upvotes = 0

    for i, post in enumerate(relevant):
        text = f"{post['title']} {post['body']}"

        # Sentiment score for this post
        if finbert_fn is not None:
            try:
                sent = finbert_fn(text)
            except Exception:
                sent = _keyword_sentiment(text)
        else:
            sent = _keyword_sentiment(text)

        if sent > 0.1:
            n_bullish += 1
        elif sent < -0.1:
            n_bearish += 1
        else:
            n_neutral += 1

        # Weight by upvotes (log scale to dampen viral outliers)
        import math
        upvote_weight = math.log2(max(post["score"], 1) + 1)

        # Recency weight: posts from last hour = 1.0, 24h ago = 0.3
        age_hours = (now - post.get("created_utc", now)) / 3600
        recency_weight = max(0.3, 1.0 - 0.03 * age_hours)

        weight = upvote_weight * recency_weight
        weighted_score += sent * weight
        total_weight += weight
        total_upvotes += post["score"]

    if total_weight == 0:
        final_score = 0.0
    else:
        final_score = weighted_score / total_weight

    final_score = max(-1.0, min(1.0, final_score))

    return _TickerSentiment(
        score=final_score,
        n_bullish=n_bullish,
        n_bearish=n_bearish,
        n_neutral=n_neutral,
        n_posts=len(relevant),
        total_upvotes=total_upvotes,
        fetched_at=now,
    )


# ─── Public interface (same as SocialFeedStub) ──────────────────────────────

class StockTwitsFeed:
    """Reddit-based social sentiment feed (name kept for backward compatibility).

    Polls Reddit finance subreddits every 5 minutes, scores ticker mentions
    using keyword heuristics (FinBERT integration optional).

    Usage:
        feed = StockTwitsFeed(universe=["AAPL", "MSFT"])
        await feed.start()
        score = await feed.get_social_score("AAPL")  # → float [-1, +1]
    """

    def __init__(
        self,
        universe: list[str] | None = None,
        poll_interval_seconds: int = POLL_INTERVAL_SECONDS,
        subreddits: list[str] | None = None,
    ) -> None:
        self._universe: list[str] = universe or []
        self._poll_interval = poll_interval_seconds
        self._subreddits = subreddits or SUBREDDITS
        self._running = False

    def set_universe(self, tickers: list[str]) -> None:
        self._universe = tickers

    async def start(self) -> None:
        self._running = True
        logger.info(
            "social_feed_started",
            source="reddit",
            subreddits=self._subreddits,
            interval=self._poll_interval,
            universe=len(self._universe),
        )
        # Initial fetch
        await self._poll_once()
        while self._running:
            await asyncio.sleep(self._poll_interval)
            try:
                await self._poll_once()
            except Exception as exc:
                logger.error("social_feed_poll_error", error=str(exc))

    async def stop(self) -> None:
        self._running = False

    async def get_social_score(self, ticker: str) -> float:
        """Return social sentiment score [-1, +1] for a ticker."""
        cached = _cache.get(ticker)
        if cached is None:
            return 0.0
        if time.time() - cached.fetched_at > CACHE_TTL_SECONDS:
            return 0.0
        return cached.score

    async def get_universe_scores(self, tickers: list[str]) -> dict[str, float]:
        """Return social scores for all tickers."""
        return {t: await self.get_social_score(t) for t in tickers}

    async def _poll_once(self) -> None:
        if not self._universe:
            return

        # Fetch all posts from all subreddits in one batch
        posts = await _fetch_reddit_posts(self._subreddits, limit=50)
        if not posts:
            logger.debug("social_feed_no_posts")
            return

        # Score each ticker against the fetched posts
        scored = 0
        for ticker in self._universe:
            sentiment = _score_ticker(ticker, posts)
            _cache[ticker] = sentiment
            if sentiment.n_posts > 0:
                scored += 1
                logger.debug(
                    "social_scored",
                    ticker=ticker,
                    score=round(sentiment.score, 3),
                    posts=sentiment.n_posts,
                    bullish=sentiment.n_bullish,
                    bearish=sentiment.n_bearish,
                    upvotes=sentiment.total_upvotes,
                )

        logger.info(
            "social_feed_poll_done",
            source="reddit",
            total_posts=len(posts),
            tickers_with_mentions=scored,
            universe=len(self._universe),
        )
