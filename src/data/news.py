"""News feed integration: NewsAPI + Benzinga polling.

Polls every 5 minutes during market hours.
Filters articles by ticker mentions from the trading universe.
Stores raw articles; FinBERT sentiment scoring happens in Phase 3.
"""

from __future__ import annotations

import asyncio
import logging
import structlog
import re
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx
from sqlalchemy.dialects.postgresql import insert

from src.config import get_settings
from src.data.db import NewsRaw, get_session_factory

logger = structlog.get_logger(__name__)

NEWSAPI_BASE = "https://newsapi.org/v2"
BENZINGA_BASE = "https://api.benzinga.com/api/v2"

POLL_INTERVAL_SECONDS = 300   # 5 minutes


# ─── Ticker mention detector ──────────────────────────────────────────────────

def _extract_mentioned_tickers(text: str, universe: set[str]) -> list[str]:
    """Return tickers from universe that appear as whole words in text."""
    words = set(re.findall(r"\b[A-Z]{1,5}\b", text))
    return sorted(universe & words)


# ─── NewsAPI client ───────────────────────────────────────────────────────────

class NewsAPIClient:
    """Wrapper around newsapi.org REST API."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def fetch_recent(
        self, query: str = "stock market earnings", hours_back: int = 1
    ) -> list[dict[str, Any]]:
        if not self._settings.news_api_key:
            return []

        from_dt = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).isoformat()
        params = {
            "q": query,
            "from": from_dt,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 100,
            "apiKey": self._settings.news_api_key,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            resp = await client.get(f"{NEWSAPI_BASE}/everything", params=params)
            resp.raise_for_status()
            data = resp.json()
            return data.get("articles", [])

    def parse_article(self, raw: dict[str, Any], universe: set[str]) -> list[dict[str, Any]]:
        """Parse one NewsAPI article into one row per mentioned ticker."""
        headline = raw.get("title", "")
        body = raw.get("description", "") or raw.get("content", "")
        full_text = f"{headline} {body}"
        tickers = _extract_mentioned_tickers(full_text, universe)
        if not tickers:
            return []

        published_raw = raw.get("publishedAt", "")
        try:
            published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published_at = datetime.now(timezone.utc)

        return [
            {
                "published_at": published_at,
                "ticker": ticker,
                "headline": headline,
                "body": body,
                "source": raw.get("source", {}).get("name", "newsapi"),
                "url": raw.get("url", ""),
                "sentiment_score": None,
                "sentiment_label": None,
                "relevance_score": None,
                "raw": raw,
            }
            for ticker in tickers
        ]


# ─── Benzinga client ──────────────────────────────────────────────────────────

class BenzingaClient:
    """Wrapper around Benzinga news REST API."""

    def __init__(self) -> None:
        self._settings = get_settings()

    async def fetch_recent(self, tickers: list[str], hours_back: int = 1) -> list[dict[str, Any]]:
        if not self._settings.benzinga_api_key:
            return []

        # Benzinga supports per-ticker queries
        results: list[dict[str, Any]] = []
        from_dt = (datetime.now(timezone.utc) - timedelta(hours=hours_back)).strftime(
            "%Y-%m-%dT%H:%M:%S"
        )
        async with httpx.AsyncClient(timeout=20) as client:
            # Batch in groups of 10
            for i in range(0, len(tickers), 10):
                batch = tickers[i : i + 10]
                params = {
                    "token": self._settings.benzinga_api_key,
                    "tickers": ",".join(batch),
                    "dateFrom": from_dt,
                    "displayOutput": "full",
                    "pageSize": 50,
                }
                try:
                    resp = await client.get(f"{BENZINGA_BASE}/news", params=params)
                    resp.raise_for_status()
                    data = resp.json()
                    results.extend(data if isinstance(data, list) else data.get("news", []))
                except Exception as exc:
                    logger.warning("benzinga_fetch_error", tickers=batch, error=str(exc))

        return results

    def parse_article(self, raw: dict[str, Any], universe: set[str]) -> list[dict[str, Any]]:
        """Parse one Benzinga article into ticker-tagged rows."""
        headline = raw.get("title", raw.get("headline", ""))
        body = raw.get("body", raw.get("teaser", ""))
        full_text = f"{headline} {body}"

        # Benzinga includes explicit ticker tags
        explicit_tickers = [
            s.get("name", "").upper()
            for s in raw.get("stocks", [])
            if s.get("name", "").upper() in universe
        ]
        mention_tickers = _extract_mentioned_tickers(full_text, universe)
        tickers = list(set(explicit_tickers + mention_tickers))

        if not tickers:
            return []

        published_raw = raw.get("created", raw.get("date", ""))
        try:
            published_at = datetime.fromisoformat(published_raw.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            published_at = datetime.now(timezone.utc)

        return [
            {
                "published_at": published_at,
                "ticker": ticker,
                "headline": headline,
                "body": body,
                "source": "benzinga",
                "url": raw.get("url", ""),
                "sentiment_score": None,
                "sentiment_label": None,
                "relevance_score": None,
                "raw": raw,
            }
            for ticker in tickers
        ]


# ─── DB writer ────────────────────────────────────────────────────────────────

async def _write_articles(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    session_factory = get_session_factory()
    async with session_factory() as session:
        stmt = insert(NewsRaw).values(rows).on_conflict_do_nothing()
        await session.execute(stmt)
        await session.commit()
    logger.info("news_articles_written", count=len(rows))


# ─── Poller ───────────────────────────────────────────────────────────────────

class NewsPoller:
    """Polls NewsAPI and Benzinga every POLL_INTERVAL_SECONDS.

    Args:
        universe: Set of ticker symbols to watch.
    """

    def __init__(self, universe: set[str]) -> None:
        self.universe = universe
        self._newsapi = NewsAPIClient()
        self._benzinga = BenzingaClient()
        self._running = False

    async def start(self) -> None:
        self._running = True
        logger.info("news_poller_started", universe_size=len(self.universe))
        while self._running:
            try:
                await self._poll_once()
            except Exception as exc:
                logger.error("news_poll_error", error=str(exc))
            await asyncio.sleep(POLL_INTERVAL_SECONDS)

    async def stop(self) -> None:
        self._running = False

    async def _poll_once(self) -> None:
        all_rows: list[dict[str, Any]] = []

        # NewsAPI — broad financial query
        try:
            articles = await self._newsapi.fetch_recent(hours_back=1)
            for art in articles:
                all_rows.extend(self._newsapi.parse_article(art, self.universe))
        except Exception as exc:
            logger.warning("newsapi_error", error=str(exc))

        # Benzinga — per-ticker
        try:
            tickers = sorted(self.universe)
            raw_news = await self._benzinga.fetch_recent(tickers, hours_back=1)
            for art in raw_news:
                all_rows.extend(self._benzinga.parse_article(art, self.universe))
        except Exception as exc:
            logger.warning("benzinga_error", error=str(exc))

        if all_rows:
            await _write_articles(all_rows)

    async def update_universe(self, new_universe: set[str]) -> None:
        """Hot-swap the ticker universe without restarting the poller."""
        self.universe = new_universe
        logger.info("news_poller_universe_updated", size=len(new_universe))
