"""Social/tweet signal stub — placeholder for future Twitter/X integration.

Returns neutral (0.0) scores for all tickers. When the Twitter API budget
is available ($100+/mo), replace this with a real implementation that:
  1. Streams cashtag mentions from key financial influencers
  2. Filters by follower count (>1000) to reduce bot noise
  3. Scores via existing FinBERT pipeline
  4. Caches rolling 1-hour sentiment per ticker

Interface is stable — PipelineBEngine depends on get_social_score().
"""

from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


class SocialFeedStub:
    """Stub social sentiment feed — returns 0.0 for all tickers.

    Usage:
        feed = SocialFeedStub()
        score = await feed.get_social_score("AAPL")  # → 0.0
    """

    async def start(self) -> None:
        logger.info("social_feed_stub_started (returns 0.0 for all tickers)")

    async def stop(self) -> None:
        pass

    async def get_social_score(self, ticker: str) -> float:
        """Return social sentiment score [-1, +1] for a ticker.

        Stub always returns 0.0 (neutral).
        """
        return 0.0

    async def get_universe_scores(self, tickers: list[str]) -> dict[str, float]:
        """Return social scores for all tickers in the universe."""
        return {t: 0.0 for t in tickers}
