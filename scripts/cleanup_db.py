"""Cleanup old DB rows to free disk space.

Keeps only the most recent N days of data in each large table.
Run this when the Railway Postgres volume is near capacity.

Usage:
    python scripts/cleanup_db.py --days 14
    python scripts/cleanup_db.py --days 14 --dry-run   # preview only
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta, timezone

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine


async def cleanup(db_url: str, keep_days: int, dry_run: bool) -> None:
    engine = create_async_engine(db_url, echo=False)
    cutoff = datetime.now(timezone.utc) - timedelta(days=keep_days)
    cutoff_str = cutoff.isoformat()

    tables = [
        ("feature_matrix", "time"),    # biggest: JSONB per ticker per 1m bar
        ("ohlcv_1m", "time"),          # raw 1m bars
        ("ohlcv_5m", "time"),          # raw 5m bars
        ("signals", "time"),           # model outputs
        ("news_raw", "published_at"),  # news articles
        ("prediction_outcomes", "predicted_at"),  # IC tracking
    ]

    async with engine.begin() as conn:
        # First: show current table sizes
        print("=== Current table sizes ===")
        result = await conn.execute(text("""
            SELECT relname AS table,
                   pg_size_pretty(pg_total_relation_size(C.oid)) AS total_size,
                   pg_total_relation_size(C.oid) AS size_bytes
            FROM pg_class C
            LEFT JOIN pg_namespace N ON (N.oid = C.relnamespace)
            WHERE nspname = 'public'
              AND C.relkind = 'r'
            ORDER BY pg_total_relation_size(C.oid) DESC
            LIMIT 15;
        """))
        for row in result:
            print(f"  {row[0]:30s} {row[1]:>12s}")

        print(f"\n=== Cleanup: deleting rows older than {keep_days} days (before {cutoff_str[:10]}) ===")

        for table, time_col in tables:
            # Count rows to delete
            count_result = await conn.execute(text(
                f"SELECT COUNT(*) FROM {table} WHERE {time_col} < :cutoff"
            ), {"cutoff": cutoff})
            n_delete = count_result.scalar() or 0

            count_total = await conn.execute(text(f"SELECT COUNT(*) FROM {table}"))
            n_total = count_total.scalar() or 0

            print(f"\n  {table}: {n_delete:,} of {n_total:,} rows to delete")

            if dry_run:
                print(f"    [DRY RUN] would delete {n_delete:,} rows")
                continue

            if n_delete == 0:
                print(f"    nothing to delete")
                continue

            # Delete in batches to avoid long locks
            batch_size = 50_000
            deleted_total = 0
            while deleted_total < n_delete:
                result = await conn.execute(text(f"""
                    DELETE FROM {table}
                    WHERE ctid IN (
                        SELECT ctid FROM {table}
                        WHERE {time_col} < :cutoff
                        LIMIT :batch
                    )
                """), {"cutoff": cutoff, "batch": batch_size})
                batch_deleted = result.rowcount
                deleted_total += batch_deleted
                print(f"    deleted {deleted_total:,} / {n_delete:,}...")
                if batch_deleted < batch_size:
                    break

            print(f"    done — {deleted_total:,} rows deleted")

        if not dry_run:
            # VACUUM to actually reclaim disk space
            print("\n=== Running VACUUM to reclaim disk space ===")
            # Need autocommit for VACUUM
            await conn.execute(text("COMMIT"))

    # VACUUM requires a separate connection outside a transaction
    if not dry_run:
        raw_engine = create_async_engine(db_url, echo=False, isolation_level="AUTOCOMMIT")
        async with raw_engine.connect() as conn:
            for table, _ in tables:
                print(f"  VACUUM {table}...")
                await conn.execute(text(f"VACUUM {table}"))
            print("  VACUUM complete")
        await raw_engine.dispose()

        # Show sizes after cleanup
        async with engine.begin() as conn:
            print("\n=== Table sizes after cleanup ===")
            result = await conn.execute(text("""
                SELECT relname AS table,
                       pg_size_pretty(pg_total_relation_size(C.oid)) AS total_size
                FROM pg_class C
                LEFT JOIN pg_namespace N ON (N.oid = C.relnamespace)
                WHERE nspname = 'public'
                  AND C.relkind = 'r'
                ORDER BY pg_total_relation_size(C.oid) DESC
                LIMIT 15;
            """))
            for row in result:
                print(f"  {row[0]:30s} {row[1]:>12s}")

    await engine.dispose()
    print("\nDone!")


async def close_orphan_crypto_trades(db_url: str, dry_run: bool) -> None:
    """Close crypto trades with NULL exit_time.

    Crypto was removed from the trading universe on 2026-03-27.
    These orphan positions were never closed. Mark them closed at entry price
    (0 PnL) with exit_reason = 'universe_removed'.
    """
    engine = create_async_engine(db_url, echo=False)
    async with engine.begin() as conn:
        # Preview
        result = await conn.execute(text("""
            SELECT id, ticker, side, entry_time, entry_price, shares
            FROM trades
            WHERE exit_time IS NULL
              AND ticker LIKE '%/%'
            ORDER BY entry_time
        """))
        rows = result.fetchall()
        if not rows:
            print("No orphan crypto trades found.")
            await engine.dispose()
            return

        print(f"Found {len(rows)} orphan crypto trades:")
        for r in rows:
            print(f"  id={r[0]} {r[1]} {r[2]} @ {r[4]:.2f} x{r[5]} entered {r[3]}")

        if dry_run:
            print("\n[DRY RUN] Would close all of the above with PnL=0, reason='universe_removed'")
            await engine.dispose()
            return

        # Close them
        result = await conn.execute(text("""
            UPDATE trades
            SET exit_time = entry_time + interval '1 hour',
                exit_price = entry_price,
                pnl = 0.0,
                pnl_pct = 0.0,
                exit_reason = 'universe_removed'
            WHERE exit_time IS NULL
              AND ticker LIKE '%/%'
        """))
        print(f"\nClosed {result.rowcount} orphan crypto trades (PnL=0, reason=universe_removed)")

    await engine.dispose()


def main():
    parser = argparse.ArgumentParser(description="Cleanup old DB rows")
    parser.add_argument("--days", type=int, default=14,
                        help="Keep only the most recent N days (default: 14)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without actually deleting")
    parser.add_argument("--db-url", type=str, default=None,
                        help="Database URL (default: reads from .env / src.config)")
    parser.add_argument("--close-crypto", action="store_true",
                        help="Close orphan crypto trades (NULL exit_time) instead of general cleanup")
    args = parser.parse_args()

    if args.db_url:
        db_url = args.db_url
    else:
        import os
        db_url = os.environ.get("DATABASE_URL")
        if not db_url:
            try:
                sys.path.insert(0, ".")
                from src.config import get_settings
                db_url = get_settings().database_url
            except Exception:
                print("ERROR: Pass --db-url or set DATABASE_URL env var")
                sys.exit(1)

    print(f"Database: {db_url.split('@')[1] if '@' in db_url else db_url}")
    print(f"Mode: {'DRY RUN' if args.dry_run else 'LIVE'}\n")

    if args.close_crypto:
        asyncio.run(close_orphan_crypto_trades(db_url, args.dry_run))
    else:
        print(f"Keeping: last {args.days} days\n")
        asyncio.run(cleanup(db_url, args.days, args.dry_run))


if __name__ == "__main__":
    main()
