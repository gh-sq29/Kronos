import asyncio
import json
import logging
import time
import urllib.request

import websockets

import db

logger = logging.getLogger(__name__)

WS_URL = "wss://fstream.binance.com/ws/btcusdt@kline_1m"
REST_URL = "https://fapi.binance.com/fapi/v1/klines"

SYMBOL = "BTCUSDT"
BATCH_SIZE = 500        # Binance allows up to 1500; 500 = weight 2 (safe)
TARGET_BARS = 1440      # 24h of 1-min bars
MIN_BARS = 512          # minimum needed for inference

_latest_price: float = 0.0


def get_latest_price() -> float:
    return _latest_price


def _fetch_batch_sync(end_time_ms: int, limit: int) -> list:
    url = (f"{REST_URL}?symbol={SYMBOL}&interval=1m"
           f"&limit={limit}&endTime={end_time_ms}")
    with urllib.request.urlopen(url, timeout=15) as resp:
        return json.loads(resp.read())


async def _fetch_batch(end_time_ms: int, limit: int) -> list:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _fetch_batch_sync, end_time_ms, limit)


async def prefetch_if_needed():
    """Fetch up to 24h of historical klines if DB has fewer than MIN_BARS."""
    existing = await db.get_klines(limit=MIN_BARS)
    if len(existing) >= MIN_BARS:
        logger.info("DB has %d klines, skipping prefetch", len(existing))
        return

    logger.info("DB has %d klines (need %d) — prefetching up to %dh of history",
                len(existing), MIN_BARS, TARGET_BARS // 60)

    end_time_ms = int(time.time() * 1000)
    total_stored = 0
    batches_done = 0
    max_batches = (TARGET_BARS + BATCH_SIZE - 1) // BATCH_SIZE  # ceil(1440/500) = 3

    while batches_done < max_batches:
        remaining = TARGET_BARS - total_stored
        limit = min(BATCH_SIZE, remaining)
        try:
            rows = await _fetch_batch(end_time_ms, limit)
        except Exception as e:
            logger.warning("Prefetch batch %d failed: %s", batches_done + 1, e)
            break

        if not rows:
            break

        for r in rows:
            await db.upsert_kline(
                open_time=int(r[0]),
                o=float(r[1]),
                h=float(r[2]),
                l=float(r[3]),
                c=float(r[4]),
                v=float(r[5]),
            )

        total_stored += len(rows)
        batches_done += 1
        logger.info("Prefetch batch %d: stored %d bars (total %d)",
                    batches_done, len(rows), total_stored)

        # Move end_time back to just before the oldest bar in this batch
        end_time_ms = int(rows[0][0]) - 1

        if total_stored >= TARGET_BARS or len(rows) < limit:
            break

        # Small delay between batches to stay well under rate limits
        await asyncio.sleep(0.3)

    logger.info("Prefetch complete: %d bars stored across %d batches",
                total_stored, batches_done)


async def run():
    await prefetch_if_needed()

    backoff = 1
    while True:
        try:
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=10) as ws:
                logger.info("Binance WS connected")
                backoff = 1
                async for raw in ws:
                    msg = json.loads(raw)
                    k = msg.get("k", {})
                    global _latest_price
                    _latest_price = float(k.get("c", _latest_price))
                    if k.get("x"):  # candle closed
                        await db.upsert_kline(
                            open_time=int(k["t"]),
                            o=float(k["o"]),
                            h=float(k["h"]),
                            l=float(k["l"]),
                            c=float(k["c"]),
                            v=float(k["v"]),
                        )
        except Exception as e:
            logger.warning("Binance WS error: %s — reconnecting in %ds", e, backoff)
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 30)
