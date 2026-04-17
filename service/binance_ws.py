import asyncio
import json
import logging
import time

import websockets

import db

logger = logging.getLogger(__name__)

WS_URL = "wss://fstream.binance.com/ws/btcusdt@kline_1m"

_latest_price: float = 0.0


def get_latest_price() -> float:
    return _latest_price


async def run():
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
