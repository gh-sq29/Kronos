import asyncio
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import binance_ws
import db
import predictor
import scheduler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent

_last_predict_time: float = 0.0
RATE_LIMIT_SECONDS = 10


@asynccontextmanager
async def lifespan(app: FastAPI):
    await db.init()
    predictor.load_model()
    asyncio.create_task(binance_ws.run())
    asyncio.create_task(scheduler.run_loop())
    logger.info("Service started")
    yield


app = FastAPI(title="Kronos BTCUSDT Dashboard", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html", {})


@app.post("/api/predict")
async def predict():
    global _last_predict_time
    now = time.time()
    elapsed = now - _last_predict_time
    if elapsed < RATE_LIMIT_SECONDS:
        retry_after = int(RATE_LIMIT_SECONDS - elapsed) + 1
        raise HTTPException(
            status_code=429,
            detail=f"Rate limited. Retry after {retry_after}s.",
            headers={"Retry-After": str(retry_after)},
        )

    klines = await db.get_klines(limit=512)
    if len(klines) < 60:
        raise HTTPException(status_code=503, detail="Not enough kline data yet.")

    loop = asyncio.get_event_loop()
    try:
        preds = await loop.run_in_executor(None, predictor.run_inference, klines)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    _last_predict_time = time.time()
    t_kline = klines[-1]
    return {
        "predictions": preds,
        "t_timestamp": t_kline["open_time"],
        "t_price": t_kline["close"],
    }


@app.get("/api/klines")
async def get_klines(limit: int = 300):
    return {"klines": await db.get_klines(min(limit, 1000))}


@app.get("/api/latest-prediction")
async def latest_prediction():
    return {"predictions": await db.get_latest_prediction()}


@app.get("/api/predictions/history")
async def get_historical_prediction(timestamp: str, tz_offset: float = 8.0):
    """
    Return stored predictions for the run matching the given timestamp.

    - timestamp: datetime string in "%Y-%m-%dT%H:%M:%S" format, interpreted in tz_offset timezone
    - tz_offset: UTC offset in hours, default 8 (UTC+8). Pass 0 for UTC.
    """
    from datetime import datetime, timezone, timedelta
    try:
        dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        raise HTTPException(status_code=400, detail="timestamp must be in format %Y-%m-%dT%H:%M:%S")
    dt_utc = dt - timedelta(hours=tz_offset)
    utc_ms = int(dt_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)
    preds = await db.get_predictions_by_run_time(utc_ms)
    return {"predictions": preds}


@app.get("/api/stats")
async def get_stats(threshold: float = 0.5, act_threshold: float = 0.1):
    now_ms = int(time.time() * 1000)
    base = await db.get_latest_stats()
    for w in [5, 10, 15, 20]:
        pairs = await scheduler.get_window_pairs(w, now_ms)
        breakdown = scheduler.compute_direction_breakdown(pairs, threshold, act_threshold, compute_dedup=(w == 20))
        if base[w] is not None:
            base[w]["direction"] = breakdown
        elif breakdown:
            base[w] = {"direction": breakdown}
    return {"stats": base, "threshold": threshold, "act_threshold": act_threshold}


@app.get("/api/stats/range")
async def get_stats_range(start: str, end: str, threshold: float = 0.5, act_threshold: float = 0.1, m: int = 0):
    """
    Compute stats for a custom time range.
    start/end: local datetime strings (UTC+8), e.g. "2025-01-01T12:00" or "2025-01-01T12:00:00".
    m: optional comparison interval — also compute direction accuracy against actual at T+m.
    """
    from datetime import datetime, timezone, timedelta
    try:
        start_dt = datetime.fromisoformat(start) - timedelta(hours=8)
        end_dt   = datetime.fromisoformat(end)   - timedelta(hours=8)
    except ValueError:
        raise HTTPException(status_code=400, detail="datetime must be ISO format, e.g. 2025-01-01T12:00")
    start_ms = int(start_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ms   = int(end_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
    result = {}
    for w in [5, 10, 15, 20]:
        pairs = await scheduler.get_window_pairs_for_range(w, start_ms, end_ms)
        breakdown = scheduler.compute_direction_breakdown(pairs, threshold, act_threshold, compute_dedup=(w == 20))
        breakdown_m = None
        if m > 0:
            pairs_m = await scheduler.get_window_pairs_for_range_vs_m(w, m, start_ms, end_ms)
            breakdown_m = scheduler.compute_direction_breakdown(pairs_m, threshold, act_threshold)
        if pairs:
            errors = [p["err"] for p in pairs]
            result[w] = {
                "mae": sum(errors) / len(errors),
                "max_deviation": max(errors),
                "min_deviation": min(errors),
                "sample_count": len(pairs),
                "direction": breakdown,
                "direction_m": breakdown_m,
            }
        else:
            result[w] = None
    return {"stats": result, "threshold": threshold, "act_threshold": act_threshold, "m": m}


@app.get("/api/point-history")
async def point_history(start: str, duration: int = 20, tz_offset: float = 8.0):
    """
    For each minute from `start` to `start + duration` minutes, return the stored
    prediction run at that minute and the K-line just before it as baseline.

    - start: local datetime string, e.g. "2026-04-23T12:56:00"
    - duration: how many minutes forward to query (inclusive, so duration+1 rows)
    - tz_offset: UTC offset of the supplied datetime, default 8 (UTC+8)

    Each row contains:
      - query_ms: UTC ms of the query minute
      - baseline_ms: UTC ms of the bar just before (T-1 min)
      - baseline_close: close price of that bar
      - steps: {5, 10, 15, 20} → {close, pct} predicted close and % change from baseline
    """
    from datetime import datetime, timezone, timedelta
    try:
        start_dt = datetime.fromisoformat(start) - timedelta(hours=tz_offset)
    except ValueError:
        raise HTTPException(status_code=400, detail="start must be ISO format, e.g. 2026-04-23T12:56:00")

    results = []
    for i in range(duration + 1):
        query_dt_utc = start_dt + timedelta(minutes=i)
        query_ms = int(query_dt_utc.replace(tzinfo=timezone.utc).timestamp() * 1000)
        baseline_ms = query_ms - 60_000

        baseline_list = await db.get_klines_in_range(baseline_ms, baseline_ms)
        baseline = baseline_list[0] if baseline_list else None

        preds = await db.get_predictions_by_run_time(query_ms)

        steps = {}
        for step in [5, 10, 15, 20]:
            if len(preds) >= step:
                pred_close = preds[step - 1]["close"]
                pct = (pred_close - baseline["close"]) / baseline["close"] * 100 if baseline else None
                steps[step] = {
                    "close": round(pred_close, 2),
                    "pct": round(pct, 3) if pct is not None else None,
                }

        results.append({
            "query_ms": query_ms,
            "baseline_ms": baseline_ms,
            "baseline_close": round(baseline["close"], 2) if baseline else None,
            "steps": steps,
        })

    return {"results": results}


@app.get("/api/price")
async def get_price():
    return {"price": binance_ws.get_latest_price()}


@app.get("/health")
async def health():
    return {"status": "ok"}
