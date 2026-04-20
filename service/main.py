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
        breakdown = scheduler.compute_direction_breakdown(pairs, threshold, act_threshold)
        if base[w] is not None:
            base[w]["direction"] = breakdown
        elif breakdown:
            base[w] = {"direction": breakdown}
    return {"stats": base, "threshold": threshold, "act_threshold": act_threshold}


@app.get("/api/price")
async def get_price():
    return {"price": binance_ws.get_latest_price()}


@app.get("/health")
async def health():
    return {"status": "ok"}
