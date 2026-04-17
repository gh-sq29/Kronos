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
    return templates.TemplateResponse("index.html", {"request": request})


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
    return {"predictions": preds}


@app.get("/api/klines")
async def get_klines(limit: int = 300):
    return {"klines": await db.get_klines(min(limit, 1000))}


@app.get("/api/latest-prediction")
async def latest_prediction():
    return {"predictions": await db.get_latest_prediction()}


@app.get("/api/stats")
async def get_stats():
    return {"stats": await db.get_latest_stats()}


@app.get("/api/price")
async def get_price():
    return {"price": binance_ws.get_latest_price()}
