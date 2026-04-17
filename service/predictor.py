import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

# model/ is importable via PYTHONPATH=/kronos set in Dockerfile
from model import KronosTokenizer, Kronos, KronosPredictor as _KronosPredictor

logger = logging.getLogger(__name__)

TOKENIZER_PATH = Path(__file__).parent / "models/finetune_tokenizer_demo/checkpoints/best_model"
PREDICTOR_PATH = Path(__file__).parent / "models/finetune_predictor_demo/checkpoints/best_model"

MAX_CONTEXT = 512
PRED_LEN = 30

_predictor: _KronosPredictor | None = None


def load_model():
    global _predictor
    logger.info("Loading Kronos model from local checkpoints...")
    tokenizer = KronosTokenizer.from_pretrained(str(TOKENIZER_PATH))
    model = Kronos.from_pretrained(str(PREDICTOR_PATH))
    _predictor = _KronosPredictor(model, tokenizer, max_context=MAX_CONTEXT)
    logger.info("Kronos model loaded on device: %s", _predictor.device)


def run_inference(klines: list[dict]) -> list[dict]:
    """Run Kronos inference on recent klines. Returns list of predicted bar dicts."""
    if _predictor is None:
        raise RuntimeError("Model not loaded")

    rows = klines[-MAX_CONTEXT:]
    if len(rows) < 60:
        raise ValueError(f"Need at least 60 klines, got {len(rows)}")

    df = pd.DataFrame(rows)[["open", "high", "low", "close", "volume"]]

    # Build timestamps from open_time (Unix ms)
    x_timestamps = pd.to_datetime([r["open_time"] for r in rows], unit="ms", utc=True)
    x_timestamp = pd.Series(x_timestamps)

    last_bar_time = pd.Timestamp(rows[-1]["open_time"], unit="ms", tz="UTC")
    y_times = [last_bar_time + timedelta(minutes=i + 1) for i in range(PRED_LEN)]
    y_timestamp = pd.Series(y_times)

    pred_df = _predictor.predict(
        df=df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=PRED_LEN,
        T=1.0,
        top_p=0.9,
        sample_count=1,
        verbose=False,
    )

    result = []
    for i, (ts, row) in enumerate(zip(y_times, pred_df.itertuples())):
        result.append({
            "bar_time": int(ts.timestamp() * 1000),
            "open": float(row.open),
            "high": float(row.high),
            "low": float(row.low),
            "close": float(row.close),
            "volume": float(row.volume) if hasattr(row, "volume") else 0.0,
        })
    return result
