import asyncio
import logging
import time
from datetime import datetime, timezone

import db
import predictor

logger = logging.getLogger(__name__)

WINDOWS = [5, 10, 15, 20]


async def compute_stats(window: int, now_ms: int):
    """Compare past predictions against actual klines for a given window."""
    # Only consider runs old enough that `window` actual bars have closed
    cutoff_ms = now_ms - window * 60 * 1000
    runs = await db.get_prediction_runs_since(now_ms - 2 * 60 * 60 * 1000)  # last 2h
    runs = [r for r in runs if _run_id_to_ms(r) <= cutoff_ms]

    if not runs:
        return

    all_errors = []
    all_dir_correct = []

    for run_id in runs:
        preds = await db.get_predictions_for_run(run_id)
        if len(preds) < window:
            continue
        pred = preds[window - 1]  # single point: T+window
        actual_list = await db.get_klines_in_range(pred["bar_time"], pred["bar_time"])
        actual = actual_list[0] if actual_list else None
        if actual is None:
            continue

        err = abs(pred["close"] - actual["close"])
        all_errors.append(err)

        # Direction vs start of run (T+0)
        run_start_ms = _run_id_to_ms(run_id)
        prev_actuals = await db.get_klines_in_range(run_start_ms - 2 * 60 * 1000, run_start_ms)
        prev_close = prev_actuals[-1]["close"] if prev_actuals else None
        if prev_close is not None:
            pred_dir = pred["close"] - prev_close
            act_dir = actual["close"] - prev_close
            if pred_dir != 0 and act_dir != 0:
                all_dir_correct.append(1 if (pred_dir > 0) == (act_dir > 0) else 0)

    if not all_errors:
        return

    mae = sum(all_errors) / len(all_errors)
    max_dev = max(all_errors)
    min_dev = min(all_errors)
    dir_acc = sum(all_dir_correct) / len(all_dir_correct) if all_dir_correct else 0.0

    await db.insert_stats(now_ms, window, mae, max_dev, min_dev, dir_acc, len(all_errors))
    logger.debug("Stats window=%d mae=%.4f dir_acc=%.2f n=%d", window, mae, dir_acc, len(all_errors))


def _run_id_to_ms(run_id: str) -> int:
    try:
        dt = datetime.fromisoformat(run_id)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp() * 1000)
    except Exception:
        return 0


async def scheduled_run():
    logger.info("Scheduler: running inference")
    try:
        klines = await db.get_klines(limit=512)
        if len(klines) < 60:
            logger.info("Scheduler: not enough klines (%d), skipping", len(klines))
            return

        loop = asyncio.get_event_loop()
        preds = await loop.run_in_executor(None, predictor.run_inference, klines)

        run_id = datetime.now(timezone.utc).isoformat()
        await db.insert_predictions(run_id, preds)
        logger.info("Scheduler: stored %d predictions run_id=%s", len(preds), run_id)

        now_ms = int(time.time() * 1000)
        for w in WINDOWS:
            await compute_stats(w, now_ms)

    except Exception:
        logger.exception("Scheduler: error during run")


async def run_loop():
    while True:
        await asyncio.sleep(60)
        await scheduled_run()
