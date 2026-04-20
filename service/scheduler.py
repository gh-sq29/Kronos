import asyncio
import logging
import time
from datetime import datetime, timezone

import db
import predictor

logger = logging.getLogger(__name__)

WINDOWS = [5, 10, 15, 20]


async def get_window_pairs(window: int, now_ms: int) -> list[dict]:
    """Return (pred_close, actual_close, prev_close) pairs for a given window."""
    cutoff_ms = now_ms - window * 60 * 1000
    runs = await db.get_prediction_runs_since(now_ms - 2 * 60 * 60 * 1000)
    runs = [r for r in runs if _run_id_to_ms(r) <= cutoff_ms]

    pairs = []
    for run_id in runs:
        preds = await db.get_predictions_for_run(run_id)
        if len(preds) < window:
            continue
        pred = preds[window - 1]
        actual_list = await db.get_klines_in_range(pred["bar_time"], pred["bar_time"])
        actual = actual_list[0] if actual_list else None
        if actual is None:
            continue
        run_start_ms = _run_id_to_ms(run_id)
        prev_actuals = await db.get_klines_in_range(run_start_ms - 2 * 60 * 1000, run_start_ms)
        prev_close = prev_actuals[-1]["close"] if prev_actuals else None
        if not prev_close:
            continue
        pairs.append({"pred_close": pred["close"], "actual_close": actual["close"], "prev_close": prev_close, "err": abs(pred["close"] - actual["close"])})
    return pairs


async def get_window_pairs_for_range(window: int, start_ms: int, end_ms: int) -> list[dict]:
    """Like get_window_pairs but queries runs within [start_ms, end_ms]."""
    cutoff_ms = end_ms - window * 60 * 1000
    runs = await db.get_prediction_runs_in_range(start_ms, end_ms)
    runs = [r for r in runs if _run_id_to_ms(r) <= cutoff_ms]

    pairs = []
    for run_id in runs:
        preds = await db.get_predictions_for_run(run_id)
        if len(preds) < window:
            continue
        pred = preds[window - 1]
        actual_list = await db.get_klines_in_range(pred["bar_time"], pred["bar_time"])
        actual = actual_list[0] if actual_list else None
        if actual is None:
            continue
        run_start_ms = _run_id_to_ms(run_id)
        prev_actuals = await db.get_klines_in_range(run_start_ms - 2 * 60 * 1000, run_start_ms)
        prev_close = prev_actuals[-1]["close"] if prev_actuals else None
        if not prev_close:
            continue
        pairs.append({"pred_close": pred["close"], "actual_close": actual["close"],
                      "prev_close": prev_close, "err": abs(pred["close"] - actual["close"])})
    return pairs


def compute_direction_breakdown(pairs: list[dict], pred_threshold_pct: float, act_threshold_pct: float) -> dict | None:
    """Compute detailed long/flat_long/flat_short/short breakdown from prediction-actual pairs."""
    if not pairs:
        return None

    def classify(pct_change, thr):
        if pct_change > thr:
            return "long"
        if pct_change < -thr:
            return "short"
        return "flat_long" if pct_change >= 0 else "flat_short"

    CATS = ["long", "flat_long", "flat_short", "short"]
    pred_counts = {c: 0 for c in CATS}
    act_counts  = {c: 0 for c in CATS}
    outcomes: dict[tuple, int] = {}

    for p in pairs:
        prev = p["prev_close"]
        pd = classify((p["pred_close"] - prev) / prev * 100, pred_threshold_pct)
        ad = classify((p["actual_close"] - prev) / prev * 100, act_threshold_pct)
        pred_counts[pd] += 1
        act_counts[ad] += 1
        outcomes[(pd, ad)] = outcomes.get((pd, ad), 0) + 1

    n = len(pairs)
    pl, ps = pred_counts["long"], pred_counts["short"]

    def r(v, total):
        return round(v / total * 100, 1) if total else 0.0

    def o(pd, ad):
        return r(outcomes.get((pd, ad), 0), pred_counts[pd])

    ll_count = outcomes.get(("long", "long"), 0)
    ss_count = outcomes.get(("short", "short"), 0)
    directional = pl + ps

    return {
        "dir_acc": round((ll_count + ss_count) / directional * 100, 1) if directional else None,
        "pred_long":       r(pl, n),
        "pred_flat_long":  r(pred_counts["flat_long"], n),
        "pred_flat_short": r(pred_counts["flat_short"], n),
        "pred_short":      r(ps, n),
        "act_long":        r(act_counts["long"], n),
        "act_flat_long":   r(act_counts["flat_long"], n),
        "act_flat_short":  r(act_counts["flat_short"], n),
        "act_short":       r(act_counts["short"], n),
        "ll":  o("long",  "long"),
        "lfl": o("long",  "flat_long"),
        "lfs": o("long",  "flat_short"),
        "ls":  o("long",  "short"),
        "ss":  o("short", "short"),
        "sfs": o("short", "flat_short"),
        "sfl": o("short", "flat_long"),
        "sl":  o("short", "long"),
    }


async def compute_stats(window: int, now_ms: int):
    """Compare past predictions against actual klines for a given window."""
    pairs = await get_window_pairs(window, now_ms)
    if not pairs:
        return

    errors = [p["err"] for p in pairs]
    mae = sum(errors) / len(errors)
    max_dev = max(errors)
    min_dev = min(errors)

    # Legacy direction accuracy (no threshold)
    dir_correct = []
    for p in pairs:
        pd = p["pred_close"] - p["prev_close"]
        ad = p["actual_close"] - p["prev_close"]
        if pd != 0 and ad != 0:
            dir_correct.append(1 if (pd > 0) == (ad > 0) else 0)
    dir_acc = sum(dir_correct) / len(dir_correct) if dir_correct else 0.0

    await db.insert_stats(now_ms, window, mae, max_dev, min_dev, dir_acc, len(errors))
    logger.debug("Stats window=%d mae=%.4f dir_acc=%.2f n=%d", window, mae, dir_acc, len(errors))


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
