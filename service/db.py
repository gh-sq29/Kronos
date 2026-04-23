import time
import aiosqlite
from pathlib import Path

DB_PATH = Path(__file__).parent / "kronos.db"


async def init():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executescript("""
            CREATE TABLE IF NOT EXISTS klines (
                open_time  INTEGER PRIMARY KEY,
                open  REAL, high REAL, low REAL, close REAL, volume REAL
            );
            CREATE TABLE IF NOT EXISTS predictions (
                run_id    TEXT,
                bar_time  INTEGER,
                open REAL, high REAL, low REAL, close REAL, volume REAL,
                PRIMARY KEY (run_id, bar_time)
            );
            CREATE TABLE IF NOT EXISTS stats (
                computed_at      INTEGER,
                window_minutes   INTEGER,
                mae              REAL,
                max_deviation    REAL,
                min_deviation    REAL,
                direction_accuracy REAL,
                sample_count     INTEGER,
                PRIMARY KEY (computed_at, window_minutes)
            );
        """)
        await db.commit()


async def upsert_kline(open_time: int, o: float, h: float, l: float, c: float, v: float):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO klines VALUES (?,?,?,?,?,?)",
            (open_time, o, h, l, c, v),
        )
        await db.commit()


async def get_klines(limit: int = 600) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM klines ORDER BY open_time DESC LIMIT ?", (limit,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in reversed(rows)]


async def insert_predictions(run_id: str, rows: list[dict]):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.executemany(
            "INSERT OR REPLACE INTO predictions VALUES (?,?,?,?,?,?,?)",
            [(run_id, r["bar_time"], r["open"], r["high"], r["low"], r["close"], r["volume"]) for r in rows],
        )
        await db.commit()


async def get_latest_prediction() -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT run_id FROM predictions ORDER BY bar_time DESC LIMIT 1"
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return []
        run_id = row["run_id"]
        async with db.execute(
            "SELECT * FROM predictions WHERE run_id=? ORDER BY bar_time", (run_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def insert_stats(computed_at: int, window: int, mae: float, max_dev: float,
                       min_dev: float, dir_acc: float, n: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT OR REPLACE INTO stats VALUES (?,?,?,?,?,?,?)",
            (computed_at, window, mae, max_dev, min_dev, dir_acc, n),
        )
        await db.commit()


async def get_latest_stats() -> dict:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        result = {}
        for w in (5, 10, 15, 20):
            async with db.execute(
                "SELECT * FROM stats WHERE window_minutes=? ORDER BY computed_at DESC LIMIT 1",
                (w,),
            ) as cur:
                row = await cur.fetchone()
            result[w] = dict(row) if row else None
    return result


async def get_prediction_runs_since(since_ms: int) -> list[str]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT DISTINCT run_id FROM predictions WHERE bar_time >= ? ORDER BY run_id",
            (since_ms,),
        ) as cur:
            rows = await cur.fetchall()
    return [r["run_id"] for r in rows]


async def get_predictions_for_run(run_id: str) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM predictions WHERE run_id=? ORDER BY bar_time", (run_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_predictions_by_run_time(run_time_ms: int, tolerance_ms: int = 30_000) -> list[dict]:
    """Return predictions for the run whose run_id falls within the same minute as run_time_ms."""
    from datetime import datetime, timezone, timedelta
    base_dt = datetime.fromtimestamp(run_time_ms / 1000, tz=timezone.utc)
    # Align to the minute boundary so passing HH:MM:00 finds any run from HH:MM:00 to HH:MM:59
    lo_dt = base_dt.replace(second=0, microsecond=0)
    hi_dt = lo_dt + timedelta(seconds=59, microseconds=999999)
    # run_id is ISO format UTC string — lexicographic comparison works for same-timezone strings
    lo_str = lo_dt.isoformat()
    hi_str = hi_dt.isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT DISTINCT run_id FROM predictions WHERE run_id >= ? AND run_id <= ? ORDER BY run_id LIMIT 1",
            (lo_str, hi_str),
        ) as cur:
            row = await cur.fetchone()
        if not row:
            return []
        run_id = row["run_id"]
        async with db.execute(
            "SELECT * FROM predictions WHERE run_id=? ORDER BY bar_time", (run_id,)
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]


async def get_prediction_runs_in_range(start_ms: int, end_ms: int) -> list[str]:
    from datetime import datetime, timezone
    lo = datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc).isoformat()
    hi = datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc).isoformat()
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT DISTINCT run_id FROM predictions WHERE run_id >= ? AND run_id <= ? ORDER BY run_id",
            (lo, hi),
        ) as cur:
            rows = await cur.fetchall()
    return [r["run_id"] for r in rows]


async def get_klines_in_range(start_ms: int, end_ms: int) -> list[dict]:
    async with aiosqlite.connect(DB_PATH) as db:
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM klines WHERE open_time >= ? AND open_time <= ? ORDER BY open_time",
            (start_ms, end_ms),
        ) as cur:
            rows = await cur.fetchall()
    return [dict(r) for r in rows]
