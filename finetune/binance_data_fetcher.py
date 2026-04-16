import datetime
import io
import os
import zipfile

import requests
from tqdm import tqdm

BINANCE_DATA_BASE = "https://data.binance.vision/data/spot/daily/klines/BTCUSDT"


class BinanceDataFetcher:
    """
    Downloads daily 1-min K-line zip files from Binance Vision and extracts
    them as CSVs to a local directory. Resume-safe: skips already-downloaded dates.
    """

    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        start_date: str = "2020-01-01",
        end_date: str = "2025-06-05",
        save_dir: str = "./data/binance_raw",
    ):
        self.symbol = symbol
        self.interval = interval
        self.start_date = datetime.date.fromisoformat(start_date)
        self.end_date = datetime.date.fromisoformat(end_date)
        self.save_dir = os.path.join(save_dir, symbol, interval)
        os.makedirs(self.save_dir, exist_ok=True)

    def fetch_all(self):
        """Downloads all missing daily files for the configured date range."""
        today = datetime.date.today()
        dates = [
            self.start_date + datetime.timedelta(days=i)
            for i in range((self.end_date - self.start_date).days + 1)
            if (self.start_date + datetime.timedelta(days=i)) < today
        ]

        success, skipped, failed = 0, 0, 0
        for date in tqdm(dates, desc=f"Fetching {self.symbol} {self.interval}"):
            csv_path = self._csv_path(date)
            if os.path.exists(csv_path):
                skipped += 1
                continue
            ok = self._fetch_one_day(date)
            if ok:
                success += 1
            else:
                failed += 1

        print(
            f"Done. Downloaded: {success}, Already existed: {skipped}, Failed: {failed}"
        )

    def _fetch_one_day(self, date: datetime.date) -> bool:
        """Downloads and extracts one day's zip file. Returns True on success."""
        filename = f"{self.symbol}-{self.interval}-{date}.zip"
        url = f"{BINANCE_DATA_BASE}/{self.interval}/{filename}"

        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 404:
                # Data not yet published for this date (common for recent days).
                return False
            resp.raise_for_status()

            with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                csv_name = f"{self.symbol}-{self.interval}-{date}.csv"
                if csv_name not in zf.namelist():
                    # Some archives use a slightly different inner filename.
                    csv_name = zf.namelist()[0]
                zf.extract(csv_name, self.save_dir)

                # Rename to a predictable path if the extracted name differs.
                extracted = os.path.join(self.save_dir, csv_name)
                target = self._csv_path(date)
                if extracted != target:
                    os.rename(extracted, target)

            return True
        except Exception as e:
            print(f"  Warning: failed to fetch {date}: {e}")
            return False

    def _csv_path(self, date: datetime.date) -> str:
        return os.path.join(
            self.save_dir, f"{self.symbol}-{self.interval}-{date}.csv"
        )


if __name__ == "__main__":
    fetcher = BinanceDataFetcher(
        symbol="BTCUSDT",
        interval="1m",
        start_date="2020-01-01",
        end_date="2025-06-05",
        save_dir="./data/binance_raw",
    )
    fetcher.fetch_all()
