import glob
import os
import pickle

import pandas as pd
from tqdm import tqdm


class BinanceConfig:
    """Configuration for the Binance BTC/USDT preprocessing pipeline."""

    symbol = "BTCUSDT"
    interval = "1m"
    raw_data_dir = "./data/binance_raw"
    dataset_path = "./data/processed_datasets_btc"

    lookback_window = 90   # Past time steps used as model input.
    predict_window = 10    # Future time steps to predict.

    feature_list = ['open', 'high', 'low', 'close', 'vol', 'amt']

    # Per-year split boundaries (month numbers, inclusive).
    # Each year: Jan–Aug → train, Sep–Oct → valid, Nov–Dec → test.
    train_months = range(1, 9)   # 1–8
    val_months   = range(9, 11)  # 9–10
    test_months  = range(11, 13) # 11–12


# Binance kline CSV column positions (no header row in the file).
_COL_OPEN_TIME  = 0
_COL_OPEN       = 1
_COL_HIGH       = 2
_COL_LOW        = 3
_COL_CLOSE      = 4
_COL_VOLUME     = 5   # Base asset volume (BTC)
_COL_QUOTE_VOL  = 7   # Quote asset volume (USDT) — used as `amt`


class BinanceDataPreprocessor:
    """
    Loads raw Binance daily CSV files, processes them into the feature format
    expected by finetune/dataset.py, and splits them into train/val/test pickle files.

    Output pickle structure (identical to QlibDataPreprocessor output):
        dict[symbol -> DataFrame]
        DataFrame: DatetimeIndex, columns = ['open', 'high', 'low', 'close', 'vol', 'amt']
    """

    def __init__(self):
        self.config = BinanceConfig()
        self.data: dict[str, pd.DataFrame] = {}

    def load_data(self):
        """Reads all daily CSV files, concatenates them, and stores in self.data."""
        csv_dir = os.path.join(
            self.config.raw_data_dir, self.config.symbol, self.config.interval
        )
        csv_files = sorted(
            glob.glob(os.path.join(csv_dir, f"{self.config.symbol}-{self.config.interval}-*.csv"))
        )
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in {csv_dir}. "
                "Run binance_data_fetcher.py first."
            )

        print(f"Loading {len(csv_files)} daily CSV files from {csv_dir} ...")
        frames = []
        for path in tqdm(csv_files, desc="Reading CSVs"):
            df = pd.read_csv(path, header=None)
            frames.append(df)

        raw = pd.concat(frames, ignore_index=True)

        # Some files have a header row or a trailing summary row with non-numeric
        # values. Convert to numeric first and drop any rows that fail to parse.
        numeric_ts = pd.to_numeric(raw[_COL_OPEN_TIME], errors='coerce')

        if numeric_ts.notna().any():
            # Standard Binance format: Unix millisecond (or microsecond) integers.
            raw[_COL_OPEN_TIME] = numeric_ts
            raw = raw.dropna(subset=[_COL_OPEN_TIME])

            # Normalize timestamps to milliseconds per-row.
            # Binance changed some files to microseconds (16-digit) from milliseconds (13-digit).
            # Threshold: pandas datetime64[ns] max is ~year 2262, i.e. ~9.2e12 ms.
            # Any value above that must be in microseconds → divide by 1000.
            _MS_MAX = 9_999_999_999_999
            mask_us = raw[_COL_OPEN_TIME] > _MS_MAX
            raw.loc[mask_us, _COL_OPEN_TIME] = raw.loc[mask_us, _COL_OPEN_TIME] // 1000
            raw = raw[raw[_COL_OPEN_TIME] <= _MS_MAX]

            ts = pd.to_datetime(
                raw[_COL_OPEN_TIME].astype('int64'), unit='ms', utc=True
            ).dt.tz_convert(None)
        else:
            # Alternate format: datetime strings (e.g. "2020-01-01 00:00:00").
            parsed = pd.to_datetime(raw[_COL_OPEN_TIME], errors='coerce', utc=True)
            valid = parsed.notna()
            raw = raw[valid]
            ts = parsed[valid].dt.tz_convert(None)

        ts.name = 'datetime'

        # Build the required feature columns.
        symbol_df = pd.DataFrame(index=ts.values)
        symbol_df.index.name = 'datetime'
        symbol_df['open']  = raw[_COL_OPEN].to_numpy(dtype=float)
        symbol_df['high']  = raw[_COL_HIGH].to_numpy(dtype=float)
        symbol_df['low']   = raw[_COL_LOW].to_numpy(dtype=float)
        symbol_df['close'] = raw[_COL_CLOSE].to_numpy(dtype=float)
        symbol_df['vol']   = raw[_COL_VOLUME].to_numpy(dtype=float)
        symbol_df['amt']   = raw[_COL_QUOTE_VOL].to_numpy(dtype=float)

        symbol_df = symbol_df.sort_index()
        symbol_df = symbol_df.dropna()

        min_len = self.config.lookback_window + self.config.predict_window + 1
        if len(symbol_df) < min_len:
            raise ValueError(
                f"Insufficient data: {len(symbol_df)} rows, need at least {min_len}."
            )

        self.data[self.config.symbol] = symbol_df
        print(f"Loaded {len(symbol_df):,} rows for {self.config.symbol}.")

    def prepare_dataset(self):
        """Splits data per-year into train/val/test and saves as pickle files.

        Each calendar year is split by month:
          train: Jan–Aug (months 1–8)
          val:   Sep–Oct (months 9–10)
          test:  Nov–Dec (months 11–12)
        All years are then concatenated into the respective splits.
        """
        if not self.data:
            raise RuntimeError("No data loaded. Call load_data() first.")

        print("Splitting into train / val / test (per-year) ...")
        train_data, val_data, test_data = {}, {}, {}

        for symbol, df in tqdm(self.data.items(), desc="Preparing datasets"):
            years = sorted(df.index.year.unique())
            print(f"  {symbol}: {years[0]}–{years[-1]}, {len(df):,} rows total")

            train_frames, val_frames, test_frames = [], [], []
            for year in years:
                year_df = df[df.index.year == year]
                month = year_df.index.month
                train_frames.append(year_df[month.isin(self.config.train_months)])
                val_frames.append(year_df[month.isin(self.config.val_months)])
                test_frames.append(year_df[month.isin(self.config.test_months)])

            train_data[symbol] = pd.concat(train_frames)
            val_data[symbol]   = pd.concat(val_frames)
            test_data[symbol]  = pd.concat(test_frames)

        os.makedirs(self.config.dataset_path, exist_ok=True)
        for name, split in [("train", train_data), ("val", val_data), ("test", test_data)]:
            path = os.path.join(self.config.dataset_path, f"{name}_data.pkl")
            with open(path, 'wb') as f:
                pickle.dump(split, f)
            rows = sum(len(v) for v in split.values())
            print(f"  {name}: {rows:,} rows → {path}")

        print("Datasets prepared and saved successfully.")


if __name__ == "__main__":
    preprocessor = BinanceDataPreprocessor()
    preprocessor.load_data()
    preprocessor.prepare_dataset()
