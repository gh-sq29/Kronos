import os
import sys
import argparse
import pickle

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.append('../')
from config import Config
from model.kronos import Kronos, KronosTokenizer, auto_regressive_inference


def load_models(config: dict):
    device = torch.device(config['device'])
    print(f"Loading models onto device: {device}...")
    tokenizer = KronosTokenizer.from_pretrained(config['tokenizer_path']).to(device).eval()
    model = Kronos.from_pretrained(config['model_path']).to(device).eval()
    return tokenizer, model


class EvalDataset(Dataset):
    """Sliding-window dataset over test_data.pkl, returns normalized x and ground truth."""

    def __init__(self, data: dict, config: Config):
        self.config = config
        self.feature_list = config.feature_list
        self.time_feature_list = config.time_feature_list
        self.window_size = config.lookback_window + config.predict_window
        self.indices = []

        print("Building dataset indices...")
        self.data = {}
        for symbol, df in data.items():
            df = df.reset_index()
            df['minute'] = df['datetime'].dt.minute
            df['hour'] = df['datetime'].dt.hour
            df['weekday'] = df['datetime'].dt.weekday
            df['day'] = df['datetime'].dt.day
            df['month'] = df['datetime'].dt.month
            self.data[symbol] = df

            num_samples = len(df) - self.window_size + 1
            for i in range(max(0, num_samples)):
                timestamp = df.iloc[i + config.lookback_window - 1]['datetime']
                self.indices.append((symbol, i, timestamp))

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        symbol, start_idx, timestamp = self.indices[idx]
        df = self.data[symbol]

        context_end = start_idx + self.config.lookback_window
        predict_end = context_end + self.config.predict_window

        context_df = df.iloc[start_idx:context_end]
        predict_df = df.iloc[context_end:predict_end]

        x = context_df[self.feature_list].values.astype(np.float32)
        x_stamp = context_df[self.time_feature_list].values.astype(np.float32)
        y_stamp = predict_df[self.time_feature_list].values.astype(np.float32)

        x_mean = np.mean(x, axis=0)
        x_std = np.std(x, axis=0)
        x = (x - x_mean) / (x_std + 1e-5)
        x = np.clip(x, -self.config.clip, self.config.clip)

        # Normalize ground truth with same params as context
        y_actual = predict_df[self.feature_list].values.astype(np.float32)
        y_norm = (y_actual - x_mean) / (x_std + 1e-5)
        y_norm = np.clip(y_norm, -self.config.clip, self.config.clip)

        return (
            torch.from_numpy(x),
            torch.from_numpy(x_stamp),
            torch.from_numpy(y_stamp),
            symbol,
            timestamp,
            torch.from_numpy(y_norm),
            torch.from_numpy(x_mean),   # per-feature mean of context
            torch.from_numpy(x_std),    # per-feature std of context
        )


def collate_fn_with_gt(batch):
    x, x_stamp, y_stamp, symbols, timestamps, y_gt, x_mean, x_std = zip(*batch)
    return (
        torch.stack(x),
        torch.stack(x_stamp),
        torch.stack(y_stamp),
        list(symbols),
        list(timestamps),
        torch.stack(y_gt),
        torch.stack(x_mean),
        torch.stack(x_std),
    )


def main():
    parser = argparse.ArgumentParser(description='Evaluate CSV-finetuned Kronos model on test_data.pkl')
    parser.add_argument('--tokenizer_path', required=True, help='Path to finetuned tokenizer checkpoint')
    parser.add_argument('--model_path', required=True, help='Path to finetuned predictor checkpoint')
    parser.add_argument('--data_path', default='./data/processed_datasets_btc', help='Directory containing test_data.pkl')
    parser.add_argument('--lookback_window', type=int, default=512)
    parser.add_argument('--predict_window', type=int, default=48)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--sample_count', type=int, default=1)
    parser.add_argument('--output', default='eval_results.csv', help='Path to save per-sample predictions')
    parser.add_argument('--max_samples', type=int, default=0, help='Limit samples for timing test (0 = all)')
    args = parser.parse_args()

    tokenizer, model = load_models({
        'device': args.device,
        'tokenizer_path': args.tokenizer_path,
        'model_path': args.model_path,
    })
    device = torch.device(args.device)

    test_data_path = os.path.join(args.data_path, 'test_data.pkl')
    print(f"Loading test data from {test_data_path}...")
    with open(test_data_path, 'rb') as f:
        test_data = pickle.load(f)

    config = Config()
    config.lookback_window = args.lookback_window
    config.predict_window = args.predict_window

    dataset = EvalDataset(data=test_data, config=config)
    if args.max_samples > 0:
        dataset.indices = dataset.indices[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for timing test")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() // 2),
        collate_fn=collate_fn_with_gt,
    )

    all_pred_close = []
    all_true_close = []
    records = []

    import time
    t0 = time.time()
    print(f"Running inference on {len(dataset)} samples...")
    with torch.no_grad():
        for x, x_stamp, y_stamp, symbols, timestamps, y_gt, x_mean, x_std in tqdm(loader):
            preds = auto_regressive_inference(
                tokenizer, model,
                x.to(device), x_stamp.to(device), y_stamp.to(device),
                max_context=config.max_context,
                pred_len=args.predict_window,
                clip=config.clip,
                T=0.6, top_k=0, top_p=0.9,
                sample_count=args.sample_count,
            )
            # preds: (batch, lookback + pred_len, 6) — take last pred_len steps
            pred_window = preds[:, -args.predict_window:, :]  # (batch, pred_len, 6)
            y_gt_np = y_gt.numpy()                            # (batch, pred_len, 6)
            mean_np = x_mean.numpy()                          # (batch, 6)
            std_np = x_std.numpy()                            # (batch, 6)

            # close is at feature index 3
            pred_close_norm = pred_window[:, :, 3]   # (batch, pred_len)
            true_close_norm = y_gt_np[:, :, 3]

            # denormalize: value * (std + 1e-5) + mean
            close_mean = mean_np[:, 3:4]             # (batch, 1)
            close_std  = std_np[:, 3:4]
            pred_close = pred_close_norm * (close_std + 1e-5) + close_mean
            true_close = true_close_norm * (close_std + 1e-5) + close_mean

            all_pred_close.append(pred_close)
            all_true_close.append(true_close)

            for i, (sym, ts) in enumerate(zip(symbols, timestamps)):
                records.append({
                    'symbol': sym,
                    'timestamp': ts,
                    'pred_close_t15': float(pred_close[i, 14]),
                    'true_close_t15': float(true_close[i, 14]),
                    'pred_close_t20': float(pred_close[i, 19]),
                    'true_close_t20': float(true_close[i, 19]),
                    'pred_close_last': float(pred_close[i, -1]),
                    'true_close_last': float(true_close[i, -1]),
                    'pred_close_mean': float(pred_close[i].mean()),
                    'true_close_mean': float(true_close[i].mean()),
                })

    all_pred = np.concatenate(all_pred_close, axis=0).flatten()
    all_true = np.concatenate(all_true_close, axis=0).flatten()

    mse = float(np.mean((all_pred - all_true) ** 2))
    mae = float(np.mean(np.abs(all_pred - all_true)))

    def step_metrics(pred_key, true_key):
        p = np.array([r[pred_key] for r in records])
        t = np.array([r[true_key] for r in records])
        mse_ = float(np.mean((p - t) ** 2))
        mae_ = float(np.mean(np.abs(p - t)))
        dir_ = float(np.mean(np.sign(p - t) == np.sign(t - t)))  # direction vs context mean=0
        return mse_, mae_

    mse_t15, mae_t15 = step_metrics('pred_close_t15', 'true_close_t15')
    mse_t20, mae_t20 = step_metrics('pred_close_t20', 'true_close_t20')
    mse_last, mae_last = step_metrics('pred_close_last', 'true_close_last')

    elapsed = time.time() - t0
    total_samples = len(dataset)
    print(f"\nInference time: {elapsed:.1f}s for {total_samples} samples "
          f"({elapsed/total_samples*1000:.1f} ms/sample)")

    print(f"\n{'='*45}")
    print(f"Evaluation Results (actual price space)")
    print(f"{'='*45}")
    print(f"  Samples : {len(records)}")
    print(f"  {'':10s}  {'MSE':>12s}  {'MAE':>12s}")
    print(f"  {'T+15':10s}  {mse_t15:>12.4f}  {mae_t15:>12.4f}")
    print(f"  {'T+20':10s}  {mse_t20:>12.4f}  {mae_t20:>12.4f}")
    print(f"  {'T+48(last)':10s}  {mse_last:>12.4f}  {mae_last:>12.4f}")
    print(f"  {'all steps':10s}  {mse:>12.4f}  {mae:>12.4f}")
    print(f"{'='*45}")

    results_df = pd.DataFrame(records)
    results_df.to_csv(args.output, index=False)
    print(f"Per-sample predictions saved to {args.output}")


if __name__ == '__main__':
    main()
