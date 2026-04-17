#!/bin/bash
cd "$(dirname "$0")"

python eval_csv_model.py \
  --tokenizer_path /home/qings/Kronos/finetune/outputs/models/finetune_tokenizer_demo/checkpoints/best_model \
  --model_path /home/qings/Kronos/finetune/outputs/models/finetune_predictor_demo/checkpoints/best_model \
  --data_path ./data/processed_datasets_btc \
  --lookback_window 512 \
  --predict_window 48 \
  --device cuda:0 \
  --batch_size 32 \
  --sample_count 1 \
  --max_samples 100 \
  --output eval_results.csv
