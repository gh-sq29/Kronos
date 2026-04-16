# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/test_kronos_regression.py

# Run a single test
pytest tests/test_kronos_regression.py::TestKronosRegression::test_name

# Run web UI
cd webui && python run.py

# Finetune tokenizer (Qlib pipeline, multi-GPU)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py

# Finetune predictor (Qlib pipeline, multi-GPU)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py

# Finetune via CSV pipeline (single-GPU or CPU)
python finetune_csv/train_sequential.py --config configs/config_ali09988_candle-5min.yaml
```

## Architecture

Kronos is a two-stage foundation model for financial K-line (OHLCV candlestick) time series forecasting.

### Stage 1 — KronosTokenizer (`model/kronos.py`)
Converts continuous OHLCV data into discrete tokens via Binary Spherical Quantization (BSQuantizer in `model/module.py`). Architecture: `Embed → Encoder (Transformer blocks) → BSQuantizer → Decoder (Transformer blocks) → Head`. The quantizer maps latent vectors onto a binary hypersphere, producing hierarchical tokens (`s1_bits` coarse + `s2_bits` fine).

### Stage 2 — Kronos Predictor (`model/kronos.py`)
An autoregressive Transformer decoder that operates on the discrete tokens produced by the tokenizer. `KronosPredictor` wraps both stages: it handles normalization/denormalization, context windowing, and autoregressive sampling (temperature, nucleus sampling, `sample_count` for ensemble).

### Public API (`model/__init__.py`)
```python
from model import KronosTokenizer, Kronos, KronosPredictor
```
Models load from Hugging Face Hub via `PyTorchModelHubMixin` (e.g., `KronosPredictor.from_pretrained("Qings/Kronos-mini")`).

### Model Zoo
| Model | Params | Context |
|---|---|---|
| Kronos-mini | 4.1M | 2048 |
| Kronos-small | 24.7M | 512 |
| Kronos-base | 102.3M | 512 |

### Directory Map
| Directory | Purpose |
|---|---|
| `model/` | Core inference classes (tokenizer, predictor, supporting modules) |
| `finetune/` | Qlib-based finetuning + backtesting pipeline for A-share markets |
| `finetune_csv/` | YAML-configured CSV-based finetuning pipeline (no Qlib dependency) |
| `webui/` | Flask + Plotly web UI for interactive prediction and visualization |
| `examples/` | Reference scripts for prediction, batch inference, and backtesting |
| `tests/` | Regression tests with fixed model revisions and reference outputs |

## Data Format

Input DataFrames must contain at minimum: `open`, `high`, `low`, `close` (plus optional `volume`, `amount`). A datetime index or timestamp column is required for windowing. Values should be raw prices — `KronosPredictor` handles normalization internally.

## Finetuning Pipelines

**Qlib pipeline** (`finetune/`): Requires Qlib data installation. Configure paths and date ranges in `finetune/config.py`. Run `qlib_data_preprocess.py` first to prepare data, then train in sequence: tokenizer → predictor. Use `finetune/qlib_test.py` for backtesting.

**CSV pipeline** (`finetune_csv/`): Provide a YAML config (see `finetune_csv/configs/` for examples). `train_sequential.py` runs both stages automatically. Results and checkpoints are saved alongside the config.
