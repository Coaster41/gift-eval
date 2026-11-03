# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

GIFT-Eval is a benchmark for general time series forecasting evaluation. This checkout is configured for evaluating the in-house `moiraic` / `moiraie` / `moirai2` models (from the sibling `uni2ts` repo) against the benchmark on a SLURM cluster.

## Setup

```shell
python3 -m venv myenv && source myenv/bin/activate
pip install -e '.[baseline]'        # baseline adds torch/lightning/statsforecast
```

The benchmark data path is read from `.env` via `GIFT_EVAL` (currently points at the uni2ts copy of the test set: `/srv/disk00/ctadler/uni2ts/datasets/giftevaltest`). Optional `NUM_CPUS` controls feature-analysis parallelism.

Each `Dataset` is loaded from `$GIFT_EVAL/<name>` as a HuggingFace dataset on disk — `name` is a config like `electricity/15T` or `m4_yearly`.

## Commands

**Run the full evaluation (SLURM, primary workflow):**
```shell
sbatch slurm/scripts/run_eval.sh
```
Edit the `MODEL_NAME`, `MODEL_PATH`, `OUTPUT_DIR`, `CONTEXT_LENGTH`, `BATCH_SIZE` block at the top of the script before submitting. The job launches one `evaluate_worker.py` process per GPU via `srun`, then merges shards with `merge_results.py`. Logs land in `slurm/logs/`.

**Run a single worker directly (debug, one GPU):**
```shell
python moiraic_eval/evaluate_worker.py \
  --model-name moiraic_base --checkpoint-type hf \
  --model-path <HF_checkpoint_dir> --output-dir results/<run> \
  --context-length 4000 --batch-size 128 --rank 0 --world-size 1
```

**Merge per-rank shards into the final csv:**
```shell
python moiraic_eval/merge_results.py --output-dir results/<run>
```

**Feature analysis (Hydra CLI):**
```shell
python -m cli.analysis datasets=all_datasets   # or datasets=sample for a quick run
```

**Format:** `black src/ cli/ moiraic_eval/` and `isort ...` (black profile, line length 88).

## Architecture

### Data loading — `src/gift_eval/data.py`
`Dataset(name, term, to_univariate=False)` is the single entry point and follows the GluonTS interface. Key derived behavior (all `cached_property`):
- **`term`** is `short`/`medium`/`long` with multipliers `1`/`10`/`15`. `prediction_length = multiplier * base_pred_len`, where base length comes from `PRED_LENGTH_MAP` (or `M4_PRED_LENGTH_MAP` for m4 datasets), keyed by normalized frequency.
- **`windows`** = number of rolling test windows, computed from `TEST_SPLIT=0.1` of the shortest series, capped at `MAX_WINDOW=20` (m4 datasets always use 1).
- Splits are produced with GluonTS `split()` at offsets relative to `prediction_length * windows`: `training_dataset`, `validation_dataset`, `test_data`.
- Multivariate series can be flattened to univariate via `to_univariate=True` (`MultivariateToUnivariate`).

### Evaluation worker — `moiraic_eval/evaluate_worker.py`
This is the custom (non-upstream) eval harness. One process owns one GPU (pinned through `CUDA_VISIBLE_DEVICES=$SLURM_LOCALID`).
- The full **work list** is the cross-product of every dataset in `SHORT_DATASETS` ∪ `MED_LONG_DATASETS` with applicable terms (`medium`/`long` only apply to `MED_LONG_DATASETS`). These dataset lists are hardcoded constants in the worker — they define the 98-row benchmark scope.
- Work is **round-robin sharded** deterministically by `i % WORLD_SIZE == RANK`, so all ranks build the identical sorted list and each takes a disjoint slice.
- `MODEL_NAME` substring selects the model family: `moiraic` → `Moiraic*`, `moiraie` → `Moiraie*`, else `moirai` → `Moirai2*` (Module + Forecast classes imported from `uni2ts.model.*`).
- `MoiraiQuantilePredictor` wraps the uni2ts `*Forecast` LightningModule and emits GluonTS `QuantileForecast`s at quantile levels `0.1…0.9`. moiraic/moiraie use `ar_method="trajectory"`.
- Metrics are computed with GluonTS `evaluate_model` (`axis=None`, `mask_invalid_label=True`, `allow_nan_forecast=False`, seasonality from `get_seasonality(freq)`).
- Each rank writes `all_results_rank{RANK}.csv`; `merge_results.py` concatenates, dedups on `(dataset, model)`, sorts, and writes `all_results.csv`.

### Result format & submission
A complete `results/<model>/all_results.csv` has **98 rows** (one per dataset config, named `f"{dataset_name}/{freq}/{term}"`) and 15 columns (dataset/model/domain/num_variates + 11 metrics). A `config.json` accompanies it for leaderboard submission (see README for required fields). `results/naive/all_results.csv` is a reference sample.

### Reference baselines — `notebooks/`
Each upstream baseline (chronos, moirai, timesfm, toto, …) has a self-contained notebook showing how to wrap a model as a GluonTS predictor and run it over the dataset splits. Use these as templates; model-specific install steps live inside the relevant notebook.

## Conventions
- Default to SLURM for any multi-GPU or long-running eval. For a quick single-GPU check, pick an unused GPU from `nvidia-smi`.
- The `moiraic_eval/` directory and `slurm/` scripts are local additions for this project; `src/gift_eval/`, `cli/`, and `notebooks/` are largely upstream — prefer touching the local files when changing the eval workflow.
- Dataset name prettification for output rows uses `PRETTY_NAMES` in the worker (e.g. `saugeenday` → `saugeen`, `*_with_missing` stripped).
