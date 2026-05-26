#!/usr/bin/env python
"""
Parallel evaluation worker.
Each process owns one GPU and a shard of (dataset, term) work items.
"""
import os
import csv
import json
import logging
import warnings
import argparse
import traceback
from typing import Optional

import numpy as np
import torch
from dotenv import load_dotenv

from gluonts.itertools import batcher
from gluonts.model.forecast import QuantileForecast
from gluonts.model import evaluate_model
from gluonts.time_feature import get_seasonality
from gluonts.ev.metrics import (
    MSE, MAE, MASE, MAPE, SMAPE, MSIS, RMSE, NRMSE, ND,
    MeanWeightedSumQuantileLoss,
)

from gift_eval.data import Dataset
from uni2ts.model.moiraic import MoiraicForecast, MoiraicModule
from uni2ts.model.moiraie import MoiraieForecast, MoiraieModule
from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

load_dotenv()
logging.getLogger("gluonts.model.forecast").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# CLI / ENV
# ============================================================
parser = argparse.ArgumentParser()
parser.add_argument("--rank", type=int, default=None,
                    help="Worker rank. Defaults to $SLURM_PROCID.")
parser.add_argument("--world-size", type=int, default=None,
                    help="Total number of workers. Defaults to $SLURM_NTASKS.")
parser.add_argument("--local-rank", type=int, default=None,
                    help="Local rank for GPU pinning. Defaults to $SLURM_LOCALID.")
parser.add_argument("--model-name", type=str, default="moiraie_base")
parser.add_argument("--checkpoint-type", type=str, default="hf", choices=["hf", "ckpt"])
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--context-length", type=int, default=4000)
parser.add_argument("--batch-size", type=int, default=128)
args = parser.parse_args()

RANK       = args.rank        if args.rank        is not None else int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = args.world_size  if args.world_size  is not None else int(os.environ.get("SLURM_NTASKS", 1))
LOCAL_RANK = args.local_rank  if args.local_rank  is not None else int(os.environ.get("SLURM_LOCALID", RANK))

MODEL_NAME       = args.model_name
CHECKPOINT_TYPE  = args.checkpoint_type
MODEL_PATH       = args.model_path
OUTPUT_DIR       = args.output_dir
CONTEXT_LENGTH   = args.context_length
BATCH_SIZE       = args.batch_size
QUANTILE_LEVELS  = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)

# Pin to a single visible GPU. Using cuda:0 inside the process is cleanest
# because CUDA_VISIBLE_DEVICES restricts what the process sees.
os.environ["CUDA_VISIBLE_DEVICES"] = str(LOCAL_RANK)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(f"[rank {RANK}/{WORLD_SIZE}] local_rank={LOCAL_RANK} device={device} "
      f"visible={os.environ.get('CUDA_VISIBLE_DEVICES')}")

# ============================================================
# Dataset configuration (same as original)
# ============================================================
SHORT_DATASETS = "m4_yearly m4_quarterly m4_monthly m4_weekly m4_daily m4_hourly electricity/15T electricity/H electricity/D electricity/W solar/10T solar/H solar/D solar/W hospital covid_deaths us_births/D us_births/M us_births/W saugeenday/D saugeenday/M saugeenday/W temperature_rain_with_missing kdd_cup_2018_with_missing/H kdd_cup_2018_with_missing/D car_parts_with_missing restaurant hierarchical_sales/D hierarchical_sales/W LOOP_SEATTLE/5T LOOP_SEATTLE/H LOOP_SEATTLE/D SZ_TAXI/15T SZ_TAXI/H M_DENSE/H M_DENSE/D ett1/15T ett1/H ett1/D ett1/W ett2/15T ett2/H ett2/D ett2/W jena_weather/10T jena_weather/H jena_weather/D bitbrains_fast_storage/5T bitbrains_fast_storage/H bitbrains_rnd/5T bitbrains_rnd/H bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"
MED_LONG_DATASETS = "electricity/15T electricity/H solar/10T solar/H kdd_cup_2018_with_missing/H LOOP_SEATTLE/5T LOOP_SEATTLE/H SZ_TAXI/15T M_DENSE/H ett1/15T ett1/H ett2/15T ett2/H jena_weather/10T jena_weather/H bitbrains_fast_storage/5T bitbrains_rnd/5T bizitobs_application bizitobs_service bizitobs_l2c/5T bizitobs_l2c/H"

PRETTY_NAMES = {
    "saugeenday": "saugeen",
    "temperature_rain_with_missing": "temperature_rain",
    "kdd_cup_2018_with_missing": "kdd_cup_2018",
    "car_parts_with_missing": "car_parts",
}

# ============================================================
# Build the global, deterministic work list (identical on all ranks)
# ============================================================
short_set = set(SHORT_DATASETS.split())
medlong_set = set(MED_LONG_DATASETS.split())
all_datasets = sorted(short_set | medlong_set)  # sorted for determinism

work_items = []  # list of (ds_name, term)
for ds_name in all_datasets:
    for term in ["short", "medium", "long"]:
        if term in ("medium", "long") and ds_name not in medlong_set:
            continue
        work_items.append((ds_name, term))

# Round-robin shard. The sorted() above + RANK stride gives reasonable
# load balancing assuming dataset cost has no strong correlation to alpha order.
my_items = [w for i, w in enumerate(work_items) if i % WORLD_SIZE == RANK]
print(f"[rank {RANK}] assigned {len(my_items)}/{len(work_items)} items: {my_items}")

# ============================================================
# Model loading (once per worker)
# ============================================================
def load_module(model_path: str, checkpoint_type: str, ModuleCls):
    if checkpoint_type == "hf":
        return ModuleCls.from_pretrained(model_path)
    elif checkpoint_type == "ckpt":
        return ModuleCls.load_from_checkpoint(model_path)
    raise ValueError(checkpoint_type)

model_type = ("moiraic" if "moiraic" in MODEL_NAME
              else "moiraie" if "moiraie" in MODEL_NAME
              else "moirai")
ModuleCls   = {"moiraic": MoiraicModule,   "moiraie": MoiraieModule,   "moirai": Moirai2Module}[model_type]
ForecastCls = {"moiraic": MoiraicForecast, "moiraie": MoiraieForecast, "moirai": Moirai2Forecast}[model_type]
module = load_module(MODEL_PATH, CHECKPOINT_TYPE, ModuleCls)
print(f"[rank {RANK}] loaded module from {MODEL_PATH}")

# ============================================================
# Predictor (unchanged from original)
# ============================================================
class MoiraiQuantilePredictor:
    def __init__(self, module, prediction_length, context_length=4000,
                 target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
                 device=torch.device("cpu"), batch_size=512,
                 quantile_levels=QUANTILE_LEVELS):
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.device = device
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels
        self.model = ForecastCls(
            module=module,
            prediction_length=prediction_length,
            context_length=context_length,
            target_dim=target_dim,
            feat_dynamic_real_dim=feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
            ar_method="trajectory"
        ).to(self.device)

    def predict(self, test_data_input):
        bs = self.batch_size
        while True:
            try:
                fq = []
                for batch in batcher(test_data_input, batch_size=bs):
                    fq.append(self.model.predict([e["target"] for e in batch]))
                fq = np.concatenate(fq)
                break
            except torch.cuda.OutOfMemoryError:
                bs //= 2
                if bs < 1:
                    raise
                print(f"[rank {RANK}] OOM — reducing batch_size to {bs}")
                torch.cuda.empty_cache()

        out = []
        for item, ts in zip(fq, test_data_input):
            out.append(QuantileForecast(
                item_id=ts["item_id"],
                forecast_arrays=item,
                start_date=ts["start"] + len(ts["target"]),
                forecast_keys=list(map(str, self.quantile_levels)),
            ))
        return out

# ============================================================
# Metrics
# ============================================================
metrics = [
    MSE(forecast_type="mean"), MSE(forecast_type=0.5), MAE(), MASE(),
    MAPE(), SMAPE(), MSIS(), RMSE(), NRMSE(), ND(),
    MeanWeightedSumQuantileLoss(quantile_levels=list(QUANTILE_LEVELS)),
]
METRIC_COLUMNS = [
    "eval_metrics/MSE[mean]", "eval_metrics/MSE[0.5]", "eval_metrics/MAE[0.5]",
    "eval_metrics/MASE[0.5]", "eval_metrics/MAPE[0.5]", "eval_metrics/sMAPE[0.5]",
    "eval_metrics/MSIS", "eval_metrics/RMSE[mean]", "eval_metrics/NRMSE[mean]",
    "eval_metrics/ND[0.5]", "eval_metrics/mean_weighted_sum_quantile_loss",
]
METRIC_KEYS = [
    "MSE[mean]", "MSE[0.5]", "MAE[0.5]", "MASE[0.5]", "MAPE[0.5]",
    "sMAPE[0.5]", "MSIS", "RMSE[mean]", "NRMSE[mean]", "ND[0.5]",
    "mean_weighted_sum_quantile_loss",
]

dataset_properties_map = json.load(open(
    "/srv/disk00/ctadler/gift-eval/notebooks/dataset_properties.json"))

def resolve_ds_config(ds_name, term):
    if "/" in ds_name:
        ds_key, ds_freq = ds_name.split("/", 1)
        ds_key = PRETTY_NAMES.get(ds_key.lower(), ds_key.lower())
    else:
        ds_key = PRETTY_NAMES.get(ds_name.lower(), ds_name.lower())
        ds_freq = dataset_properties_map[ds_key]["frequency"]
    return ds_key, ds_freq, f"{ds_key}/{ds_freq}/{term}"

# ============================================================
# Run my shard
# ============================================================
os.makedirs(OUTPUT_DIR, exist_ok=True)
csv_path = os.path.join(OUTPUT_DIR, f"all_results_rank{RANK}.csv")
with open(csv_path, "w", newline="") as f:
    csv.writer(f).writerow(["dataset", "model"] + METRIC_COLUMNS + ["domain", "num_variates"])

for ds_name, term in my_items:
    try:
        ds_key, ds_freq, ds_config = resolve_ds_config(ds_name, term)
        print(f"\n[rank {RANK}] {'='*50}\n[rank {RANK}] Evaluating: {ds_config}")

        to_univariate = Dataset(name=ds_name, term=term, to_univariate=False).target_dim != 1
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)

        predictor = MoiraiQuantilePredictor(
            module=module,
            prediction_length=dataset.prediction_length,
            context_length=CONTEXT_LENGTH,
            target_dim=1,
            past_feat_dynamic_real_dim=dataset.past_feat_dynamic_real_dim,
            device=device,
            batch_size=BATCH_SIZE,
            quantile_levels=QUANTILE_LEVELS,
        )

        season_length = get_seasonality(dataset.freq)
        res = evaluate_model(
            predictor,
            test_data=dataset.test_data,
            metrics=metrics,
            batch_size=BATCH_SIZE,
            axis=None,
            mask_invalid_label=True,
            allow_nan_forecast=False,
            seasonality=season_length,
        )

        row = [ds_config, MODEL_NAME] + [res[k][0] for k in METRIC_KEYS] + [
            dataset_properties_map[ds_key]["domain"],
            dataset_properties_map[ds_key]["num_variates"],
        ]
        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        print(f"[rank {RANK}] ✓ {ds_config}")

        # Free predictor / cuda memory between datasets
        del predictor
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"[rank {RANK}] ✗ FAILED on ({ds_name}, {term}): {e}")
        traceback.print_exc()
        # Continue with the rest of the shard rather than aborting all work.

print(f"[rank {RANK}] DONE → {csv_path}")