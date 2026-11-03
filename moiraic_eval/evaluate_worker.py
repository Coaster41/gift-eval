#!/usr/bin/env python
"""
Parallel evaluation worker.
Each process owns one GPU and a shard of (dataset, term) work items.
"""
import os
import csv
import fcntl
import inspect
import time
import json
import logging
import warnings
import argparse
import importlib.util
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

from gluonts.dataset.split import split as gluonts_split
from gift_eval.data import Dataset

# Model code comes from one of two packages, whichever the active venv has:
#   uni2ts  -- /srv/disk00/ctadler/uni2ts (all model families)
#   moiraix -- /srv/disk00/ctadler/moiraix (MoiraiX only; moiraic/moiraie
#              checkpoints load through its decoder/encoder presets)
# `--backend auto` prefers uni2ts so existing jobs behave exactly as before.
_backend_parser = argparse.ArgumentParser(add_help=False)
_backend_parser.add_argument("--backend", default="auto",
                             choices=["auto", "uni2ts", "moiraix"])
BACKEND = _backend_parser.parse_known_args()[0].backend
if BACKEND == "auto":
    BACKEND = "uni2ts" if importlib.util.find_spec("uni2ts") else "moiraix"

if BACKEND == "uni2ts":
    from uni2ts.model.moiraic import MoiraicForecast, MoiraicModule
    from uni2ts.model.moiraie import MoiraieForecast, MoiraieModule
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module
    from uni2ts.model.moiraix import MoiraiXForecast, MoiraiXModule
    try:
        from uni2ts.model.moiraix_conf import MoiraiXConfForecast, MoiraiXConfModule
    except ModuleNotFoundError:
        # Only present on the feat/moiraix-conf branch/worktree; the shared venv's
        # editable uni2ts install tracks main, so this module is absent there.
        # Fine as long as --model-type moiraixconf isn't requested.
        MoiraiXConfForecast = MoiraiXConfModule = None
else:
    from moiraix.model import (
        FORECAST_DEFAULTS as MOIRAIX_FORECAST_DEFAULTS,
        MoiraiXForecast,
        MoiraiXModule,
        load_module as moiraix_load_module,
    )
    MoiraicForecast = MoiraicModule = MoiraieForecast = MoiraieModule = None
    Moirai2Forecast = Moirai2Module = None
    MoiraiXConfForecast = MoiraiXConfModule = None
try:
    from timesfm.pretrain import TimesFMXForecast, TimesFMXModule
except ModuleNotFoundError:
    # Not installed in the moiraix venv; only needed for timesfmx model names.
    TimesFMXForecast = TimesFMXModule = None

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
parser.add_argument("--schedule", type=str, default="roundrobin",
                    choices=["roundrobin", "dynamic"],
                    help="How work items are assigned to ranks. 'roundrobin' is the "
                         "historical static i%%WORLD_SIZE split -- it balances item COUNT, "
                         "not item COST, so one rank drawing several expensive configs "
                         "sets the wall clock (job 30056: rank 3 ran its last item alone "
                         "for ~7%% of the job). 'dynamic' has ranks atomically claim the "
                         "next unclaimed item from a shared flock'd counter, so the only "
                         "imbalance left is the tail item. Default stays roundrobin "
                         "because this file is shared across sessions/branches; "
                         "run_eval.sh opts in explicitly.")
parser.add_argument("--model-name", type=str, default="moiraie_base")
parser.add_argument("--checkpoint-type", type=str, default="hf", choices=["hf", "ckpt"])
parser.add_argument("--model-path", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--context-length", type=int, default=4000)
parser.add_argument("--batch-size", type=int, default=128)
parser.add_argument("--max-prediction-length", type=int, default=None,
                    help="Cap prediction length to this value and evaluate only "
                         "the shortest-term version of each dataset.")
parser.add_argument("--no-spread-head", action="store_true",
                    help="For moiraix checkpoints with num_spread_token > 0 "
                         "(which auto-enable the long-horizon quantile-spread "
                         "head at eval time): force it off, so the same "
                         "checkpoint gives both arms of the spread-head A/B "
                         "(HANDOFF_SPREAD_HEAD §7.C). Ignored otherwise.")
parser.add_argument("--no-conf", action="store_true",
                    help="For moiraixconf checkpoints: force conf_mode_override="
                         "'off' at inference time, so the same checkpoint gives "
                         "both arms of the confidence-feedback A/B "
                         "(HANDOFF_CONF_AR_FEEDBACK §12.2/§13) without touching "
                         "the checkpoint's serialized conf_mode. Ignored "
                         "otherwise.")
parser.add_argument("--backend", type=str, default="auto",
                    choices=["auto", "uni2ts", "moiraix"],
                    help="Package to load models from (resolved at import time, "
                         "see BACKEND above). 'auto' = uni2ts if installed, "
                         "else moiraix.")
args = parser.parse_args()

RANK       = args.rank        if args.rank        is not None else int(os.environ.get("SLURM_PROCID", 0))
WORLD_SIZE = args.world_size  if args.world_size  is not None else int(os.environ.get("SLURM_NTASKS", 1))
LOCAL_RANK = args.local_rank  if args.local_rank  is not None else int(os.environ.get("SLURM_LOCALID", RANK))

MODEL_NAME            = args.model_name
CHECKPOINT_TYPE       = args.checkpoint_type
MODEL_PATH            = args.model_path
OUTPUT_DIR            = args.output_dir
CONTEXT_LENGTH        = args.context_length
BATCH_SIZE            = args.batch_size
MAX_PREDICTION_LENGTH = args.max_prediction_length
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

# When a prediction length cap is set, keep one term per dataset.
# If MAX_PREDICTION_LENGTH > 48, prefer medium (if it exists) else short.
# Otherwise keep short (first in short→medium→long order).
if MAX_PREDICTION_LENGTH is not None:
    prefer_medium = MAX_PREDICTION_LENGTH > 48
    ds_order = []
    ds_to_term: dict = {}
    for ds_name, term in work_items:
        if ds_name not in ds_to_term:
            ds_order.append(ds_name)
            ds_to_term[ds_name] = term
        elif prefer_medium and term == "medium" and ds_to_term[ds_name] == "short":
            ds_to_term[ds_name] = term
    work_items = [(ds_name, ds_to_term[ds_name]) for ds_name in ds_order]

# ---- work assignment -------------------------------------------------------
# Static round-robin balances the NUMBER of items per rank, not their cost, and
# GIFT-Eval item cost spans orders of magnitude (windows x series x context).
# Measured on job 30056 (4 ranks, 55 items): counts came out 14/14/14/13 but one
# rank was still running its final item long after the other three had finished.
#
# 'dynamic' replaces the static split with a shared claim counter: every rank
# repeatedly takes the next unclaimed index under an flock, so a rank that draws
# a cheap item immediately comes back for more. Residual imbalance is bounded by
# the duration of the single longest item rather than by the luck of the split.
# All ranks run on one node (/srv/disk00 is node-local to markov-2), so an
# flock on a local file is sound here.
#  Scoped to the job id (identical across ranks, unique across runs) rather than
#  a fixed name: a fixed name would leave a stale counter in OUTPUT_DIR that
#  makes the NEXT run's ranks exit with zero work, and clearing it at startup
#  would race ranks that have already begun claiming.
CLAIM_PATH = os.path.join(
    OUTPUT_DIR, f".work_claim.{os.environ.get('SLURM_JOB_ID', 'local')}"
)


def claim_next_index() -> Optional[int]:
    """Atomically take the next unclaimed work index, or None when exhausted.

    Creation happens INSIDE the lock ("a+" creates without truncating), so there
    is no initialization race: whichever rank gets there first sees an empty file
    and treats it as 0. A rank-0-initializes design would be racy -- its write
    would sit outside the lock and could truncate a claim another rank had
    already taken.
    """
    with open(CLAIM_PATH, "a+") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        try:
            fh.seek(0)
            raw = fh.read().strip()
            idx = int(raw) if raw else 0
            if idx >= len(work_items):
                return None
            fh.seek(0)
            fh.truncate()
            fh.write(str(idx + 1))
            fh.flush()
            os.fsync(fh.fileno())
            return idx
        finally:
            fcntl.flock(fh, fcntl.LOCK_UN)


DYNAMIC = args.schedule == "dynamic"
if DYNAMIC:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    my_items = None  # claimed one at a time below
    print(f"[rank {RANK}] dynamic scheduling over {len(work_items)} items")
else:
    my_items = [w for i, w in enumerate(work_items) if i % WORLD_SIZE == RANK]
    print(f"[rank {RANK}] assigned {len(my_items)}/{len(work_items)} items: {my_items}")


def iter_my_items():
    """Yield this rank's work items under either schedule."""
    if not DYNAMIC:
        yield from my_items
        return
    while True:
        idx = claim_next_index()
        if idx is None:
            return
        yield work_items[idx]

# ============================================================
# Model loading (once per worker)
# ============================================================
def load_module(model_path: str, checkpoint_type: str, ModuleCls):
    if checkpoint_type == "hf":
        return ModuleCls.from_pretrained(model_path)
    elif checkpoint_type == "ckpt":
        return ModuleCls.load_from_checkpoint(model_path)
    raise ValueError(checkpoint_type)

MOIRAIX_PRESET = None  # moiraix backend: preset for pre-flag moiraic/moiraie checkpoints
model_type = ("chronos" if "chronos" in MODEL_NAME.lower()
              else "timesfmx" if "timesfmx" in MODEL_NAME
              else "moiraixconf" if "moiraixconf" in MODEL_NAME   # BEFORE moiraix!
              else "moiraix" if "moiraix" in MODEL_NAME
              else "moiraic" if "moiraic" in MODEL_NAME
              else "moiraie" if "moiraie" in MODEL_NAME
              else "moirai")
if model_type == "chronos":
    from chronos import BaseChronosPipeline

    module = BaseChronosPipeline.from_pretrained(MODEL_PATH, device_map=device)
    ForecastCls = None
elif BACKEND == "moiraix":
    MOIRAIX_PRESET = {"moiraix": None, "moiraic": "decoder", "moiraie": "encoder"}.get(model_type, "unsupported")
    if MOIRAIX_PRESET == "unsupported":
        raise SystemExit(f"model type {model_type!r} needs --backend uni2ts "
                         f"(the moiraix package only has MoiraiX)")
    if CHECKPOINT_TYPE != "hf":
        raise SystemExit("--backend moiraix supports --checkpoint-type hf only")
    module = moiraix_load_module(MODEL_PATH, MOIRAIX_PRESET)
    ForecastCls = MoiraiXForecast
else:
    ModuleCls   = {"timesfmx": TimesFMXModule,   "moiraixconf": MoiraiXConfModule,   "moiraix": MoiraiXModule,   "moiraic": MoiraicModule,   "moiraie": MoiraieModule,   "moirai": Moirai2Module}[model_type]
    ForecastCls = {"timesfmx": TimesFMXForecast, "moiraixconf": MoiraiXConfForecast, "moiraix": MoiraiXForecast, "moiraic": MoiraicForecast, "moiraie": MoiraieForecast, "moirai": Moirai2Forecast}[model_type]
    module = load_module(MODEL_PATH, CHECKPOINT_TYPE, ModuleCls)
print(f"[rank {RANK}] loaded module from {MODEL_PATH}")

# Decode auto-detection: next-token (predict_next) modules need an explicit AR
# decode. MoiraiXForecast defaults to ar_method=None (single-pass), which leaves
# long horizons decoded from runs of mask tokens; "branch" is the decode the
# moiraic family (same objective) has always used via MoiraicForecast's default.
# Other model types keep their own class defaults.
FORECAST_EXTRA_KWARGS = {}
if BACKEND == "moiraix" and MOIRAIX_PRESET is not None:
    # Decode defaults of the former MoiraicForecast / MoiraieForecast classes.
    FORECAST_EXTRA_KWARGS.update(MOIRAIX_FORECAST_DEFAULTS[MOIRAIX_PRESET])
if model_type == "moiraix" and getattr(module, "predict_next", False):
    # env-overridable (ar_calibration exp-000): "branch" (default), "naive",
    # or "trajectory" (only valid for predict_next=True models, which this is).
    FORECAST_EXTRA_KWARGS["ar_method"] = os.environ.get("MOIRAIX_AR_METHOD", "branch")
if model_type == "moiraix" and (nt := os.environ.get("MOIRAIX_NUM_THINK")):
    # Pause / "think" tokens (HANDOFF_PAUSE_TOKENS.md §5d). A checkpoint trained
    # with k_max_think>0 otherwise defaults to num_think=None => K=k_max_think;
    # this requests a specific K, so ONE checkpoint yields the whole K sweep,
    # K=0 included. That is the only unconfounded comparison available: two
    # separately-trained checkpoints differ by the +-2.8% seed-noise band, which
    # is larger than the effect being measured.
    # Strictly opt-in, and it degrades to a warning rather than a crash when the
    # installed uni2ts predates the feature -- this file is shared across
    # sessions and branches, so it must stay safe on main.
    k_max = getattr(module, "k_max_think", 0) or 0
    if k_max == 0:
        print(f"[rank {RANK}] MOIRAIX_NUM_THINK={nt} ignored: "
              f"checkpoint has k_max_think=0 (no pause tokens)")
    elif "num_think" not in inspect.signature(ForecastCls.__init__).parameters:
        print(f"[rank {RANK}] MOIRAIX_NUM_THINK={nt} ignored: installed "
              f"{ForecastCls.__name__} has no num_think parameter")
    else:
        FORECAST_EXTRA_KWARGS["num_think"] = int(nt)
if model_type == "moiraix" and args.no_spread_head:
    # A checkpoint with num_spread_token > 0 auto-enables the long-horizon
    # quantile-spread head at eval time; this forces it off so the same
    # checkpoint gives both arms of the spread-head A/B (HANDOFF_SPREAD_HEAD
    # §7.C) -- the only honest comparison, since two separately-trained
    # checkpoints are confounded by the ±2.8% seed-noise band.
    FORECAST_EXTRA_KWARGS["use_spread_head"] = False
if model_type == "moiraixconf":
    # conf feedback is defined for naive AR only (§9.4); the moiraix default is
    # "branch", which would raise. Default to naive and allow the off-arm A/B.
    FORECAST_EXTRA_KWARGS["ar_method"] = os.environ.get("MOIRAIX_AR_METHOD", "naive")
    if args.no_conf:                      # argparse store_true, next to --no-spread-head
        FORECAST_EXTRA_KWARGS["conf_mode_override"] = "off"
    # Inference-time conf_tau override for the tau sweep (exp-002 diagnostic 4).
    # `MoiraiXConfModule._squash` reads `self.conf_tau` at call time, so setting
    # the attribute post-load is sufficient -- no re-instantiation needed. This
    # tests whether the checkpoint's tau TARGET is right (tau = median z-width
    # puts the median AR feedback at exp(-1) = 0.37 confidence), which is a
    # different question from the tau DRIFT that exp-002's Diagnostic 2 ruled
    # out. As tau -> inf, conf -> 1 everywhere, which for conf_mode="inband" is
    # exactly the conf-off model, so the sweep traces a continuous curve between
    # the two arms of the A/B. Strictly opt-in: unset => byte-identical behavior.
    if (tau := os.environ.get("MOIRAIXCONF_TAU")):
        print(f"[rank {RANK}] conf_tau override: {module.conf_tau} -> {float(tau)}")
        module.conf_tau = float(tau)
elif model_type == "timesfmx":
    # TimesFMX is univariate; GIFT-Eval multivariate datasets are already
    # split channel-independently at the dataset level (to_univariate below),
    # so target_dim=1 is correct here. AR decode + single-pass horizon are
    # eval-time knobs (env-overridable for the PR6 inference sweep).
    FORECAST_EXTRA_KWARGS["ar_method"] = os.environ.get("TFMX_AR_METHOD", "naive")
    if (sph := os.environ.get("TFMX_SPH")):
        FORECAST_EXTRA_KWARGS["single_pass_horizon"] = int(sph)
print(f"[rank {RANK}] forecast kwargs auto-detected: {FORECAST_EXTRA_KWARGS} "
      f"(predict_next={getattr(module, 'predict_next', None)})")

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
            **FORECAST_EXTRA_KWARGS,
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

class ChronosQuantilePredictor:
    """Same protocol as MoiraiQuantilePredictor (univariate targets, context
    truncation, quantile grid) but backed by a chronos-forecasting pipeline."""

    def __init__(self, module, prediction_length, context_length=4000,
                 batch_size=512, quantile_levels=QUANTILE_LEVELS, **_ignored):
        self.pipeline = module
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.batch_size = batch_size
        self.quantile_levels = quantile_levels

    def _predict_batch(self, batch):
        inputs = [
            {"target": np.asarray(e["target"], dtype=np.float32)[-self.context_length:]}
            for e in batch
        ]
        quantiles, _ = self.pipeline.predict_quantiles(
            inputs=inputs,
            prediction_length=self.prediction_length,
            quantile_levels=list(self.quantile_levels),
        )
        if hasattr(quantiles, "cpu"):
            quantiles = quantiles.cpu().numpy()
        arr = np.asarray(quantiles, dtype=np.float32)
        # Normalize to [b, Q, H]: chronos-2 may emit a size-1 target axis
        # ([b, Q, 1, H] / [b, 1, H, Q]) and orders quantiles last ([b, H, Q]).
        n_q, horizon = len(self.quantile_levels), self.prediction_length
        if arr.ndim == 2:
            arr = arr[None]
        while arr.ndim > 3:
            drop = next((ax for ax in range(1, arr.ndim)
                         if arr.shape[ax] == 1 and arr.shape[ax] != n_q), None)
            if drop is None:
                break
            arr = np.squeeze(arr, axis=drop)
        if arr.shape[-1] == n_q and arr.shape[1] == horizon:  # [b, H, Q]
            arr = arr.transpose(0, 2, 1)
        elif arr.shape[1] != n_q and arr.shape[-1] == n_q:
            arr = np.moveaxis(arr, -1, 1)
        return arr[:, :, :horizon]

    def predict(self, test_data_input):
        bs = self.batch_size
        while True:
            try:
                fq = []
                for batch in batcher(test_data_input, batch_size=bs):
                    fq.append(self._predict_batch(batch))
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

_item_times: list[tuple[str, float]] = []
for ds_name, term in iter_my_items():
    _t0 = time.time()
    try:
        ds_key, ds_freq, ds_config = resolve_ds_config(ds_name, term)
        print(f"\n[rank {RANK}] {'='*50}\n[rank {RANK}] Evaluating: {ds_config}")

        to_univariate = Dataset(name=ds_name, term=term, to_univariate=False).target_dim != 1
        dataset = Dataset(name=ds_name, term=term, to_univariate=to_univariate)

        prediction_length = dataset.prediction_length
        if MAX_PREDICTION_LENGTH is not None:
            prediction_length = min(prediction_length, MAX_PREDICTION_LENGTH)

        # When prediction_length is capped, regenerate test instances with the
        # shorter horizon so that label shapes match the forecast shapes.
        if prediction_length < dataset.prediction_length:
            _, test_template = gluonts_split(
                dataset.gluonts_dataset,
                offset=-dataset.prediction_length * dataset.windows,
            )
            test_data = test_template.generate_instances(
                prediction_length=prediction_length,
                windows=dataset.windows,
                distance=dataset.prediction_length,
            )
        else:
            test_data = dataset.test_data

        PredictorCls = (ChronosQuantilePredictor if model_type == "chronos"
                        else MoiraiQuantilePredictor)
        predictor = PredictorCls(
            module=module,
            prediction_length=prediction_length,
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
            test_data=test_data,
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
    finally:
        # Per-item wall time: the input any future cost-aware scheduling needs,
        # and the only way to tell a balanced run from a lucky one.
        _dt = time.time() - _t0
        _item_times.append((f"{ds_name}/{term}", _dt))
        print(f"[rank {RANK}] ⏱ {ds_name}/{term} took {_dt:.1f}s")

_busy = sum(dt for _, dt in _item_times)
print(f"[rank {RANK}] DONE → {csv_path}")
print(f"[rank {RANK}] TIMING busy={_busy:.1f}s items={len(_item_times)} "
      f"slowest={sorted(_item_times, key=lambda x: -x[1])[:3]}")