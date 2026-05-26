#!/usr/bin/env python
import argparse, glob, os
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--output-dir", required=True)
args = ap.parse_args()

shards = sorted(glob.glob(os.path.join(args.output_dir, "all_results_rank*.csv")))
if not shards:
    raise SystemExit(f"No shards found in {args.output_dir}")

dfs = [pd.read_csv(s) for s in shards]
merged = pd.concat(dfs, ignore_index=True)
merged = merged.drop_duplicates(subset=["dataset", "model"], keep="last")
merged = merged.sort_values("dataset").reset_index(drop=True)

out = os.path.join(args.output_dir, "all_results.csv")
merged.to_csv(out, index=False)
print(f"Merged {len(shards)} shards → {out}  ({len(merged)} rows)")