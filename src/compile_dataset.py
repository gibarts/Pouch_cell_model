"""
Collect the per-case shards written by src.optimisation into one dataset file per
topology, named as the files in data/.

Failed cases keep their row, with the objectives that were not computed left empty and
the failure recorded in the Error column. All 800 published cases succeeded.

Usage (from the repository root):
    python -m src.compile_dataset --cooling Tab
"""
import argparse
import glob
import os

import pandas as pd

from src.paths import OUTPUT_DIR, SHARD_DIR

DATASET_NAMES = {
    "Tab": "Master_{topo}_Optimization_Dataset.csv",
    "Surface": "Master_{topo}_Optimization_Dataset_Surface.csv",
}


def compile_shards(cooling, shard_dir=None, out_dir=None):
    shard_dir = shard_dir or os.path.join(SHARD_DIR, cooling)
    out_dir = out_dir or os.path.join(OUTPUT_DIR, "datasets")
    os.makedirs(out_dir, exist_ok=True)

    for topo in ("CT", "NT"):
        shard_files = glob.glob(os.path.join(shard_dir, f"shard_{topo}_run_*.csv"))
        if not shard_files:
            continue

        frames = []
        for f in shard_files:
            try:
                df = pd.read_csv(f)
            except pd.errors.EmptyDataError:
                print(f"   [!] Skipped empty shard: {f}")
                continue
            # Run ID from the file name, e.g. shard_CT_run_42.csv -> 42
            df.insert(0, "Run_ID", int(os.path.basename(f).split("_run_")[-1].replace(".csv", "")))
            frames.append(df)

        master = pd.concat(frames, ignore_index=True).sort_values("Run_ID").reset_index(drop=True)
        n_success = int((master["Error"] == "Success").sum())
        print(f"{topo}, {cooling} cooling: {len(master)} cases, {n_success} succeeded")

        out_csv = os.path.join(out_dir, DATASET_NAMES[cooling].format(topo=topo))
        master.to_csv(out_csv, index=False)
        print(f"   written to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile DoE shards into dataset files.")
    parser.add_argument("--cooling", choices=sorted(DATASET_NAMES), required=True)
    parser.add_argument("--shard-dir", default=None, help="default: outputs/shards/<cooling>")
    parser.add_argument("--out-dir", default=None, help="default: outputs/datasets")
    args = parser.parse_args()
    compile_shards(args.cooling, args.shard_dir, args.out_dir)
