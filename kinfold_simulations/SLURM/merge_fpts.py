"""
merge_fpts.py

After all SLURM jobs finish, run this locally to assemble the per-row
.npy files into a final parquet with an `fpts` column.

Usage:
    python merge_fpts.py --input dataset.parquet --fptdir fpt_results/ --output dataset_with_fpts.parquet
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

N_RUNS = 1000

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="features.parquet")
    parser.add_argument("--fptdir", type=str, default="fpt_results")
    parser.add_argument("--output", type=str, default="dataset_with_fpts.parquet")
    args = parser.parse_args()

    df     = pd.read_parquet(args.input)
    n_rows = len(df)
    fptdir = Path(args.fptdir)

    print(f"Merging FPTs for {n_rows} sequences from {fptdir}...")

    fpt_matrix = []
    missing    = []

    for i in range(n_rows):
        fpath = fptdir / f"fpts_{i:06d}.txt"
        if fpath.exists():
            with open(fpath, 'r') as file:
                num_fpts_done = sum(1 for line in file)
            if num_fpts_done == N_RUNS:
                arr_fpts = np.loadtxt(fpath)
            elif num_fpts_done >= 2:
                arr_fpts = np.pad(np.loadtxt(fpath), (0,N_RUNS-num_fpts_done),
                                  constant_values=np.float64(np.nan))
            elif num_fpts_done == 1:
                arr_fpts = np.pad([np.loadtxt(fpath)], (0,N_RUNS-num_fpts_done),
                                  constant_values=np.float64(np.nan))
            else:
                arr_fpts = np.full(N_RUNS, np.nan, dtype=np.float64)
        else:
            missing.append(i)
            arr_fpts = np.full(N_RUNS, np.nan, dtype=np.float64)

        if arr_fpts.size != N_RUNS:
            print(i, num_fpts_done, arr_fpts)
        fpt_matrix.append(arr_fpts)

    df['fpts'] = fpt_matrix
    df['fpts'] = df['fpts'].apply(lambda x: np.array(x).flatten() if isinstance(x, (list, np.ndarray)) else x)
    # mask_rows = df['fpts'].apply(lambda arr: np.any(np.isnan(arr)))
    # df_subset = df[mask_rows]
    # df = df_subset

    df.to_parquet(args.output, index=False)
    print(f"Done. Wrote {len(df)} rows -> {args.output}")
    print(f"To get FPT matrix: np.stack(df['fpts'].values)  # shape ({len(df)}, N_RUNS)")

if __name__ == "__main__":
    main()
