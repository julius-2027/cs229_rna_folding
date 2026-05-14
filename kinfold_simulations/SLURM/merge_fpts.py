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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  type=str, default="dataset.parquet")
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
        fpath = fptdir / f"fpts_{i:06d}.npy"
        if fpath.exists():
            fpt_matrix.append(np.load(fpath))
        else:
            missing.append(i)
            fpt_matrix.append(np.full(1000, np.nan, dtype=np.float32))

    if missing:
        print(f"WARNING: {len(missing)} missing FPT files (rows): {missing[:20]}{'...' if len(missing) > 20 else ''}")
        print("Resubmit missing jobs with:")
        print(f"  sbatch --array={','.join(str(i) for i in missing[:20])} submit_kinfold.sh")

    df["fpts"] = fpt_matrix

    df.to_parquet(args.output, index=False)
    print(f"Done. Wrote {n_rows} rows -> {args.output}")
    print(f"To get FPT matrix: np.stack(df['fpts'].values)  # shape ({n_rows}, 1000)")


if __name__ == "__main__":
    main()
