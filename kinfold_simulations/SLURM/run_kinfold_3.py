"""
run_kinfold_3.py (Even more memory efficient version)

Runs Kinfold 1000 times for a single sequence (identified by row index)
and saves the FPT vector to a temporary .txt file.

Usage:
    python run_kinfold.py --index 42 --input dataset.parquet --outdir fpt_results/

The SLURM array job calls this with --index $SLURM_ARRAY_TASK_ID
"""

import argparse
import subprocess
import numpy as np
import pandas as pd
from pathlib import Path


N_RUNS  = 1000
KINFOLD = "Kinfold"   # change if not on PATH


def run_kinfold(sequence: str, mfe: float, n_runs: int = N_RUNS) -> list:
    """
    Run Kinfold once with --num n_runs and parse all FPTs from output.
    Kinfold input: sequence on one line, start structure (open chain) on next.
    """
    open_chain = "." * len(sequence)
    inp = f"{sequence}\n{open_chain}\n"

    fpts = []
    
    for i in range(n_runs):
        result = subprocess.run(
            #[KINFOLD, "--num", str(n_runs), "--time", "1000000", "--fpt"],
            #[KINFOLD, "--num", str(1), "--time", "1000000000", "--log", "kinout", "--lmin"],
            [KINFOLD, "--num", str(1), "--time", "1000000000", "--log", "kinout", "--cut", str(mfe+1e-5)],
            input=inp,
            capture_output=True,
            text=True
        )

        for line in result.stdout.splitlines():
            line = line.strip()
            #print(line)
            if line[-2:]=="X1":
                try:
                    fpt = float(line.split(" ")[-2].strip())
                    #print(fpt)
                    fpts.append(fpt)
                except ValueError:
                    pass

    return fpts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index",  type=int, required=True,
                        help="Row index into the parquet file")
    parser.add_argument("--input",  type=str, default="dataset.parquet",
                        help="Input parquet file")
    parser.add_argument("--outdir", type=str, default="fpt_results",
                        help="Directory to write per-row .txt files")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    #out_path = outdir / f"fpts_{args.index:06d}.npy"
    out_path = outdir / f"fpts_{args.index:06d}.txt"
    
    if out_path.exists():
        print(f"[{args.index}] Already done, skipping.")
        return

    df = pd.read_parquet(args.input, columns=["sequence","mfe"])
    seq = df.iloc[args.index]["sequence"]
    mfe = df.iloc[args.index]["mfe"]
    print(f"[{args.index}] Running Kinfold for: {seq[:30]}...")

    fpts = run_kinfold(seq, mfe, N_RUNS)

    if len(fpts) != N_RUNS:
        print(f"[{args.index}] WARNING: got {len(fpts)}/{N_RUNS} FPTs")

    np.savetxt(out_path, np.array(fpts, dtype=np.float32))
    print(f"[{args.index}] Saved {len(fpts)} FPTs -> {out_path}")


if __name__ == "__main__":
    main()
