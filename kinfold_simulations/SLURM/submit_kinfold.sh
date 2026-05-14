#!/bin/bash
#SBATCH --job-name=kinfold_fpts
#SBATCH --array=0-7999           # adjust to nrows-1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=02:00:00          # adjust after testing one sequence
#SBATCH --output=logs/kinfold_%A_%a.out
#SBATCH --error=logs/kinfold_%A_%a.err

# ── adjust these ──────────────────────────────────────────────────────────────
INPUT_PARQUET="dataset.parquet"
OUTDIR="fpt_results"
CONDA_ENV="your_env_name"        # name from your environment.yml
# ─────────────────────────────────────────────────────────────────────────────

mkdir -p logs

source $(conda info --base)/etc/profile.d/conda.sh
conda activate $CONDA_ENV

python run_kinfold.py \
    --index  $SLURM_ARRAY_TASK_ID \
    --input  $INPUT_PARQUET \
    --outdir $OUTDIR
