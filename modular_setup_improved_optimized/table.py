"""
table.py
--------
Scrubs existing tune_results/*.json files and writes a CSV of each model's
best hyperparameters found during tuning (lr, weight_decay, batch_size).
lr and weight_decay are rounded to 2 significant figures.

Excludes bilstm_rna_fm_proj per request.

Run with: python table.py
"""

import json
import csv
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
TUNE_RESULTS_DIR = BASE_DIR / "tune_results"
OUTPUT_CSV = BASE_DIR / "best_hyperparams.csv"
EXCLUDE_MODELS = {"bilstm_rna_fm_proj"}


def sig2(x):
    return float(f"{x:.2g}")


rows = []
for path in sorted(TUNE_RESULTS_DIR.glob("tune_results_*.json")):
    with open(path) as f:
        data = json.load(f)

    model_name = data["model"]
    if model_name in EXCLUDE_MODELS:
        continue

    best_params = data["best_params"]
    rows.append({
        "model": model_name,
        "lr": sig2(best_params["lr"]),
        "weight_decay": sig2(best_params["weight_decay"]),
        "batch_size": best_params["batch_size"],
    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["model", "lr", "weight_decay", "batch_size"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {OUTPUT_CSV}")
for row in rows:
    print(row)
