"""
naive_baseline.py
-------------------
Computes a "predict the training-set marginal distribution for every
example" baseline (zero resolution, by construction) and appends it as a row
to both:
  - expressiveness.csv           (pred_diversity, true_diversity, expressiveness_ratio, calibration)
  - cross_val_final_with_test.csv (cross_val_kl, test_kl)

This gives readers an absolute floor: any model failing to beat this is
learning nothing from its inputs.

Run with: python naive_baseline.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from sklearn.model_selection import train_test_split, KFold

from data import prepare_data, load_dataset, build_histogram_targets
from train_utils import set_seed

BASE_DIR = Path(__file__).resolve().parent
EXPR_DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
CV_DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered.parquet"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
EXPRESSIVENESS_CSV = BASE_DIR / "expressiveness.csv"
CV_CSV = BASE_DIR / "cross_val_results" / "cross_val_final_with_test.csv"

SEED = 42
N_BINS = 50
K = 5
TEST_SIZE = 0.15
EPS = 1e-8


def kl_div(target, pred):
    """KL(target || pred), per-row, target/pred shape (n, n_bins). Matches
    nn.KLDivLoss's convention: 0*log(0) terms contribute 0."""
    pred = np.clip(pred, EPS, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(target > 0, target * (np.log(target) - np.log(pred)), 0.0)
    return terms.sum(axis=-1)


def append_row(csv_path, new_row):
    df = pd.read_csv(csv_path)
    df = df[df["model"] != new_row["model"]]  # replace if rerun
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    df.to_csv(csv_path, index=False)
    return df


def compute_expressiveness_naive():
    handpicked_cols = ["gc_content", "mfe", "n_local_minima"]
    set_seed(SEED)
    data = prepare_data(
        EXPR_DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols,
        bin_edges_path=BIN_EDGES_PATH, data_mask=None,
    )
    train_targets = data.train.targets
    test_targets = data.test.targets

    mean_train = train_targets.mean(axis=0)
    mean_test = test_targets.mean(axis=0)

    true_diversity = float(np.mean([jensenshannon(t, mean_test) for t in test_targets]))
    calibration = float(jensenshannon(mean_train, mean_test))

    row = {
        "model": "naive_baseline",
        "pred_diversity": 0.0,          # constant prediction -> zero spread, by construction
        "true_diversity": true_diversity,
        "expressiveness_ratio": 0.0,
        "calibration": calibration,
    }
    print(f"[expressiveness] naive_baseline: true_diversity={true_diversity:.4f} calibration={calibration:.4f}")
    return row


def compute_cv_naive():
    df = load_dataset(CV_DATA_PATH)
    y, _ = build_histogram_targets(df, n_bins=N_BINS, bin_edges=np.load(BIN_EDGES_PATH))

    _, _, y_trainval, y_test = train_test_split(
        np.arange(len(y)), y, test_size=TEST_SIZE, random_state=SEED
    )

    kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)
    fold_kl = []
    for train_idx, val_idx in kfold.split(y_trainval):
        y_train_fold, y_val_fold = y_trainval[train_idx], y_trainval[val_idx]
        fold_mean_pred = y_train_fold.mean(axis=0)
        fold_kl.append(float(kl_div(y_val_fold, np.tile(fold_mean_pred, (len(y_val_fold), 1))).mean()))

    cv_kl_mean = float(np.mean(fold_kl))
    cv_kl_std = float(np.std(fold_kl))

    trainval_mean_pred = y_trainval.mean(axis=0)
    test_kl = float(kl_div(y_test, np.tile(trainval_mean_pred, (len(y_test), 1))).mean())

    print(f"[cross_val] naive_baseline: fold_kl={fold_kl}")
    print(f"[cross_val] naive_baseline: cv_kl={cv_kl_mean:.4f} +/- {cv_kl_std:.4f}  test_kl={test_kl:.4f}")

    return {
        "cross_val_kl": f"{cv_kl_mean:.4f} +/- {cv_kl_std:.4f}",
        "test_kl": f"{test_kl:.4f}",
        "model": "naive_baseline",
    }


if __name__ == "__main__":
    expr_row = compute_expressiveness_naive()
    expr_df = append_row(EXPRESSIVENESS_CSV, expr_row)
    print(f"\nUpdated {EXPRESSIVENESS_CSV}:")
    print(expr_df.to_string(index=False))

    cv_row = compute_cv_naive()
    cv_df = append_row(CV_CSV, cv_row)
    print(f"\nUpdated {CV_CSV}:")
    print(cv_df.to_string(index=False))
