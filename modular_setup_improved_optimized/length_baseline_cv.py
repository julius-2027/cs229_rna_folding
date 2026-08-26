"""
length_baseline_cv.py
------------------------
Computes length_baseline's 5-fold CV KL and test KL using the exact same
data pipeline/splits as cross_val_final_with_test.py (DATA_PATH=
synthetic+real_dataset_filtered.parquet, MASK_PATH=None, test_size=0.15,
K=5 KFold, seed=42) so it's directly comparable to the other models in that
table. length_baseline predicts, for each example, the mean target
distribution of all *fold-training* examples with the same sequence length
(falls back to the fold's overall training mean for unseen lengths).

Run with: python length_baseline_cv.py
"""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split, KFold

from data import load_dataset, build_histogram_targets

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered.parquet"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"

SEED = 42
N_BINS = 50
K = 5
TEST_SIZE = 0.15
EPS = 1e-8


def kl_div(target, pred):
    pred = np.clip(pred, EPS, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(target > 0, target * (np.log(target) - np.log(pred)), 0.0)
    return terms.sum(axis=-1)


def build_length_lookup(lengths, targets):
    lookup = {}
    for length in np.unique(lengths):
        lookup[int(length)] = targets[lengths == length].mean(axis=0)
    return lookup


def predict_for_lengths(lengths, lookup, fallback):
    return np.stack([lookup.get(int(l), fallback) for l in lengths])


if __name__ == "__main__":
    df = load_dataset(DATA_PATH)
    y, _ = build_histogram_targets(df, n_bins=N_BINS, bin_edges=np.load(BIN_EDGES_PATH))
    lengths = df["length"].to_numpy()

    idx = np.arange(len(y))
    idx_trainval, idx_test = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)

    y_trainval, y_test = y[idx_trainval], y[idx_test]
    lengths_trainval, lengths_test = lengths[idx_trainval], lengths[idx_test]

    kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)
    fold_kl = []
    for train_pos, val_pos in kfold.split(idx_trainval):
        y_train_fold = y_trainval[train_pos]
        lengths_train_fold = lengths_trainval[train_pos]
        y_val_fold = y_trainval[val_pos]
        lengths_val_fold = lengths_trainval[val_pos]

        lookup = build_length_lookup(lengths_train_fold, y_train_fold)
        fallback = y_train_fold.mean(axis=0)
        preds = predict_for_lengths(lengths_val_fold, lookup, fallback)
        fold_kl.append(float(kl_div(y_val_fold, preds).mean()))

    cv_kl_mean = float(np.mean(fold_kl))
    cv_kl_std = float(np.std(fold_kl))

    lookup = build_length_lookup(lengths_trainval, y_trainval)
    fallback = y_trainval.mean(axis=0)
    test_preds = predict_for_lengths(lengths_test, lookup, fallback)
    test_kl = float(kl_div(y_test, test_preds).mean())

    print(f"fold_kl={fold_kl}")
    print(f"cv_kl={cv_kl_mean:.4f} +/- {cv_kl_std:.4f}  test_kl={test_kl:.4f}")
