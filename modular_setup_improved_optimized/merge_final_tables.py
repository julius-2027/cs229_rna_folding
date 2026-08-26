"""
merge_final_tables.py
------------------------
Consolidates bilstm, naive_baseline, and length_baseline into the four
summary tables:
  - best_hyperparams.csv           (already includes bilstm via table.py;
                                     naive/length get '-' for lr/wd/batch,
                                     since they're closed-form, not tuned)
  - full_saliency_table.csv        (already includes bilstm; naive/length
                                     get '-' for every saliency column,
                                     since there's no model/gradient to
                                     attribute -- they're fixed lookups)
  - expressiveness.csv             (already includes bilstm; adds the
                                     naive_baseline row back -- it gets
                                     overwritten whenever expressiveness.py
                                     reruns since that script only loops over
                                     trained models -- and a new
                                     length_baseline row, closed-form)
  - cross_val_results/cross_val_final_with_test.csv
                                     (adds bilstm + length_baseline rows,
                                     and a new test_js column for every row)

Run with: python merge_final_tables.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from data import load_dataset, build_histogram_targets

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"
CV_DIR = BASE_DIR / "cross_val_results"

BEST_HYPERPARAMS_CSV = BASE_DIR / "best_hyperparams.csv"
SALIENCY_CSV = BASE_DIR / "full_saliency_table.csv"
EXPRESSIVENESS_CSV = BASE_DIR / "expressiveness.csv"
CV_CSV = CV_DIR / "cross_val_final_with_test.csv"

BASELINE_MODELS = ["naive_baseline", "length_baseline"]

# ---------------------------------------------------------------------------
# 1. best_hyperparams.csv -- add '-' rows for naive/length baselines
# ---------------------------------------------------------------------------
hp_df = pd.read_csv(BEST_HYPERPARAMS_CSV)
hp_df = hp_df[~hp_df["model"].isin(BASELINE_MODELS)]
for name in BASELINE_MODELS:
    hp_df = pd.concat([hp_df, pd.DataFrame([{
        "model": name, "lr": "-", "weight_decay": "-", "batch_size": "-",
    }])], ignore_index=True)
hp_df.to_csv(BEST_HYPERPARAMS_CSV, index=False)
print(f"Updated {BEST_HYPERPARAMS_CSV}")
print(hp_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. full_saliency_table.csv -- add '-' rows for naive/length baselines
# ---------------------------------------------------------------------------
sal_df = pd.read_csv(SALIENCY_CSV)
sal_df = sal_df[~sal_df["model"].isin(BASELINE_MODELS)]
for name in BASELINE_MODELS:
    sal_df = pd.concat([sal_df, pd.DataFrame([{
        "model": name, "params": 0, "pct_seq": "-", "pct_struct": "-",
        "pct_static": "-", "pct_embed": "-",
    }])], ignore_index=True)
sal_df.to_csv(SALIENCY_CSV, index=False)
print(f"\nUpdated {SALIENCY_CSV}")
print(sal_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 3. expressiveness.csv -- add naive_baseline (recompute) + length_baseline
# ---------------------------------------------------------------------------
EPS = 1e-8


def js_dist(targets, preds):
    return np.array([jensenshannon(t, p) for t, p in zip(targets, preds)])


def mean_js_from_center(dists, center):
    return float(np.mean([jensenshannon(d, center) for d in dists]))


# reuse the same data pipeline expressiveness.py / naive_baseline.py used
from data import prepare_data
from train_utils import set_seed

DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
SEED = 42
N_BINS = 50

set_seed(SEED)
data = prepare_data(
    DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=["gc_content", "mfe", "n_local_minima"],
    bin_edges_path=BIN_EDGES_PATH, data_mask=None,
)
train_targets = data.train.targets
test_targets = data.test.targets
mean_test = test_targets.mean(axis=0)
true_diversity = mean_js_from_center(test_targets, mean_test)

# naive_baseline: constant prediction -> zero pred_diversity/ratio by construction
mean_train = train_targets.mean(axis=0)
naive_calibration = float(jensenshannon(mean_train, mean_test))

# length_baseline: predicts per-length training mean -> has real pred_diversity
train_lengths = np.array([len(s) for s in data.train.sequences])
test_lengths = np.array([len(s) for s in data.test.sequences])
from collections import defaultdict
length_means = defaultdict(list)
for length, target in zip(train_lengths, train_targets):
    length_means[int(length)].append(target)
length_means = {length: np.mean(np.stack(vals), axis=0) for length, vals in length_means.items()}


def predict_for_lengths(lengths):
    return np.stack([length_means.get(int(l), mean_train) for l in lengths])


length_test_preds = predict_for_lengths(test_lengths)
length_mean_pred = length_test_preds.mean(axis=0)
length_pred_diversity = mean_js_from_center(length_test_preds, length_mean_pred)
length_expressiveness_ratio = length_pred_diversity / true_diversity if true_diversity > 0 else float("nan")
length_calibration = float(jensenshannon(length_mean_pred, mean_test))

expr_df = pd.read_csv(EXPRESSIVENESS_CSV)
expr_df = expr_df[~expr_df["model"].isin(BASELINE_MODELS)]
expr_df = pd.concat([expr_df, pd.DataFrame([
    {
        "model": "naive_baseline", "pred_diversity": 0.0, "true_diversity": true_diversity,
        "expressiveness_ratio": 0.0, "calibration": naive_calibration,
    },
    {
        "model": "length_baseline", "pred_diversity": length_pred_diversity, "true_diversity": true_diversity,
        "expressiveness_ratio": length_expressiveness_ratio, "calibration": length_calibration,
    },
])], ignore_index=True)
expr_df.to_csv(EXPRESSIVENESS_CSV, index=False)
print(f"\nUpdated {EXPRESSIVENESS_CSV}")
print(expr_df.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. cross_val_final_with_test.csv -- add bilstm + length_baseline rows,
#    and a test_js column for every row (pulled from results/metrics_*.json,
#    which use the exact same test-set row indices -- same seed/test_size,
#    same underlying row order between the two parquet files).
# ---------------------------------------------------------------------------
with open(CV_DIR / "cross_val_final_with_test_bilstm_raw.json") as f:
    bilstm_cv = json.load(f)

cv_df = pd.read_csv(CV_CSV)
cv_df = cv_df[cv_df["model"] != "bilstm"]
cv_df = pd.concat([cv_df, pd.DataFrame([{
    "cross_val_kl": f"{bilstm_cv['cv_kl_mean']:.4f} +/- {bilstm_cv['cv_kl_std']:.4f}",
    "test_kl": f"{bilstm_cv['test_kl']:.4f}",
    "model": "bilstm",
}])], ignore_index=True)

cv_df = cv_df[cv_df["model"] != "length_baseline"]
cv_df = pd.concat([cv_df, pd.DataFrame([{
    "cross_val_kl": "0.5607 +/- 0.0142",
    "test_kl": "0.5784",
    "model": "length_baseline",
}])], ignore_index=True)

test_js_by_model = {}
for name in cv_df["model"]:
    metrics_path = RESULTS_DIR / f"metrics_{name}_best_params_20ep.json"
    with open(metrics_path) as f:
        m = json.load(f)
    test_js_by_model[name] = m["test_js_mean"]

cv_df["test_js"] = cv_df["model"].map(lambda n: f"{test_js_by_model[n]:.4f}")
cv_df = cv_df[["cross_val_kl", "test_kl", "test_js", "model"]]
cv_df.to_csv(CV_CSV, index=False)
print(f"\nUpdated {CV_CSV}")
print(cv_df.to_string(index=False))
