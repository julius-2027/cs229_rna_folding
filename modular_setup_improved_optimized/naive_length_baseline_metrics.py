"""
naive_length_baseline_metrics.py
----------------------------------
Writes results/metrics_naive_baseline_best_params_20ep.json and
results/metrics_length_baseline_best_params_20ep.json in the same format as
run_experiments_best_params.py's outputs, so both baselines can be plotted
alongside the trained models in loss_graphing_train_val.py. Both are
closed-form (not gradient-trained), so their "history" is a flat line
repeated 20 times -- there's nothing that changes epoch to epoch.

  naive_baseline:  always predicts the training-set marginal distribution.
  length_baseline: predicts the mean target distribution of all training
                   examples with the SAME sequence length (falls back to the
                   overall training marginal for lengths never seen in
                   training).

Uses the same data pipeline (DATA_PATH, split) as run_experiments_best_params.py
so results are directly comparable to the trained models' metrics jsons.

Run with: python naive_length_baseline_metrics.py
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import jensenshannon

from data import prepare_data
from train_utils import set_seed

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
RESULTS_DIR = BASE_DIR / "results"
SEED = 42
N_BINS = 50
EPOCHS = 20
EPS = 1e-8


def kl_div(target, pred):
    pred = np.clip(pred, EPS, None)
    pred = pred / pred.sum(axis=-1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = np.where(target > 0, target * (np.log(target) - np.log(pred)), 0.0)
    return terms.sum(axis=-1)


def js_dist(targets, preds):
    return np.array([jensenshannon(t, p) for t, p in zip(targets, preds)])


def write_metrics_json(name, train_loss, val_loss, test_kl, test_js_mean, test_js_median, params=0, extra=None):
    history = {
        "train_loss": [float(train_loss)] * EPOCHS,
        "val_loss": [float(val_loss)] * EPOCHS,
    }
    summary = {
        "model": name,
        "version": "best_params_20ep",
        "epochs": EPOCHS,
        "params": params,
        "final_train_loss": float(train_loss),
        "final_val_loss": float(val_loss),
        "best_val_loss": float(val_loss),
        "test_kl": float(test_kl),
        "test_js_mean": float(test_js_mean),
        "test_js_median": float(test_js_median),
        "history": history,
    }
    if extra:
        summary.update(extra)

    out_path = RESULTS_DIR / f"metrics_{name}_best_params_20ep.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"Wrote {out_path}")
    print(f"  train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  test_kl={test_kl:.4f}  test_js_mean={test_js_mean:.4f}")


if __name__ == "__main__":
    handpicked_cols = ["gc_content", "mfe", "n_local_minima"]

    set_seed(SEED)
    data = prepare_data(
        DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols,
        bin_edges_path=BIN_EDGES_PATH, data_mask=None,
    )

    train_targets = data.train.targets
    val_targets = data.val.targets
    test_targets = data.test.targets

    # ---- naive_baseline: always predict the training marginal ------------
    mean_train = train_targets.mean(axis=0)

    train_pred = np.tile(mean_train, (len(train_targets), 1))
    val_pred = np.tile(mean_train, (len(val_targets), 1))
    test_pred = np.tile(mean_train, (len(test_targets), 1))

    naive_train_loss = kl_div(train_targets, train_pred).mean()
    naive_val_loss = kl_div(val_targets, val_pred).mean()
    naive_test_kl = kl_div(test_targets, test_pred).mean()
    naive_test_js = js_dist(test_targets, test_pred)

    write_metrics_json(
        "naive_baseline", naive_train_loss, naive_val_loss,
        naive_test_kl, naive_test_js.mean(), np.median(naive_test_js),
        params=0, extra={"note": "closed-form training-set mean, not gradient-trained"},
    )

    # ---- length_baseline: predict the mean target for that sequence length
    train_lengths = np.array([len(s) for s in data.train.sequences])
    val_lengths = np.array([len(s) for s in data.val.sequences])
    test_lengths = np.array([len(s) for s in data.test.sequences])

    length_means = defaultdict(list)
    for length, target in zip(train_lengths, train_targets):
        length_means[int(length)].append(target)
    length_means = {length: np.mean(np.stack(vals), axis=0) for length, vals in length_means.items()}
    n_length_groups = len(length_means)

    def predict_for_lengths(lengths):
        return np.stack([length_means.get(int(l), mean_train) for l in lengths])

    train_pred = predict_for_lengths(train_lengths)
    val_pred = predict_for_lengths(val_lengths)
    test_pred = predict_for_lengths(test_lengths)

    length_train_loss = kl_div(train_targets, train_pred).mean()
    length_val_loss = kl_div(val_targets, val_pred).mean()
    length_test_kl = kl_div(test_targets, test_pred).mean()
    length_test_js = js_dist(test_targets, test_pred)

    n_unseen_test_lengths = sum(1 for l in test_lengths if int(l) not in length_means)
    write_metrics_json(
        "length_baseline", length_train_loss, length_val_loss,
        length_test_kl, length_test_js.mean(), np.median(length_test_js),
        params=0,
        extra={
            "note": "closed-form per-length training mean, not gradient-trained",
            "n_length_groups": n_length_groups,
            "n_test_examples_with_unseen_length": n_unseen_test_lengths,
        },
    )
