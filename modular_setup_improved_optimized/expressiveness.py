"""
expressiveness.py
-------------------
Checks whether each model is actually using its inputs to vary its predicted
distribution across examples, or just collapsing to (approximately) the same
prediction every time -- i.e. whether it learned anything more than "always
predict the marginal target distribution".

For each model, across the whole held-out test set:
  - pred_diversity: mean JS distance of each example's PREDICTED distribution
    from the average predicted distribution. Low = predictions barely vary
    across different inputs (mean-collapse).
  - true_diversity: same measure computed on the TRUE target distributions --
    the ceiling a maximally expressive model could hope to match.
  - expressiveness_ratio = pred_diversity / true_diversity. ~0 means the
    model is basically ignoring its input and always predicting close to the
    same (marginal) distribution; ~1 means its predictions vary across
    examples about as much as the real targets do.
  - calibration: JS distance between the average prediction and the average
    true target -- checks the model's mean prediction isn't off-center even
    before asking about spread.

Uses the checkpoints/best_<name>_best_params_20ep.pth checkpoints from
run_experiments_best_params.py. Pure inference, no training.

Run with: python expressiveness.py
"""

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import jensenshannon

from run_experiments_best_params import build_model_configs, EXCLUDE_MODELS
from data import prepare_data, make_dataloader
from train_utils import set_seed, evaluate_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
OUTPUT_CSV = BASE_DIR / "expressiveness.csv"

SEED = 42
N_BINS = 50
DEVICE = "cpu"
VERSION_SUFFIX = "best_params_20ep"

MODEL_ORDER = [
    "glm_baseline",
    "mean_pool_mlp",
    "all_local_mlp",
    "lstm",
    "bilstm",
    "lstm_rna_fm",
    "struct_lstm_rna_fm",
    "transformer_rna_fm",
    "loc_rna_fm",
]


def mean_js_from_center(dists, center):
    return float(np.mean([jensenshannon(d, center) for d in dists]))


def run_one_model(name, embeddings_dict):
    is_glm = (name == "glm_baseline")
    all_local_data = (name == "all_local_mlp")
    if is_glm:
        handpicked_cols = [
            "mfe", "n_local_minima", "gc_content",
            "freq_A", "freq_U", "freq_G", "freq_C",
            "freq_AA", "freq_AU", "freq_AG", "freq_AC",
            "freq_UA", "freq_UU", "freq_UG", "freq_UC",
            "freq_GA", "freq_GU", "freq_GG", "freq_GC",
            "freq_CA", "freq_CU", "freq_CG", "freq_CC",
        ]
    else:
        handpicked_cols = ["gc_content", "mfe", "n_local_minima"]

    set_seed(SEED)
    data = prepare_data(
        DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols,
        bin_edges_path=BIN_EDGES_PATH, data_mask=None,
        local_minima_k=10 if all_local_data else None,
    )
    static_dim = data.train.static_features.shape[1]
    output_dim = data.train.targets.shape[1]

    model_configs = build_model_configs(static_dim, output_dim)
    model = model_configs[name]["build"]().to(DEVICE)

    checkpoint_path = CHECKPOINT_DIR / f"best_{name}_{VERSION_SUFFIX}.pth"
    test_loader = make_dataloader(
        data.test, batching=model.BATCHING, batch_size=64,
        shuffle=False, embedding_dict=embeddings_dict,
    )

    eval_result = evaluate_model(model, test_loader, device=DEVICE, checkpoint_path=checkpoint_path)
    preds = eval_result["preds"]
    targets = eval_result["targets"]

    mean_pred = preds.mean(axis=0)
    mean_target = targets.mean(axis=0)

    pred_diversity = mean_js_from_center(preds, mean_pred)
    true_diversity = mean_js_from_center(targets, mean_target)
    expressiveness_ratio = pred_diversity / true_diversity if true_diversity > 0 else float("nan")
    calibration = float(jensenshannon(mean_pred, mean_target))

    print(f"{name:20s} pred_diversity={pred_diversity:.4f} true_diversity={true_diversity:.4f} "
          f"ratio={expressiveness_ratio:.3f} calibration={calibration:.4f}")

    return {
        "model": name,
        "pred_diversity": pred_diversity,
        "true_diversity": true_diversity,
        "expressiveness_ratio": expressiveness_ratio,
        "calibration": calibration,
        "n_test": len(preds),
    }


if __name__ == "__main__":
    print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
    embeddings_dict = torch.load(EMBEDDING_PATH)

    rows = []
    for name in MODEL_ORDER:
        if name in EXCLUDE_MODELS:
            continue
        rows.append(run_one_model(name, embeddings_dict))

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["model", "pred_diversity", "true_diversity", "expressiveness_ratio", "calibration"]
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "model": r["model"],
                "pred_diversity": f"{r['pred_diversity']:.4f}",
                "true_diversity": f"{r['true_diversity']:.4f}",
                "expressiveness_ratio": f"{r['expressiveness_ratio']:.4f}",
                "calibration": f"{r['calibration']:.4f}",
            })

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    print(pd.DataFrame(rows)[["model", "pred_diversity", "true_diversity", "expressiveness_ratio", "calibration"]].to_string(index=False))
