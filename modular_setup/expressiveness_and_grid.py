"""
expressiveness_and_grid.py
-----------------------------
For the OLD modular_setup embed_transformer checkpoint (best_embed_transformer_v1.pth):
  1. Computes the expressiveness ratio (pred_diversity / true_diversity) and
     calibration, same definitions as modular_setup_improved_optimized's
     expressiveness.py: pred_diversity/true_diversity are the mean JS
     distance of each example's distribution from the mean distribution
     (predicted vs. true respectively); calibration is the JS distance
     between the mean predicted and mean true distribution.
  2. Plots a 1x4 grid of 4 random test examples, same visual format as
     modular_setup_improved_optimized/example_grid.py (no axis ticks, one
     shared legend, ax.stairs for bin-accurate curves, KL divergence
     annotated per-cell, L-shaped Probability/ln(folding time) axis
     indicator in the bottom-left) -- except colors are swapped: True is
     now blue, Predicted is now black.

Run with: python expressiveness_and_grid.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch
from scipy.spatial.distance import jensenshannon

import models as ms
from data import prepare_data, make_dataloader
from train_utils import set_seed, evaluate_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding/kinpfn_testing_set/parquet_parsing/test_val_dataset.parquet"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_embed_transformer_v1.pth"
SEED = 42
N_BINS = 50
DEVICE = "cpu"
N_RANDOM_EXAMPLES = 4
RANDOM_SEED_FOR_EXAMPLES = 123
EPS = 1e-8


def kl_div(target, pred):
    pred = np.clip(pred, EPS, None)
    pred = pred / pred.sum()
    mask = target > 0
    return float(np.sum(target[mask] * (np.log(target[mask]) - np.log(pred[mask]))))


def mean_js_from_center(dists, center):
    return float(np.mean([jensenshannon(d, center) for d in dists]))


if __name__ == "__main__":
    set_seed(SEED)
    handpicked_cols = ["gc_content", "mfe", "n_local_minima"]
    data = prepare_data(DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols)
    bin_edges = data.bin_edges
    static_dim = data.train.static_features.shape[1]
    output_dim = data.train.targets.shape[1]

    model = ms.Embed_Transformer(
        embedding_dim=640, projection_dim=64, num_heads=4, num_layers=1,
        dropout=0.5, static_feature_size=static_dim, output_size=output_dim,
        mlp_hidden_size=64,
    ).to(DEVICE)

    test_loader = make_dataloader(data.test, batching=model.BATCHING, batch_size=64, shuffle=False)
    eval_result = evaluate_model(model, test_loader, device=DEVICE, checkpoint_path=CHECKPOINT_PATH)
    preds = eval_result["preds"]
    targets = eval_result["targets"]

    # ---- expressiveness ratio ----------------------------------------------
    mean_pred = preds.mean(axis=0)
    mean_target = targets.mean(axis=0)
    pred_diversity = mean_js_from_center(preds, mean_pred)
    true_diversity = mean_js_from_center(targets, mean_target)
    expressiveness_ratio = pred_diversity / true_diversity if true_diversity > 0 else float("nan")
    calibration = float(jensenshannon(mean_pred, mean_target))

    print(f"embed_transformer (old modular_setup, n_test={len(targets)}):")
    print(f"  pred_diversity        = {pred_diversity:.4f}")
    print(f"  true_diversity        = {true_diversity:.4f}")
    print(f"  expressiveness_ratio  = {expressiveness_ratio:.4f}")
    print(f"  calibration           = {calibration:.4f}")

    # ---- 1x4 grid of random test examples ----------------------------------
    rng = np.random.default_rng(RANDOM_SEED_FOR_EXAMPLES)
    example_indices = rng.choice(len(targets), size=N_RANDOM_EXAMPLES, replace=False)
    print(f"  random example indices = {sorted(example_indices.tolist())}")

    fig, axes = plt.subplots(1, N_RANDOM_EXAMPLES, figsize=(2.2 * N_RANDOM_EXAMPLES, 4.5))

    for col, idx in enumerate(example_indices):
        ax = axes[col]
        true = targets[idx]
        pred = preds[idx]
        kl = kl_div(true, pred)

        ax.stairs(true, bin_edges, color="#1f77b4", linewidth=1.3, alpha=0.6, baseline=None)
        ax.stairs(pred, bin_edges, color="black", linewidth=1.3, baseline=None)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

        combined = np.maximum(true, pred)
        edge = max(1, len(combined) // 6)
        left_max, right_max = combined[:edge].max(), combined[-edge:].max()
        ha = "left" if left_max <= right_max else "right"
        x = 0.04 if ha == "left" else 0.96
        ax.text(
            x, 0.94, f"{kl:.2f}".lstrip("0"), transform=ax.transAxes,
            ha=ha, va="top", fontsize=20, color="black", fontweight="bold",
        )

    fig.subplots_adjust(wspace=0, hspace=0)

    legend_handles = [
        mlines.Line2D([], [], color="#1f77b4", alpha=0.6, linewidth=2, label="True"),
        mlines.Line2D([], [], color="black", linewidth=2, label="Predicted"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.55, 1.0),
        ncol=2, fontsize=20, frameon=True, shadow=True,
    )

    plt.tight_layout(rect=[0.22, 0.3, 1, 0.9])
    fig.subplots_adjust(wspace=0, hspace=0)

    corner_x, corner_y = 0.16, 0.21
    arm_len = 0.07
    fig.add_artist(mlines.Line2D(
        [corner_x, corner_x], [corner_y, corner_y + arm_len],
        transform=fig.transFigure, color="black", linewidth=1.5,
    ))
    fig.add_artist(mlines.Line2D(
        [corner_x, corner_x + arm_len], [corner_y, corner_y],
        transform=fig.transFigure, color="black", linewidth=1.5,
    ))
    fig.text(
        corner_x - 0.09, corner_y + arm_len / 2, "Probability",
        va="center", ha="center", rotation=90, fontsize=20,
    )
    fig.text(
        corner_x + arm_len / 2, corner_y - 0.13, "ln(folding time)",
        va="center", ha="center", fontsize=20,
    )

    out_path = BASE_DIR / "results" / "embed_transformer_example_grid.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
