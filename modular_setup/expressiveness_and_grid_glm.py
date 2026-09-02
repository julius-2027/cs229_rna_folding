"""
expressiveness_and_grid_glm.py
---------------------------------
Same as expressiveness_and_grid.py but for the OLD modular_setup glm_baseline
checkpoint (best_glm_baseline_v1.pth), and instead of 4 RANDOM test examples,
picks the 4 examples that most starkly demonstrate low expressivity: cases
where the true distribution is unusually far from the overall mean (high
per-example true diversity -- i.e. a case the model *should* have reacted
to) but the model's own prediction stayed close to its average prediction
(low per-example pred diversity -- i.e. it didn't react). Ranked by
(true_diversity_i - pred_diversity_i), descending.

Run with: python expressiveness_and_grid_glm.py
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
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_glm_baseline_v1.pth"
SEED = 42
N_BINS = 50
DEVICE = "cpu"
N_EXAMPLES = 4
EPS = 1e-8

HANDPICKED_COLS = [
    "mfe", "n_local_minima", "gc_content",
    "freq_A", "freq_U", "freq_G", "freq_C",
    "freq_AA", "freq_AU", "freq_AG", "freq_AC",
    "freq_UA", "freq_UU", "freq_UG", "freq_UC",
    "freq_GA", "freq_GU", "freq_GG", "freq_GC",
    "freq_CA", "freq_CU", "freq_CG", "freq_CC",
]


def kl_div(target, pred):
    pred = np.clip(pred, EPS, None)
    pred = pred / pred.sum()
    mask = target > 0
    return float(np.sum(target[mask] * (np.log(target[mask]) - np.log(pred[mask]))))


def mean_js_from_center(dists, center):
    return float(np.mean([jensenshannon(d, center) for d in dists]))


def js_per_example(dists, center):
    return np.array([jensenshannon(d, center) for d in dists])


if __name__ == "__main__":
    set_seed(SEED)
    data = prepare_data(DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=HANDPICKED_COLS)
    bin_edges = data.bin_edges
    static_dim = data.train.static_features.shape[1]
    output_dim = data.train.targets.shape[1]

    model = ms.GLMBaseline(static_dim, output_dim).to(DEVICE)

    test_loader = make_dataloader(data.test, batching=model.BATCHING, batch_size=64, shuffle=False)
    eval_result = evaluate_model(model, test_loader, device=DEVICE, checkpoint_path=CHECKPOINT_PATH)
    preds = eval_result["preds"]
    targets = eval_result["targets"]

    # ---- expressiveness ratio ----------------------------------------------
    mean_pred = preds.mean(axis=0)
    mean_target = targets.mean(axis=0)
    pred_div_i = js_per_example(preds, mean_pred)
    true_div_i = js_per_example(targets, mean_target)
    pred_diversity = float(pred_div_i.mean())
    true_diversity = float(true_div_i.mean())
    expressiveness_ratio = pred_diversity / true_diversity if true_diversity > 0 else float("nan")
    calibration = float(jensenshannon(mean_pred, mean_target))

    print(f"glm_baseline (old modular_setup, n_test={len(targets)}):")
    print(f"  pred_diversity        = {pred_diversity:.4f}")
    print(f"  true_diversity        = {true_diversity:.4f}")
    print(f"  expressiveness_ratio  = {expressiveness_ratio:.4f}")
    print(f"  calibration           = {calibration:.4f}")

    # ---- pick the 4 worst "low expressivity" examples ----------------------
    # entropy = spread/uncertainty of the distribution itself (high = wide,
    # low = peaky/narrow). Want cases where the TRUE distribution is
    # genuinely spread/uncertain (high entropy) but the model's prediction
    # is falsely peaky/overconfident (low entropy) -- true is diverse,
    # predicted is not.
    def entropy(dists):
        safe = np.clip(dists, EPS, None)
        return -np.sum(np.where(dists > 0, dists * np.log(safe), 0.0), axis=-1)

    true_entropy = entropy(targets)
    pred_entropy = entropy(preds)
    gap = true_entropy - pred_entropy
    example_indices = np.argsort(gap)[::-1][:N_EXAMPLES]
    print(f"  worst low-expressivity example indices = {example_indices.tolist()}")
    print(f"  true_entropy={true_entropy[example_indices].tolist()}")
    print(f"  pred_entropy={pred_entropy[example_indices].tolist()}")
    print(f"  their gaps = {gap[example_indices].tolist()}")

    fig, axes = plt.subplots(1, N_EXAMPLES, figsize=(2.2 * N_EXAMPLES, 5.2))

    for col, idx in enumerate(example_indices):
        ax = axes[col]
        true = targets[idx]
        pred = preds[idx]
        kl = kl_div(true, pred)

        ax.stairs(true, bin_edges, color="#1f77b4", linewidth=1.3, alpha=0.6, baseline=None)
        ax.stairs(pred, bin_edges, color="black", linewidth=1.3, baseline=None)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_box_aspect(1)
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

        combined = np.maximum(true, pred)
        edge = max(1, len(combined) // 6)
        regions = {"l": combined[:edge].max(), "r": combined[-edge:].max()}
        peak = combined.max()
        side = "l" if regions["l"] <= regions["r"] else "r"
        vpos = "b" if regions[side] > 0.5 * peak else "t"
        corners = {
            "tl": (0.04, 0.94, "left", "top"),
            "tr": (0.96, 0.94, "right", "top"),
            "bl": (0.04, 0.06, "left", "bottom"),
            "br": (0.96, 0.06, "right", "bottom"),
        }
        x, y, ha, va = corners[f"{vpos}{side}"]
        ax.text(
            x, y, f"{kl:.2f}".lstrip("0"), transform=ax.transAxes,
            ha=ha, va=va, fontsize=20, color="black", fontweight="bold",
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

    plt.tight_layout(rect=[0.26, 0.34, 1, 0.88])
    fig.subplots_adjust(wspace=0, hspace=0)

    corner_x, corner_y = 0.19, 0.24
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
        corner_x - 0.11, corner_y + arm_len / 2, "Probability",
        va="center", ha="center", rotation=90, fontsize=20,
    )
    fig.text(
        corner_x + arm_len / 2, corner_y - 0.16, "ln(folding time)",
        va="center", ha="center", fontsize=20,
    )

    out_path = BASE_DIR / "results" / "glm_baseline_low_expressivity_grid.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
