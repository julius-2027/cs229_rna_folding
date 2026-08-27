"""
example_grid.py
-----------------
Grid figure: rows = models, columns = specific chosen test-set examples.
Each cell overlays the true distribution and that model's predicted
distribution for that example, with the KL divergence annotated inside the
cell (positioned to dodge the curves). No per-cell whitespace/ticks; one
shared "Probability" / "ln folding time" label anchors the whole grid, and
one shared legend explains the true/predicted colors.

Run with: python example_grid.py
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import torch

from run_experiments_best_params import build_model_configs, best_params_by_model
from data import prepare_data, make_dataloader
from train_utils import set_seed, evaluate_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
SEED = 42
N_BINS = 50
DEVICE = "cpu"
VERSION = "best_params_20ep"

EXAMPLE_INDICES = [71, 75, 90, 24, 28, 91, 39, 76]

LABEL_OVERRIDES = {"bilstm": "lstm"}

MODEL_ORDER = [
    "glm_baseline",
    "mean_pool_mlp",
    "all_local_mlp",
    "bilstm",
    "lstm_rna_fm",
    "struct_lstm_rna_fm",
    "transformer_rna_fm",
    "loc_rna_fm",
]

CORNERS = {
    "tl": (0.04, 0.94, "left", "top"),
    "tr": (0.96, 0.94, "right", "top"),
    "bl": (0.04, 0.06, "left", "bottom"),
    "br": (0.96, 0.06, "right", "bottom"),
}


def quietest_corner(true, pred):
    """Pick whichever corner has the least curve height nearby, so the KL
    annotation doesn't sit on top of a peak."""
    n = len(true)
    edge = max(1, n // 6)
    combined = np.maximum(true, pred)
    regions = {
        "l": combined[:edge].max(),
        "r": combined[-edge:].max(),
    }
    # top corners are safe when the curve is low near the top of the y-range
    # at that edge; approximate by just using the edge's max height vs a
    # small threshold relative to the overall peak.
    peak = combined.max()
    side = "l" if regions["l"] <= regions["r"] else "r"
    # prefer top, but drop to bottom if this side's near-edge height is
    # still a large fraction of the peak (curve still high there)
    vpos = "b" if regions[side] > 0.5 * peak else "t"
    key = f"{vpos}{side}"
    return CORNERS[key]


if __name__ == "__main__":
    print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
    embeddings_dict = torch.load(EMBEDDING_PATH)

    bin_centers = None
    true_dists = {}       # example_idx -> true distribution
    preds_by_model = {}   # model_name -> {example_idx: (pred, kl)}

    for name in MODEL_ORDER:
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
        if bin_centers is None:
            bin_centers = (data.bin_edges[:-1] + data.bin_edges[1:]) / 2

        static_dim = data.train.static_features.shape[1]
        output_dim = data.train.targets.shape[1]
        model_configs = build_model_configs(static_dim, output_dim)
        model = model_configs[name]["build"]().to(DEVICE)

        checkpoint_path = BASE_DIR / "checkpoints" / f"best_{name}_{VERSION}.pth"
        test_loader = make_dataloader(
            data.test, batching=model.BATCHING, batch_size=best_params_by_model[name]["batch_size"],
            shuffle=False, embedding_dict=embeddings_dict,
        )
        eval_result = evaluate_model(model, test_loader, device=DEVICE, checkpoint_path=checkpoint_path)

        preds_by_model[name] = {}
        for idx in EXAMPLE_INDICES:
            preds_by_model[name][idx] = (eval_result["preds"][idx], eval_result["kl_scores"][idx])
            if idx not in true_dists:
                true_dists[idx] = eval_result["targets"][idx]
        print(f"{name}: done")

    n_rows = len(MODEL_ORDER)
    n_cols = len(EXAMPLE_INDICES)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.7 * n_cols, 1.5 * n_rows))

    for row, name in enumerate(MODEL_ORDER):
        for col, idx in enumerate(EXAMPLE_INDICES):
            ax = axes[row, col]
            pred, kl = preds_by_model[name][idx]
            true = true_dists[idx]
            ax.plot(bin_centers, true, color="black", linewidth=1.3, alpha=0.6)
            ax.plot(bin_centers, pred, color="crimson", linewidth=1.3)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_alpha(0.3)

            x, y, ha, va = quietest_corner(true, pred)
            ax.text(
                x, y, f"{kl:.2f}".lstrip("0"), transform=ax.transAxes,
                ha=ha, va=va, fontsize=20, color="black", fontweight="bold",
            )
            if col == 0:
                ax.set_ylabel(
                    LABEL_OVERRIDES.get(name, name), fontsize=20, fontweight="bold",
                    rotation=30, ha="right", va="center",
                )

    fig.subplots_adjust(wspace=0, hspace=0)

    legend_handles = [
        mlines.Line2D([], [], color="black", alpha=0.6, linewidth=2, label="True"),
        mlines.Line2D([], [], color="crimson", linewidth=2, label="Predicted"),
    ]
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.55, 1.0),
        ncol=2, fontsize=20, frameon=True, shadow=True,
    )

    plt.tight_layout(rect=[0.05, 0.05, 1, 0.93])
    fig.subplots_adjust(wspace=0, hspace=0)

    # L-shaped axis indicator (bottom-left corner), matching the hand sketch:
    # a vertical arm labeled "Probability" and a horizontal arm labeled
    # "ln(folding time)" meeting at a right angle.
    corner_x, corner_y = 0.045, 0.045
    arm_len = 0.10
    fig.add_artist(mlines.Line2D(
        [corner_x, corner_x], [corner_y, corner_y + arm_len],
        transform=fig.transFigure, color="black", linewidth=1.5,
    ))
    fig.add_artist(mlines.Line2D(
        [corner_x, corner_x + arm_len], [corner_y, corner_y],
        transform=fig.transFigure, color="black", linewidth=1.5,
    ))
    fig.text(
        corner_x - 0.012, corner_y + arm_len / 2, "Probability",
        va="center", ha="right", rotation=90, fontsize=20,
    )
    fig.text(
        corner_x + arm_len / 2, corner_y - 0.012, "ln(folding time)",
        va="top", ha="center", fontsize=20,
    )
    out_path = BASE_DIR / "results" / "example_grid.png"
    fig.savefig(out_path, dpi=200)
    print(f"Saved {out_path}")
