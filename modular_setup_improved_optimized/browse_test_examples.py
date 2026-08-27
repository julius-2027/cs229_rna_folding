"""
browse_test_examples.py
--------------------------
Plots a big grid of ground-truth target distributions from the held-out test
set (same prepare_data pipeline as everywhere else) so you can browse and
pick out interesting/illustrative examples by index.

Run with: python browse_test_examples.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from data import prepare_data
from train_utils import set_seed

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
SEED = 42
N_BINS = 50
N_EXAMPLES = 100
NCOLS = 10

if __name__ == "__main__":
    set_seed(SEED)
    data = prepare_data(
        DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=["gc_content", "mfe", "n_local_minima"],
        bin_edges_path=BIN_EDGES_PATH, data_mask=None,
    )
    bin_centers = (data.bin_edges[:-1] + data.bin_edges[1:]) / 2
    targets = data.test.targets

    n = min(N_EXAMPLES, len(targets))
    nrows = (n + NCOLS - 1) // NCOLS

    fig, axes = plt.subplots(nrows, NCOLS, figsize=(2.2 * NCOLS, 1.8 * nrows))
    axes = np.array(axes).reshape(-1)

    for i, ax in enumerate(axes[:n]):
        ax.plot(bin_centers, targets[i], color="black", linewidth=1)
        ax.set_title(f"#{i}", fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_alpha(0.3)

    for ax in axes[n:]:
        ax.axis("off")

    fig.suptitle(f"Test-set ground truth distributions (first {n} of {len(targets)})", fontsize=14, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = BASE_DIR / "results" / "test_examples_browse.png"
    fig.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
