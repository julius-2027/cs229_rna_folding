"""
cross_val.py
-------------
K-fold cross-validation driver for a single model config from
run_experiments.py.

Reuses data.py's building blocks (load_dataset, build_histogram_targets,
keep_handpicked_columns, build_local_minima_features, _bucket, make_dataloader)
instead of prepare_data()'s single train/val/test split, since here the
splitting is driven by sklearn's KFold over the whole dataset. For each fold,
a fresh model instance (same build/epochs/lr as run_experiments.py) is
trained on that fold's train split and evaluated on its held-out val split
every epoch via train_utils.train_model. The per-epoch train/val loss
histories are then aggregated across folds (mean +/- std at each epoch) and
plotted with error bars, since a single train/val split can make a model
look better or worse than it really is just by chance.

Run with:  python cross_val.py --model bilstm --version v1 --k 5
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

import models as ms
from data import (
    _bucket,
    build_histogram_targets,
    build_local_minima_features,
    keep_handpicked_columns,
    load_dataset,
    local_minima_feature_names,
    make_dataloader,
)
from train_utils import set_seed, train_model

# generic
BASE_DIR = Path(__file__).resolve().parent
MASK_PATH = None #BASE_DIR / "mask_2peaks.parquet"  # for training on only those examples with 2 peaks
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
OUTPUT_DIR = BASE_DIR / "cross_val_results"
SEED = 42
N_BINS = 50
LOCAL_MINIMA_K = 10  # how many rank-ordered local minima to featurize for all_local_MLP
DEVICE = "cpu"  # change to "cuda" if available


def plot_cv_loss(train_mean, train_std, val_mean, val_std, model_name, version, k, out_path):
    """Mean train/val loss per epoch across folds, with +/- 1 std error bars at every point."""
    epochs_axis = np.arange(1, len(train_mean) + 1)

    surface = "#fcfcfb"
    ink_primary = "#0b0b0b"
    ink_secondary = "#52514e"
    ink_muted = "#898781"
    grid_color = "#e1e0d9"
    axis_color = "#c3c2b7"
    color_train = "#2a78d6"  # categorical slot 1 (blue)
    color_val = "#eb6834"    # categorical slot 2 (orange)

    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(surface)
    ax.set_facecolor(surface)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color=grid_color, linewidth=1)

    ax.errorbar(
        epochs_axis, train_mean, yerr=train_std,
        fmt="-o", color=color_train, ecolor=color_train, elinewidth=1,
        capsize=3, capthick=1, markersize=4,
        markerfacecolor=color_train, markeredgecolor=surface, markeredgewidth=1,
        linewidth=2, label="Train loss",
    )
    ax.errorbar(
        epochs_axis, val_mean, yerr=val_std,
        fmt="-o", color=color_val, ecolor=color_val, elinewidth=1,
        capsize=3, capthick=1, markersize=4,
        markerfacecolor=color_val, markeredgecolor=surface, markeredgewidth=1,
        linewidth=2, label="Val loss",
    )

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(axis_color)
        ax.spines[spine].set_linewidth(1)

    ax.tick_params(colors=ink_muted, labelsize=10)
    ax.set_xlabel("Epoch", fontsize=12, color=ink_muted)
    ax.set_ylabel("KL-Divergence Loss", fontsize=12, color=ink_muted)
    ax.set_title(
        f"{k}-Fold Cross-Validation Loss — {model_name} ({version})",
        fontsize=14, fontweight="bold", color=ink_primary, pad=14,
    )

    legend = ax.legend(loc="upper right", frameon=False, fontsize=10)
    for text in legend.get_texts():
        text.set_color(ink_secondary)

    fig.tight_layout()
    fig.savefig(out_path, dpi=300, facecolor=surface, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 1. Parse Command Line Arguments - same knobs as run_experiments.py, plus --k
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="K-fold cross-validate an RNA folding model.")
parser.add_argument(
    "--model",
    type=str,
    required=True,
    choices=["glm_baseline", "mean_pool_mlp", "bilstm", "bilstm_rna_fm", "bilstm_rna_fm_with_struct", "bilstm_rna_fm_proj", "embed_transformer", "rna_loc", "all_local_MLP"],
    help="Name of the model config to run"
)
parser.add_argument(
    "--version",
    type=str,
    default="v1",
    help="Version tag for saving results (e.g., v1, lr_tuning)"
)
parser.add_argument(
    "--k",
    type=int,
    default=5,
    help="Number of folds for k-fold cross-validation"
)
args = parser.parse_args()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 2. Load the full dataset ONCE - same steps prepare_data() in data.py uses,
#    minus the train/val/test split (KFold below does the splitting instead).
# ---------------------------------------------------------------------------
set_seed(SEED)
is_glm = (args.model == "glm_baseline")
all_local_data = (args.model == "all_local_MLP")
if is_glm:
    handpicked_cols = [
        "mfe", "n_local_minima", "gc_content",
        "freq_A", "freq_U", "freq_G", "freq_C",
        "freq_AA", "freq_AU", "freq_AG", "freq_AC",
        "freq_UA", "freq_UU", "freq_UG", "freq_UC",
        "freq_GA", "freq_GU", "freq_GG", "freq_GC",
        "freq_CA", "freq_CU", "freq_CG", "freq_CC"
    ]
else:
    handpicked_cols = ["gc_content", "mfe", "n_local_minima"]

data_mask = pd.read_parquet(MASK_PATH) if MASK_PATH is not None else None

df = load_dataset(DATA_PATH)
if data_mask is not None:
    df = df[np.array(data_mask["num_peaks"])]
if all_local_data:
    df = df.copy()
    df[local_minima_feature_names(LOCAL_MINIMA_K)] = build_local_minima_features(df, k=LOCAL_MINIMA_K)
    handpicked_cols = list(handpicked_cols) + local_minima_feature_names(LOCAL_MINIMA_K)

bin_edges = np.load(BIN_EDGES_PATH)
y, bin_edges = build_histogram_targets(df, n_bins=N_BINS, bin_edges=bin_edges)
X = keep_handpicked_columns(df, handpicked_cols=handpicked_cols)

static_dim = len(handpicked_cols)
output_dim = y.shape[1]

print(f"Full dataset size (all folds): {len(X)}")
print(f"Static feature dim: {static_dim} (Frequencies included: {is_glm}), output dim: {output_dim}")

# ---------------------------------------------------------------------------
# 3. Define the available models mapping - identical build/epochs/lr to
#    run_experiments.py, so CV numbers stay comparable to a single run.
# ---------------------------------------------------------------------------
model_configs = {
    "glm_baseline": {
        "build": lambda: ms.GLMBaseline(static_dim, output_dim),
        "epochs": 15,
        "lr": 1e-2,
    },
    "mean_pool_mlp": {
        "build": lambda: ms.MeanPoolMLP(static_dim, output_dim, hidden_size=64),
        "epochs": 15,
        "lr": 1e-3,
    },
    "bilstm": {
        "build": lambda: ms.DynamicHybridLSTM(
            hidden_size=64, num_layers=1, static_feature_size=static_dim,
            output_size=output_dim, mlp_hidden_size=64, bidirectional=False,
        ),
        "epochs": 45,
        "lr": 1e-3,
    },
    "bilstm_rna_fm": {
        "build": lambda: ms.DynamicEmbeddingHybridLSTM(
            hidden_size=64,
            num_layers=1,
            static_feature_size=static_dim,
            output_size=output_dim,
            embedding_dim=640,  # Explicitly configured for RNA-FM dense vectors
            mlp_hidden_size=64,
            bidirectional=True,
        ),
        "epochs": 15,
        "lr": 1e-3,
    },
    "bilstm_rna_fm_with_struct": {
        "build": lambda: ms.DynamicEmbeddingHybridLSTMwithStruct(
            hidden_size=64,
            num_layers=1,
            struct_dim=3,  # Explicitly configured for structure features
            static_feature_size=static_dim,
            output_size=output_dim,
            embedding_dim=640,  # Explicitly configured for RNA-FM dense vectors
            mlp_hidden_size=64,
            bidirectional=True,
        ),
        "epochs": 15,
        "lr": 1e-3,
    },
    "bilstm_rna_fm_proj": {
        "build": lambda: ms.DynamicEmbeddingHybridLSTM_proj(
            hidden_size=64,
            num_layers=1,
            static_feature_size=static_dim,
            output_size=output_dim,
            embedding_dim=640,  # Explicitly configured for RNA-FM dense vectors
            mlp_hidden_size=64,
            bidirectional=True,
            projection_dim=64,  # Optional projection layer dimension
        ),
        "epochs": 15,
        "lr": 1e-3,
    },
    "embed_transformer": {
        "build": lambda: ms.Embed_Transformer(
            embedding_dim=640,
            projection_dim=64,
            num_heads=4,
            num_layers=1,
            dropout=0.5,
            static_feature_size=static_dim,
            output_size=output_dim,
            mlp_hidden_size=64
        ),
        "epochs": 15,
        "lr": 1e-3,
    },
    "rna_loc": {
        "build": lambda: ms.RNALocLM(
            embedding_dim=640, cnn_channels=128,
            kernel_sizes=(3, 4, 5), lstm_hidden=128,
            lstm_layers=1, num_heads=8,
            static_feature_size=3, output_size=50,
            mlp_hidden_size=64, dropout=0.3
        ),
        "epochs": 15,
        "lr": 1e-3,
    },
    "all_local_MLP": {
        "build": lambda: ms.AllLocalMLP(
            static_feature_size=static_dim, output_size=output_dim,
            hidden_size=64, num_hidden_layers=2, dropout=0.1,
        ),
        "epochs": 15,
        "lr": 1e-3,
    },
}

cfg = model_configs[args.model]
name = args.model
version_suffix = f"{name}_{args.version}"

print(f"\n{'=' * 20} {args.k}-Fold CV: {name} ({args.version}) {'=' * 20}")

# ---------------------------------------------------------------------------
# 4. Load embedding dictionary ONCE, masked the same way run_experiments.py
#    masks it (every model's dataloader expects an embeds tensor per row,
#    even models that ignore it in forward()).
# ---------------------------------------------------------------------------
print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
embeddings_dict = torch.load(EMBEDDING_PATH)
if data_mask is not None:
    embeddings_dict = {
        key: emb for key, emb in embeddings_dict.items()
        if data_mask["num_peaks"][int(key)]
    }

# ---------------------------------------------------------------------------
# 5. K-Fold cross-validation: fresh model + fresh scaler per fold
# ---------------------------------------------------------------------------
kfold = KFold(n_splits=args.k, shuffle=True, random_state=SEED)
fold_histories = []

for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X), start=1):
    print(f"\n--- Fold {fold_idx}/{args.k} ---")

    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    scaler = StandardScaler()
    static_train = scaler.fit_transform(X_train[handpicked_cols])
    static_val = scaler.transform(X_val[handpicked_cols])

    train_bundle = _bucket(X_train, static_train, y_train)
    val_bundle = _bucket(X_val, static_val, y_val)

    set_seed(SEED)
    model = cfg["build"]().to(DEVICE)

    train_loader = make_dataloader(train_bundle, batching=model.BATCHING, batch_size=32, shuffle=True, embedding_dict=embeddings_dict)
    val_loader = make_dataloader(val_bundle, batching=model.BATCHING, batch_size=32, shuffle=False, embedding_dict=embeddings_dict)

    history = train_model(
        model, train_loader, val_loader,
        epochs=cfg["epochs"], lr=cfg["lr"],
        checkpoint_path=None, device=DEVICE,
        verbose=True, print_every=10,
    )
    fold_histories.append(history)
    print(f"Fold {fold_idx}/{args.k} | Final train: {history['train_loss'][-1]:.4f} | Final val: {history['val_loss'][-1]:.4f}")

# ---------------------------------------------------------------------------
# 6. Aggregate across folds (mean +/- std at each epoch)
# ---------------------------------------------------------------------------
train_losses = np.array([h["train_loss"] for h in fold_histories])
val_losses = np.array([h["val_loss"] for h in fold_histories])

train_mean, train_std = train_losses.mean(axis=0), train_losses.std(axis=0)
val_mean, val_std = val_losses.mean(axis=0), val_losses.std(axis=0)

summary_df = pd.DataFrame([
    {
        "fold": i + 1,
        "final_train_loss": fold_histories[i]["train_loss"][-1],
        "final_val_loss": fold_histories[i]["val_loss"][-1],
        "best_val_loss": min(fold_histories[i]["val_loss"]),
    }
    for i in range(args.k)
])
print("\nPer-fold summary:")
print(summary_df.to_string(index=False))
print(f"\nFinal-epoch val loss across folds: {val_mean[-1]:.4f} +/- {val_std[-1]:.4f}")

# ---------------------------------------------------------------------------
# 7. Save aggregated metrics + the loss-over-epochs plot to a new subfolder
# ---------------------------------------------------------------------------
metrics_path = OUTPUT_DIR / f"cross_val_metrics_{version_suffix}_k{args.k}.json"
with open(metrics_path, "w") as f:
    json.dump({
        "model": name,
        "version": args.version,
        "k": args.k,
        "epochs": cfg["epochs"],
        "lr": cfg["lr"],
        "fold_histories": fold_histories,
        "train_loss_mean": train_mean.tolist(),
        "train_loss_std": train_std.tolist(),
        "val_loss_mean": val_mean.tolist(),
        "val_loss_std": val_std.tolist(),
    }, f, indent=4)
print(f"\nSaved per-fold + aggregated metrics to {metrics_path}")

plot_path = OUTPUT_DIR / f"cross_val_loss_{version_suffix}_k{args.k}.png"
plot_cv_loss(train_mean, train_std, val_mean, val_std, name, args.version, args.k, plot_path)
print(f"Saved cross-validation loss plot to {plot_path}")
