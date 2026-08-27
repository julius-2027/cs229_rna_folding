"""
run_experiments_best_params.py
-------------------------------
Trains every model config (except bilstm_rna_fm_proj) for a fixed number of
epochs using the best lr/weight_decay/batch_size found during tuning
(best_hyperparams.csv, produced by table.py).

Run with:  python run_experiments_best_params.py
"""

import csv
import numpy as np
import json
import os
import pandas as pd
from pathlib import Path
from data import prepare_data, make_dataloader
import models as ms
from train_utils import set_seed, train_model, evaluate_model, plot_loss_curve, plot_prediction_grid
import torch

os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)

BASE_DIR = Path(__file__).resolve().parent
MASK_PATH = None
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
BEST_PARAMS_CSV = BASE_DIR / "best_hyperparams.csv"
SEED = 42
N_BINS = 50
LOCAL_MINIMA_K = 10
DEVICE = "cpu"
EPOCHS = 20
VERSION = "best_params_20ep"
EXCLUDE_MODELS = {"bilstm_rna_fm_proj","naive_baseline","length_baseline"}

# ---------------------------------------------------------------------------
# 1. Load best hyperparameters found during tuning
# ---------------------------------------------------------------------------
best_params_by_model = {}
with open(BEST_PARAMS_CSV) as f:
    for row in csv.DictReader(f):
        if row["model"] in EXCLUDE_MODELS:
            continue
        if row["lr"] == "-":
            # closed-form baseline (naive_baseline/length_baseline): not gradient-trained,
            # has no tuned hyperparameters to load
            continue
        best_params_by_model[row["model"]] = {
            "lr": float(row["lr"]),
            "weight_decay": float(row["weight_decay"]),
            "batch_size": int(row["batch_size"]),
        }

# ---------------------------------------------------------------------------
# 2. Build model configs (mirrors run_experiments.py; lr/epochs overridden below)
# ---------------------------------------------------------------------------
def build_model_configs(static_dim, output_dim):
    return {
        "glm_baseline": {
            "build": lambda: ms.GLMBaseline(static_dim, output_dim),
        },
        "mean_pool_mlp": {
            "build": lambda: ms.MeanPoolMLP(static_dim, output_dim, hidden_size=64),
        },
        "lstm": {
            "build": lambda: ms.DynamicHybridLSTM(
                hidden_size=64, num_layers=1, static_feature_size=static_dim,
                output_size=output_dim, mlp_hidden_size=64, bidirectional=False,
            ),
        },
        "bilstm": {
            "build": lambda: ms.DynamicHybridLSTM(
                hidden_size=64, num_layers=1, static_feature_size=static_dim,
                output_size=output_dim, mlp_hidden_size=64, bidirectional=True,
            ),
        },
        "lstm_rna_fm": {
            "build": lambda: ms.DynamicEmbeddingHybridLSTM(
                hidden_size=64,
                num_layers=1,
                static_feature_size=static_dim,
                output_size=output_dim,
                embedding_dim=640,
                mlp_hidden_size=64,
                bidirectional=True,
            ),
        },
        "struct_lstm_rna_fm": {
            "build": lambda: ms.DynamicEmbeddingHybridLSTMwithStruct(
                hidden_size=64,
                num_layers=1,
                struct_dim=3,
                static_feature_size=static_dim,
                output_size=output_dim,
                embedding_dim=640,
                mlp_hidden_size=64,
                bidirectional=True,
            ),
        },
        "transformer_rna_fm": {
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
        },
        "loc_rna_fm": {
            "build": lambda: ms.RNALocLM(
                embedding_dim=640, cnn_channels=128,
                kernel_sizes=(3, 4, 5), lstm_hidden=128,
                lstm_layers=1, num_heads=8,
                static_feature_size=3, output_size=50,
                mlp_hidden_size=64, dropout=0.3
            ),
        },
        "all_local_mlp": {
            "build": lambda: ms.AllLocalMLP(
                static_feature_size=static_dim, output_size=output_dim,
                hidden_size=64, num_hidden_layers=2, dropout=0.1,
            ),
        },
    }


def run_one_model(name):
    params = best_params_by_model[name]
    is_glm = (name == "glm_baseline")
    all_local_data = (name == "all_local_mlp")
    handpicked_cols = None
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

    set_seed(SEED)
    data = prepare_data(
        DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols,
        bin_edges_path=BIN_EDGES_PATH, data_mask=data_mask,
        local_minima_k=LOCAL_MINIMA_K if all_local_data else None,
    )
    static_dim = data.train.static_features.shape[1]
    output_dim = data.train.targets.shape[1]
    bin_centers = (data.bin_edges[:-1] + data.bin_edges[1:]) / 2

    print(f"Train/Val/Test sizes: {len(data.train.sequences)}/{len(data.val.sequences)}/{len(data.test.sequences)}")

    model_configs = build_model_configs(static_dim, output_dim)
    cfg = model_configs[name]
    version_suffix = f"{name}_{VERSION}"

    print(f"\n{'=' * 20} Running: {name} ({VERSION}) lr={params['lr']} "
          f"weight_decay={params['weight_decay']} batch_size={params['batch_size']} {'=' * 20}")

    set_seed(SEED)
    model = cfg["build"]().to(DEVICE)

    checkpoint_path = f"checkpoints/best_{version_suffix}.pth"
    results_json_path = f"results/metrics_{version_suffix}.json"

    print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
    embeddings_dict = torch.load(EMBEDDING_PATH)

    if data_mask is not None:
        embeddings_dict_aftermask = {}
        for key in embeddings_dict.keys():
            if data_mask['num_peaks'][int(key)]:
                embeddings_dict_aftermask[key] = embeddings_dict[key]
        embeddings_dict = embeddings_dict_aftermask

    train_loader = make_dataloader(
        data.train, batching=model.BATCHING, batch_size=params["batch_size"],
        shuffle=True, embedding_dict=embeddings_dict,
    )
    val_loader = make_dataloader(
        data.val, batching=model.BATCHING, batch_size=params["batch_size"],
        shuffle=False, embedding_dict=embeddings_dict,
    )
    test_loader = make_dataloader(
        data.test, batching=model.BATCHING, batch_size=params["batch_size"],
        shuffle=False, embedding_dict=embeddings_dict,
    )

    history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=params["lr"], weight_decay=params["weight_decay"],
        checkpoint_path=checkpoint_path, device=DEVICE,
        verbose=True, print_every=10,
    )

    eval_result = evaluate_model(model, test_loader, device=DEVICE, checkpoint_path=checkpoint_path)

    run_summary = {
        "model": name,
        "version": VERSION,
        "epochs": EPOCHS,
        "lr": params["lr"],
        "weight_decay": params["weight_decay"],
        "batch_size": params["batch_size"],
        "final_train_loss": history["train_loss"][-1],
        "final_val_loss": history["val_loss"][-1],
        "best_val_loss": min(history["val_loss"]),
        "test_js_mean": eval_result["js_mean"],
        "test_js_median": eval_result["js_median"],
        "history": {
            "train_loss": history["train_loss"],
            "val_loss": history["val_loss"],
        },
    }

    with open(results_json_path, "w") as f:
        json.dump(run_summary, f, indent=4)

    print(f"\nResults saved to {results_json_path}")
    print(pd.DataFrame([run_summary]).drop(columns=["history"]).to_string(index=False))

    import matplotlib.pyplot as plt

    loss_ax = plot_loss_curve(run_summary["history"])
    loss_ax.figure.savefig(f"results/loss_curve_{version_suffix}.png")
    plt.close(loss_ax.figure)

    pred_fig = plot_prediction_grid(bin_centers, eval_result, n_examples=6)
    pred_fig.savefig(f"results/predictions_{version_suffix}.png")
    plt.close(pred_fig)

    return run_summary


if __name__ == "__main__":
    all_models = [m for m in best_params_by_model.keys()]
    print(f"Running models: {all_models}")

    summaries = []
    for name in all_models:
        summaries.append(run_one_model(name))

    print("\n\n===== Summary across all models =====")
    print(pd.DataFrame(summaries).drop(columns=["history"]).to_string(index=False))
