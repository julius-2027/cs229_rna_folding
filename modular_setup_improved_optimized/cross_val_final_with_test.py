"""
cross_val_final_with_test.py
-----------------------------
Runs every tuned model (best lr/weight_decay/batch_size from best_hyperparams.csv)
through k-fold cross-validation with a held-out test set, then trains one more
model on the full train+val split and scores it on that held-out test set.

Split: a 15% test set is carved off first (matching the 72/12/15 train/val/test
proportions that data.prepare_data()'s default test_size=0.15/val_size=0.15
produces). The remaining 85% is 5-folded, so each fold's val split is ~17% of
the total and each fold's train split is ~68% of the total. The test set
itself is never touched until the final fit.

Metric is KL divergence throughout (nn.KLDivLoss(reduction="batchmean")), the
same loss train_model already optimizes -- not JS divergence.

For each model:
  1. 5-fold CV over the train+val 85%: fresh model per fold, trained with that
     model's tuned hyperparameters, scored by KL-divergence val loss on its
     fold's val split (the best epoch's val loss, i.e. what the checkpoint
     would have saved). Reported as mean +/- std across folds.
  2. One final model, trained on the full 85% (with a small carved-out slice
     for checkpoint/scheduler monitoring, same 0.15 fraction), evaluated once
     on the untouched 15% test set with the same KL-divergence loss.

Writes cross_val_results/cross_val_final_with_test.csv with columns:
    cross_val_kl (mean +/- std), test_kl, model

Run with:  python cross_val_final_with_test.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold

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
from train_utils import set_seed, train_model, _run_epoch

BASE_DIR = Path(__file__).resolve().parent
MASK_PATH = None
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
BEST_PARAMS_CSV = BASE_DIR / "best_hyperparams.csv"
OUTPUT_DIR = BASE_DIR / "cross_val_results"
OUTPUT_CSV = OUTPUT_DIR / "cross_val_final_with_test.csv"

SEED = 42
N_BINS = 50
LOCAL_MINIMA_K = 10
DEVICE = "cpu"
EPOCHS = 20
K = 5
TEST_SIZE = 0.15
VAL_SIZE = 0.15  # fraction of the 85% train+val used to monitor the final fit

MODEL_ORDER = [
    "glm_baseline",
    "mean_pool_mlp",
    "all_local_mlp",
    "lstm",
    "lstm_rna_fm",
    "struct_lstm_rna_fm",
    "transformer_rna_fm",
    "loc_rna_fm",
]


def build_model_configs(static_dim, output_dim):
    return {
        "glm_baseline": {"build": lambda: ms.GLMBaseline(static_dim, output_dim)},
        "mean_pool_mlp": {"build": lambda: ms.MeanPoolMLP(static_dim, output_dim, hidden_size=64)},
        "lstm": {
            "build": lambda: ms.DynamicHybridLSTM(
                hidden_size=64, num_layers=1, static_feature_size=static_dim,
                output_size=output_dim, mlp_hidden_size=64, bidirectional=False,
            ),
        },
        "lstm_rna_fm": {
            "build": lambda: ms.DynamicEmbeddingHybridLSTM(
                hidden_size=64, num_layers=1, static_feature_size=static_dim,
                output_size=output_dim, embedding_dim=640, mlp_hidden_size=64,
                bidirectional=True,
            ),
        },
        "struct_lstm_rna_fm": {
            "build": lambda: ms.DynamicEmbeddingHybridLSTMwithStruct(
                hidden_size=64, num_layers=1, struct_dim=3, static_feature_size=static_dim,
                output_size=output_dim, embedding_dim=640, mlp_hidden_size=64,
                bidirectional=True,
            ),
        },
        "transformer_rna_fm": {
            "build": lambda: ms.Embed_Transformer(
                embedding_dim=640, projection_dim=64, num_heads=4, num_layers=1,
                dropout=0.5, static_feature_size=static_dim, output_size=output_dim,
                mlp_hidden_size=64,
            ),
        },
        "loc_rna_fm": {
            "build": lambda: ms.RNALocLM(
                embedding_dim=640, cnn_channels=128, kernel_sizes=(3, 4, 5),
                lstm_hidden=128, lstm_layers=1, num_heads=8,
                static_feature_size=3, output_size=50, mlp_hidden_size=64, dropout=0.3,
            ),
        },
        "all_local_mlp": {
            "build": lambda: ms.AllLocalMLP(
                static_feature_size=static_dim, output_size=output_dim,
                hidden_size=64, num_hidden_layers=2, dropout=0.1,
            ),
        },
    }


def load_best_params():
    params = {}
    with open(BEST_PARAMS_CSV) as f:
        for row in csv.DictReader(f):
            params[row["model"]] = {
                "lr": float(row["lr"]),
                "weight_decay": float(row["weight_decay"]),
                "batch_size": int(row["batch_size"]),
            }
    return params


def run_one_model(name, params, embeddings_dict):
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

    df = load_dataset(DATA_PATH)
    if all_local_data:
        df = df.copy()
        df[local_minima_feature_names(LOCAL_MINIMA_K)] = build_local_minima_features(df, k=LOCAL_MINIMA_K)
        handpicked_cols = list(handpicked_cols) + local_minima_feature_names(LOCAL_MINIMA_K)

    bin_edges = np.load(BIN_EDGES_PATH)
    y, bin_edges = build_histogram_targets(df, n_bins=N_BINS, bin_edges=bin_edges)
    X = keep_handpicked_columns(df, handpicked_cols=handpicked_cols)

    static_dim = len(handpicked_cols)
    output_dim = y.shape[1]

    print(f"\n{'=' * 20} {name} {'=' * 20}")
    print(f"Full dataset size: {len(X)} | static_dim={static_dim} output_dim={output_dim}")

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SEED
    )

    model_configs = build_model_configs(static_dim, output_dim)
    cfg = model_configs[name]

    # ---- 1. K-fold CV over the train+val 85% -----------------------------
    kfold = KFold(n_splits=K, shuffle=True, random_state=SEED)
    fold_kl = []
    for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(X_trainval), start=1):
        X_train, X_val = X_trainval.iloc[train_idx], X_trainval.iloc[val_idx]
        y_train, y_val = y_trainval[train_idx], y_trainval[val_idx]

        scaler = StandardScaler()
        static_train = scaler.fit_transform(X_train[handpicked_cols])
        static_val = scaler.transform(X_val[handpicked_cols])

        train_bundle = _bucket(X_train, static_train, y_train)
        val_bundle = _bucket(X_val, static_val, y_val)

        set_seed(SEED)
        model = cfg["build"]().to(DEVICE)

        train_loader = make_dataloader(
            train_bundle, batching=model.BATCHING, batch_size=params["batch_size"],
            shuffle=True, embedding_dict=embeddings_dict,
        )
        val_loader = make_dataloader(
            val_bundle, batching=model.BATCHING, batch_size=params["batch_size"],
            shuffle=False, embedding_dict=embeddings_dict,
        )

        history = train_model(
            model, train_loader, val_loader,
            epochs=EPOCHS, lr=params["lr"], weight_decay=params["weight_decay"],
            checkpoint_path=None, device=DEVICE, verbose=False,
        )

        best_fold_val_kl = min(history["val_loss"])
        fold_kl.append(best_fold_val_kl)
        print(f"  Fold {fold_idx}/{K} | best val KL: {best_fold_val_kl:.4f}")

    cv_kl_mean = float(np.mean(fold_kl))
    cv_kl_std = float(np.std(fold_kl))
    print(f"  CV KL mean +/- std: {cv_kl_mean:.4f} +/- {cv_kl_std:.4f}")

    # ---- 2. Final fit on the full 85% (with a small monitoring slice),
    #         scored once on the untouched 15% test set ---------------------
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=VAL_SIZE, random_state=SEED
    )

    scaler = StandardScaler()
    static_train = scaler.fit_transform(X_train[handpicked_cols])
    static_val = scaler.transform(X_val[handpicked_cols])
    static_test = scaler.transform(X_test[handpicked_cols])

    train_bundle = _bucket(X_train, static_train, y_train)
    val_bundle = _bucket(X_val, static_val, y_val)
    test_bundle = _bucket(X_test, static_test, y_test)

    set_seed(SEED)
    final_model = cfg["build"]().to(DEVICE)

    checkpoint_path = f"checkpoints/best_{name}_cv_final.pth"
    train_loader = make_dataloader(
        train_bundle, batching=final_model.BATCHING, batch_size=params["batch_size"],
        shuffle=True, embedding_dict=embeddings_dict,
    )
    val_loader = make_dataloader(
        val_bundle, batching=final_model.BATCHING, batch_size=params["batch_size"],
        shuffle=False, embedding_dict=embeddings_dict,
    )
    test_loader = make_dataloader(
        test_bundle, batching=final_model.BATCHING, batch_size=params["batch_size"],
        shuffle=False, embedding_dict=embeddings_dict,
    )

    train_model(
        final_model, train_loader, val_loader,
        epochs=EPOCHS, lr=params["lr"], weight_decay=params["weight_decay"],
        checkpoint_path=checkpoint_path, device=DEVICE, verbose=False,
    )

    final_model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    criterion = nn.KLDivLoss(reduction="batchmean")
    test_kl = _run_epoch(final_model, test_loader, criterion, optimizer=None, device=DEVICE)
    print(f"  Test KL (final model): {test_kl:.4f}")

    return {
        "model": name,
        "fold_kl": fold_kl,
        "cv_kl_mean": cv_kl_mean,
        "cv_kl_std": cv_kl_std,
        "test_kl": test_kl,
    }


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_params = load_best_params()

    print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
    embeddings_dict = torch.load(EMBEDDING_PATH)

    results = []
    for name in MODEL_ORDER:
        if name not in best_params:
            print(f"Skipping {name}: no tuned hyperparameters found in {BEST_PARAMS_CSV}")
            continue
        results.append(run_one_model(name, best_params[name], embeddings_dict))

    with open(OUTPUT_DIR / "cross_val_final_with_test_raw.json", "w") as f:
        json.dump(results, f, indent=4)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["cross_val_kl", "test_kl", "model"])
        for r in results:
            cv_str = f"{r['cv_kl_mean']:.4f} +/- {r['cv_kl_std']:.4f}"
            writer.writerow([cv_str, f"{r['test_kl']:.4f}", r["model"]])

    print(f"\nWrote {len(results)} rows to {OUTPUT_CSV}")
    print(pd.DataFrame(results)[["model", "cv_kl_mean", "cv_kl_std", "test_kl"]].to_string(index=False))
