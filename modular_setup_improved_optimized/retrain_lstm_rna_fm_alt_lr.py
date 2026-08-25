"""
retrain_lstm_rna_fm_alt_lr.py
-------------------------------
Diagnostic: lstm_rna_fm's tuned hyperparameters (lr=0.0214, wd=7.1e-6,
batch=128) came from the same fixed-seed Optuna grid as glm_baseline /
mean_pool_mlp -- its val loss history is noisy (bounces 0.40-0.44 for most
of training) rather than smoothly decreasing, suggesting that lr is too
aggressive for a BiLSTM over 640-dim embeddings specifically.

This retrains lstm_rna_fm from scratch using struct_lstm_rna_fm's tuned
hyperparameters instead (lr=0.00053, wd=1.5e-5, batch=16) -- same
architecture, same data, same 20 epochs -- to test whether the gap to
struct_lstm_rna_fm is a learning-rate artifact rather than an architectural
one.

Run with: python retrain_lstm_rna_fm_alt_lr.py
"""

import json
from pathlib import Path

import pandas as pd
import torch

from run_experiments_best_params import build_model_configs
from data import prepare_data, make_dataloader
from train_utils import set_seed, train_model, evaluate_model

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
SEED = 42
N_BINS = 50
DEVICE = "cpu"
EPOCHS = 20
MODEL_NAME = "lstm_rna_fm"

# struct_lstm_rna_fm's tuned hyperparameters, borrowed for this test
ALT_LR = 0.0005342937261279777
ALT_WEIGHT_DECAY = 1.461896279370496e-05
ALT_BATCH_SIZE = 16

if __name__ == "__main__":
    handpicked_cols = ["gc_content", "mfe", "n_local_minima"]

    set_seed(SEED)
    data = prepare_data(
        DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols,
        bin_edges_path=BIN_EDGES_PATH, data_mask=None,
    )
    static_dim = data.train.static_features.shape[1]
    output_dim = data.train.targets.shape[1]

    model_configs = build_model_configs(static_dim, output_dim)
    model = model_configs[MODEL_NAME]["build"]().to(DEVICE)

    print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
    embeddings_dict = torch.load(EMBEDDING_PATH)

    train_loader = make_dataloader(
        data.train, batching=model.BATCHING, batch_size=ALT_BATCH_SIZE,
        shuffle=True, embedding_dict=embeddings_dict,
    )
    val_loader = make_dataloader(
        data.val, batching=model.BATCHING, batch_size=ALT_BATCH_SIZE,
        shuffle=False, embedding_dict=embeddings_dict,
    )
    test_loader = make_dataloader(
        data.test, batching=model.BATCHING, batch_size=ALT_BATCH_SIZE,
        shuffle=False, embedding_dict=embeddings_dict,
    )

    checkpoint_path = "checkpoints/best_lstm_rna_fm_alt_lr.pth"
    set_seed(SEED)
    history = train_model(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=ALT_LR, weight_decay=ALT_WEIGHT_DECAY,
        checkpoint_path=checkpoint_path, device=DEVICE,
        verbose=True, print_every=5,
    )

    eval_result = evaluate_model(model, test_loader, device=DEVICE, checkpoint_path=checkpoint_path)

    print("\nval_loss per epoch:", [round(v, 4) for v in history["val_loss"]])
    print(f"\nfinal_train_loss={history['train_loss'][-1]:.4f}  final_val_loss={history['val_loss'][-1]:.4f}"
          f"  best_val_loss={min(history['val_loss']):.4f}  test_js_mean={eval_result['js_mean']:.4f}")

    with open("results/metrics_lstm_rna_fm_alt_lr.json", "w") as f:
        json.dump({
            "model": MODEL_NAME,
            "version": "alt_lr_from_struct_lstm_rna_fm",
            "lr": ALT_LR,
            "weight_decay": ALT_WEIGHT_DECAY,
            "batch_size": ALT_BATCH_SIZE,
            "final_train_loss": history["train_loss"][-1],
            "final_val_loss": history["val_loss"][-1],
            "best_val_loss": min(history["val_loss"]),
            "test_js_mean": eval_result["js_mean"],
            "history": history,
        }, f, indent=4)
    print("\nSaved to results/metrics_lstm_rna_fm_alt_lr.json")
