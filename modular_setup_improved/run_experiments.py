
"""
run_experiments.py
-------------------
Driver script modified to train a single targeted model config.
Automatically reloads existing checkpoints to continue training, accumulates
historical training performance in a persistent results file, and tracks
total cumulative epoch count across runs.

Run with:  python run_experiments.py --model bilstm --version v1
"""

import numpy as np
import argparse
import json
import os
import pandas as pd
from pathlib import Path
from data import prepare_data, make_dataloader
import models as ms
# from models import GLMBaseline, MeanPoolMLP, DynamicHybridLSTM, DynamicEmbeddingHybridLSTM
from train_utils import set_seed, train_model, evaluate_model, plot_loss_curve, plot_prediction_grid
import torch
# Ensure output directories exist
os.makedirs("checkpoints", exist_ok=True)
os.makedirs("results", exist_ok=True)


# generic
BASE_DIR = Path(__file__).resolve().parent
#BASE_DIR = BASE_DIR / "../modular_setup_improved" # for modular_setup_improved_kmn
MASK_PATH = BASE_DIR / "mask_2peaks.parquet" # for training on only those examples with 2 peaks
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
SEED = 42
N_BINS = 50
DEVICE = "cpu"  # change to "cuda" if available

# ---------------------------------------------------------------------------
# 1. Parse Command Line Arguments
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train and evaluate a specific RNA folding model.")
parser.add_argument(
    "--model", 
    type=str, 
    required=True, 
    choices=["glm_baseline", "mean_pool_mlp", "bilstm", "bilstm_rna_fm", "bilstm_rna_fm_proj", "embed_transformer"],
    help="Name of the model config to run"
)
parser.add_argument(
    "--version", 
    type=str, 
    default="v1", 
    help="Version tag for saving results (e.g., v1, lr_tuning)"
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# 2. Prepare data ONCE
# ---------------------------------------------------------------------------
set_seed(SEED)
is_glm = (args.model == "glm_baseline")
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

if MASK_PATH is not None:
    data_mask = pd.read_parquet(MASK_PATH)
else:
    data_mask = None

data = prepare_data(DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols, bin_edges_path=BIN_EDGES_PATH, data_mask=data_mask)
static_dim = data.train.static_features.shape[1]
output_dim = data.train.targets.shape[1]
bin_centers = (data.bin_edges[:-1] + data.bin_edges[1:]) / 2

print(f"Train/Val/Test sizes: {len(data.train.sequences)}/{len(data.val.sequences)}/{len(data.test.sequences)}")
print(f"Static feature dim: {static_dim} (Frequencies included: {is_glm}), output dim: {output_dim}")
# ---------------------------------------------------------------------------
# 3. Define the available models mapping
# ---------------------------------------------------------------------------
model_configs = {
    "glm_baseline": {
        "build": lambda: ms.GLMBaseline(static_dim, output_dim),
        "epochs": 50,
        "lr": 1e-2,
    },
    "mean_pool_mlp": {
        "build": lambda: ms.MeanPoolMLP(static_dim, output_dim, hidden_size=64),
        "epochs": 50,
        "lr": 1e-3,
    },
    "bilstm": {
        "build": lambda: ms.DynamicHybridLSTM(
            hidden_size=64, num_layers=1, static_feature_size=static_dim,
            output_size=output_dim, mlp_hidden_size=64, bidirectional=False,
        ),
        "epochs": 15,
        "lr": 1e-3,
    },
    "bilstm_rna_fm": {
        "build": lambda: ms.DynamicEmbeddingHybridLSTM(
            hidden_size=64, 
            num_layers=1, 
            static_feature_size=static_dim,
            output_size=output_dim, 
            embedding_dim=640, # Explicitly configured for RNA-FM dense vectors
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
            embedding_dim=640, # Explicitly configured for RNA-FM dense vectors
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
            embedding_dim = 640, cnn_channels = 128,
            kernel_sizes = (3, 4, 5), lstm_hidden   = 128,
            lstm_layers = 1, num_heads = 8,
            static_feature_size = 3, output_size = 50,
            mlp_hidden_size = 64, dropout = 0.3
        ),
        "epochs": 15,
        "lr": 1e-3,
    }
}

# Fetch chosen config
cfg = model_configs[args.model]
name = args.model
version_suffix = f"{name}_{args.version}"

print(f"\n{'=' * 20} Running: {name} ({args.version}) {'=' * 20}")

# ---------------------------------------------------------------------------
# 4. Initialize Model and Handle Resuming / Checkpointing
# ---------------------------------------------------------------------------
set_seed(SEED)
model = cfg["build"]().to(DEVICE)

# Unique paths for this model + version combo
checkpoint_path = f"checkpoints/best_{version_suffix}.pth"
results_json_path = f"results/metrics_{version_suffix}.json"

# Read historical data if it exists
existing_history = {"train_loss": [], "val_loss": []}
cumulative_epochs = 0

if os.path.exists(checkpoint_path):
    print(f"Found existing checkpoint at {checkpoint_path}. Loading weights to resume training...")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)
        print("Weights successfully restored.")
        
        # Load past results file to inherit continuous loss arrays and epoch count
        if os.path.exists(results_json_path):
            with open(results_json_path, "r") as f:
                past_run = json.load(f)
                # Recover full historical arrays
                if "history" in past_run:
                    existing_history["train_loss"] = past_run["history"].get("train_loss", [])
                    existing_history["val_loss"] = past_run["history"].get("val_loss", [])
                # Recover cumulative epoch tally
                cumulative_epochs = past_run.get("cumulative_epochs", len(existing_history["train_loss"]))
            print(f"Loaded existing history file. Model has trained for {cumulative_epochs} total epochs so far.")
            
    except Exception as e:
        print(f"Warning: Could not reload checkpoint/history ({e}). Starting training from scratch.")
        existing_history = {"train_loss": [], "val_loss": []}
        cumulative_epochs = 0

# ---------------------------------------------------------------------------
# 5. Data Loaders, Train, and Evaluate
# ---------------------------------------------------------------------------
print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
embeddings_dict = torch.load(EMBEDDING_PATH)

if data_mask is not None:
    embeddings_dict_aftermask = {}
    for key in embeddings_dict.keys():
        if data_mask['num_peaks'][int(key)]:
            embeddings_dict_aftermask[key] = embeddings_dict[key]
    embeddings_dict = embeddings_dict_aftermask

train_loader = make_dataloader(data.train, batching=model.BATCHING, batch_size=32, shuffle=True, embedding_dict=embeddings_dict)
val_loader = make_dataloader(data.val, batching=model.BATCHING, batch_size=32, shuffle=False, embedding_dict=embeddings_dict)
test_loader = make_dataloader(data.test, batching=model.BATCHING, batch_size=32, shuffle=False, embedding_dict=embeddings_dict)


#model = cfg["build"]().to(DEVICE) # <-- overwrites the restored model!
new_history = train_model(
    model, train_loader, val_loader,
    epochs=cfg["epochs"], lr=cfg["lr"],
    checkpoint_path=checkpoint_path, device=DEVICE,
    verbose=True, print_every=10,
)

# Load best checkpoint before final evaluation
eval_result = evaluate_model(model, test_loader, device=DEVICE, checkpoint_path=checkpoint_path)

# ---------------------------------------------------------------------------
# 6. Accumulate and Save Combined Results
# ---------------------------------------------------------------------------
# Merge the old historical arrays with the new training run's arrays
combined_train_loss = existing_history["train_loss"] + new_history["train_loss"]
combined_val_loss = existing_history["val_loss"] + new_history["val_loss"]
total_epochs_now = cumulative_epochs + len(new_history["train_loss"])

run_summary = {
    "model": name,
    "version": args.version,
    "current_run_epochs": len(new_history["train_loss"]),
    "cumulative_epochs": total_epochs_now,
    "final_train_loss": new_history["train_loss"][-1],
    "final_val_loss": new_history["val_loss"][-1],
    "best_val_loss_all_time": min(combined_val_loss),
    "test_js_mean": eval_result["js_mean"],
    "test_js_median": eval_result["js_median"],
    "history": {
        "train_loss": combined_train_loss,
        "val_loss": combined_val_loss
    }
}

with open(results_json_path, "w") as f:
    json.dump(run_summary, f, indent=4)

print(f"\nResults updated in {results_json_path}")
print(f"Total Session Lifetime Epochs: {total_epochs_now}")
print(pd.DataFrame([run_summary]).drop(columns=["history"]).to_string(index=False))

# ---------------------------------------------------------------------------
# 7. Visualization
# ---------------------------------------------------------------------------
# Plot the full aggregated historical timeline
plot_loss_curve(run_summary["history"])
plot_prediction_grid(bin_centers, eval_result, n_examples=6)