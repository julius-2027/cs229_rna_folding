"""
tune.py
-------
Optuna-based hyperparameter search for a single model config from
run_experiments.py / cross_val.py.

Searches the universal training hyperparameters shared by every model
(learning rate, weight decay, batch size) since train_model's
ReduceLROnPlateau already adapts the LR during training, and architecture
hyperparameters are model-specific enough that a shared search space isn't
worth the added complexity here. Each trial trains on a single train/val
split (data.prepare_data) rather than full k-fold, since that gives more
trials for the same compute budget; re-verify the winning config with
cross_val.py once you have it.

Run with:  python tune.py --model lstm --trials 20 --epochs 10
"""

import argparse
import json
from pathlib import Path

import numpy as np
import optuna

import models as ms
from data import make_dataloader, prepare_data
from train_utils import set_seed, train_model

BASE_DIR = Path(__file__).resolve().parent
MASK_PATH = None
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
OUTPUT_DIR = BASE_DIR / "tune_results"
SEED = 42
N_BINS = 50
LOCAL_MINIMA_K = 10
DEVICE = "cpu"


def build_model_configs(static_dim, output_dim):
    """Same build/epochs as cross_val.py's model_configs, minus lr (that's what we're tuning)."""
    return {
        "glm_baseline": {
            "build": lambda: ms.GLMBaseline(static_dim, output_dim),
            "epochs": 15,
        },
        "mean_pool_mlp": {
            "build": lambda: ms.MeanPoolMLP(static_dim, output_dim, hidden_size=64),
            "epochs": 15,
        },
        "lstm": {
            "build": lambda: ms.DynamicHybridLSTM(
                hidden_size=64, num_layers=1, static_feature_size=static_dim,
                output_size=output_dim, mlp_hidden_size=64, bidirectional=False,
            ),
            "epochs": 45,
        },
        "bilstm": {
            "build": lambda: ms.DynamicHybridLSTM(
                hidden_size=64, num_layers=1, static_feature_size=static_dim,
                output_size=output_dim, mlp_hidden_size=64, bidirectional=True,
            ),
            "epochs": 45,
        },
        "lstm_rna_fm": {
            "build": lambda: ms.DynamicEmbeddingHybridLSTM(
                hidden_size=64, num_layers=1, static_feature_size=static_dim,
                output_size=output_dim, embedding_dim=640, mlp_hidden_size=64,
                bidirectional=True,
            ),
            "epochs": 15,
        },
        "struct_lstm_rna_fm": {
            "build": lambda: ms.DynamicEmbeddingHybridLSTMwithStruct(
                hidden_size=64, num_layers=1, struct_dim=3, static_feature_size=static_dim,
                output_size=output_dim, embedding_dim=640, mlp_hidden_size=64,
                bidirectional=True,
            ),
            "epochs": 15,
        },
        "bilstm_rna_fm_proj": {
            "build": lambda: ms.DynamicEmbeddingHybridLSTM_proj(
                hidden_size=64, num_layers=1, static_feature_size=static_dim,
                output_size=output_dim, embedding_dim=640, mlp_hidden_size=64,
                bidirectional=True, projection_dim=64,
            ),
            "epochs": 15,
        },
        "transformer_rna_fm": {
            "build": lambda: ms.Embed_Transformer(
                embedding_dim=640, projection_dim=64, num_heads=4, num_layers=1,
                dropout=0.5, static_feature_size=static_dim, output_size=output_dim,
                mlp_hidden_size=64,
            ),
            "epochs": 15,
        },
        "loc_rna_fm": {
            "build": lambda: ms.RNALocLM(
                embedding_dim=640, cnn_channels=128, kernel_sizes=(3, 4, 5),
                lstm_hidden=128, lstm_layers=1, num_heads=8,
                static_feature_size=3, output_size=50, mlp_hidden_size=64, dropout=0.3,
            ),
            "epochs": 15,
        },
        "all_local_mlp": {
            "build": lambda: ms.AllLocalMLP(
                static_feature_size=static_dim, output_size=output_dim,
                hidden_size=64, num_hidden_layers=2, dropout=0.1,
            ),
            "epochs": 15,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Tune lr/weight_decay/batch_size for an RNA folding model.")
    parser.add_argument(
        "--model", type=str, required=True,
        choices=["glm_baseline", "mean_pool_mlp", "lstm", "bilstm", "lstm_rna_fm", "struct_lstm_rna_fm",
                 "bilstm_rna_fm_proj", "transformer_rna_fm", "loc_rna_fm", "all_local_mlp"],
        help="Name of the model config to tune",
    )
    parser.add_argument("--trials", type=int, default=20, help="Max number of Optuna trials")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per trial (kept short; full run_experiments.py epoch counts are for the final training run, not search)")
    parser.add_argument("--version", type=str, default="v1", help="Version tag for saving results")
    parser.add_argument("--min_improvement", type=float, default=0.005, help="Stop early if the best val loss hasn't improved by at least this much over --patience trials")
    parser.add_argument("--patience", type=int, default=5, help="Number of trials to look back over when checking --min_improvement")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)

    is_glm = (args.model == "glm_baseline")
    all_local_data = (args.model == "all_local_mlp")
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

    local_minima_k = LOCAL_MINIMA_K if all_local_data else None

    prepared = prepare_data(
        DATA_PATH,
        handpicked_cols=handpicked_cols,
        n_bins=N_BINS,
        bin_edges_path=BIN_EDGES_PATH,
        local_minima_k=local_minima_k,
    )
    static_dim = len(prepared.handpicked_cols)
    output_dim = prepared.train.targets.shape[1]

    print(f"Train/val sizes: {len(prepared.train.sequences)} / {len(prepared.val.sequences)}")
    print(f"Static feature dim: {static_dim}, output dim: {output_dim}")

    import torch
    print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
    embeddings_dict = torch.load(EMBEDDING_PATH)

    cfg = build_model_configs(static_dim, output_dim)[args.model]
    name = args.model

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        set_seed(SEED)
        model = cfg["build"]().to(DEVICE)

        train_loader = make_dataloader(
            prepared.train, batching=model.BATCHING, batch_size=batch_size,
            shuffle=True, embedding_dict=embeddings_dict,
        )
        val_loader = make_dataloader(
            prepared.val, batching=model.BATCHING, batch_size=batch_size,
            shuffle=False, embedding_dict=embeddings_dict,
        )

        history = train_model(
            model, train_loader, val_loader,
            epochs=args.epochs, lr=lr, weight_decay=weight_decay,
            checkpoint_path=None, device=DEVICE,
            verbose=False,
        )

        for epoch_idx, val_loss in enumerate(history["val_loss"]):
            trial.report(val_loss, epoch_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return min(history["val_loss"])

    def make_early_stop_callback(min_improvement, patience):
        """Stops the study once the best val loss hasn't improved by more than
        `min_improvement` over the last `patience` completed trials."""
        best_history = []

        def callback(study, trial):
            best_history.append(study.best_value)
            if len(best_history) > patience:
                reference = best_history[-(patience + 1)]
                recent_best = min(best_history[-patience:])
                if reference - recent_best < min_improvement:
                    print(
                        f"Early stopping: best val loss improved by less than "
                        f"{min_improvement} over the last {patience} trials."
                    )
                    study.stop()

        return callback

    version_suffix = f"{name}_{args.version}"
    results_path = OUTPUT_DIR / f"tune_results_{version_suffix}.json"

    def save_results(study):
        with open(results_path, "w") as f:
            json.dump({
                "model": name,
                "version": args.version,
                "epochs_per_trial": args.epochs,
                "n_trials": args.trials,
                "best_value": study.best_value,
                "best_params": study.best_params,
                "trials": [
                    {
                        "number": t.number,
                        "value": t.value,
                        "params": t.params,
                        "state": str(t.state),
                    }
                    for t in study.trials
                ],
            }, f, indent=4)

    def make_save_callback():
        def callback(study, trial):
            save_results(study)
        return callback

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=SEED),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    try:
        study.optimize(
            objective,
            n_trials=args.trials,
            callbacks=[
                make_early_stop_callback(args.min_improvement, args.patience),
                make_save_callback(),
            ],
        )
    except KeyboardInterrupt:
        print("\nInterrupted - saving results collected so far.")

    if len(study.trials) == 0:
        print("No trials completed before interruption; nothing to save.")
        return

    print(f"\nBest val loss: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}")

    save_results(study)
    print(f"\nSaved tuning results to {results_path}")


if __name__ == "__main__":
    main()
