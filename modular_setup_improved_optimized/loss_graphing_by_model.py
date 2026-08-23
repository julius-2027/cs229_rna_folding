import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

# ===========================================================================
# CONFIGURATION: Simply add or remove your JSON file paths here
# ===========================================================================
BASE_DIR = Path(__file__).resolve().parent
FILE_PATHS = [
    BASE_DIR / "results" / "metrics_glm_baseline_best_params_20ep.json",
    BASE_DIR / "results" / "metrics_mean_pool_mlp_best_params_20ep.json",
    BASE_DIR / "results" / "metrics_all_local_mlp_best_params_20ep.json",
    BASE_DIR / "results" / "metrics_lstm_best_params_20ep.json",
    BASE_DIR / "results" / "metrics_lstm_rna_fm_best_params_20ep.json",
    BASE_DIR / "results" / "metrics_struct_lstm_rna_fm_best_params_20ep.json",
    BASE_DIR / "results" / "metrics_transformer_rna_fm_best_params_20ep.json",
    BASE_DIR / "results" / "metrics_loc_rna_fm_best_params_20ep.json",
    
]

def plot_learning_curves(json_paths):
    num_models = len(json_paths)
    num_cols = 3
    num_rows = (num_models + num_cols - 1) // num_cols
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(16, 5 * num_rows), sharex=True, sharey=True)
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    loaded_any = False
    for i, path in enumerate(json_paths):
        if not os.path.exists(path):
            print(f"Warning: File not found, skipping -> {path}")
            continue
        with open(path, "r") as f:
            data = json.load(f)
            model_name = data.get("model", "UnknownModel")
            version = data.get("version", "UnknownVersion")
            label_prefix = f"{model_name} ({version})"
            history = data.get("history", {})
            train_loss = history.get("train_loss", [])
            val_loss = history.get("val_loss", [])
            epochs = list(range(1, len(train_loss) + 1))
            
            # Plot training loss (dashed line)
            axes[i].plot(
                epochs, 
                train_loss, 
                linestyle="--", 
                alpha=0.7, 
                color=colors[i % len(colors)],
                label=f"{label_prefix} - Train"
            )
            # Plot validation loss (solid line, thicker to stand out)
            axes[i].plot(
                epochs, 
                val_loss, 
                linestyle="-", 
                linewidth=2, 
                color=colors[i % len(colors)],
                label=f"{label_prefix} - Val"
            )
            
            axes[i].set_title(label_prefix, fontsize=12)
            axes[i].grid(True, linestyle=":", alpha=0.6)
            axes[i].legend(loc="best", frameon=True, shadow=True)
            
            loaded_any = True
    
    if not loaded_any:
        print("No valid JSON metric files were loaded. Exiting window.")
        return
    
    # Remove empty subplots
    for i in range(num_models, num_rows * num_cols):
        fig.delaxes(axes[i])
    
    # Graph Styling Parameters
    fig.suptitle("Model Convergence Comparison (KL-Divergence Loss)", fontsize=16, fontweight="bold", y=0.95)
    fig.text(0.5, 0.04, "Epochs", ha='center', fontsize=14)
    fig.text(0.04, 0.5, "Loss Magnitude", va='center', rotation='vertical', fontsize=14)
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.95])
    plt.savefig(BASE_DIR / "results" / "model_loss_comparison_by_model.png", dpi=300)
    print(f"Saved loss comparison graph to {BASE_DIR / 'results' / 'model_loss_comparison_by_model.png'}")
if __name__ == "__main__":
    plot_learning_curves(FILE_PATHS)