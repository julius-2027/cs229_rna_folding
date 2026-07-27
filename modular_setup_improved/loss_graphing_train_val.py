import json
import os
import matplotlib.pyplot as plt
from pathlib import Path

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
# ===========================================================================
# CONFIGURATION: Simply add or remove your JSON file paths here
# ===========================================================================
FILE_PATHS = [
    BASE_DIR / "results" / "metrics_bilstm_rna_fm_v1.json",
    BASE_DIR / "results" / "metrics_embed_transformer_v1.json",
    BASE_DIR / "results" / "metrics_rna_loc_v1.json",
    BASE_DIR / "results" / "metrics_glm_baseline_v1.json",
    BASE_DIR / "results" / "metrics_mean_pool_mlp_v1.json",
    BASE_DIR / "results" / "metrics_bilstm_v1.json"
]



def plot_learning_curves(json_paths):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
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
            label = f"{model_name} ({version})"
            history = data.get("history", {})
            train_loss = history.get("train_loss", [])
            val_loss = history.get("val_loss", [])
            epochs = list(range(1, len(train_loss) + 1))
            
            # Plot training loss
            ax1.plot(
                epochs, 
                train_loss, 
                linestyle="-", 
                linewidth=2, 
                color=colors[i % len(colors)],
                label=label
            )
            
            # Plot validation loss
            ax2.plot(
                epochs, 
                val_loss, 
                linestyle="-", 
                linewidth=2, 
                color=colors[i % len(colors)],
                label=label
            )
            
            loaded_any = True
    
    if not loaded_any:
        print("No valid JSON metric files were loaded. Exiting window.")
        return
    
    # Graph Styling Parameters
    fig.suptitle("Model Convergence Comparison", fontsize=16, fontweight="bold", y=0.95)
    
    ax1.set_title("Training Loss (KL-Divergence)", fontsize=14)
    ax1.set_xlabel("Epochs", fontsize=12)
    ax1.set_ylabel("Loss Magnitude", fontsize=12)
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend(loc="best", frameon=True, shadow=True)
    
    ax2.set_title("Validation Loss (KL-Divergence)", fontsize=14)
    ax2.set_xlabel("Epochs", fontsize=12)
    ax2.set_ylabel("Loss Magnitude", fontsize=12)
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend(loc="best", frameon=True, shadow=True)
    
    plt.tight_layout(rect=[0.04, 0.03, 1, 0.95])
    plt.show()
    fig.savefig('graphing_train_val_loss.png', dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    plot_learning_curves(FILE_PATHS)