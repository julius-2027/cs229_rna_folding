import json
import os
import matplotlib.pyplot as plt

# ===========================================================================
# CONFIGURATION: Simply add or remove your JSON file paths here
# ===========================================================================
FILE_PATHS = [
    "results/metrics_bilstm_rna_fm_test.json",
    "results/metrics_embed_transformer_test.json",
    "results/metrics_glm_baseline_freqs.json",
    "results/metrics_bilstm_rna_fm_test.json"
]

def plot_learning_curves(json_paths):
    plt.figure(figsize=(10, 6))
    
    # Track if we actually successfully loaded anything to prevent blank plots
    loaded_any = False
    
    for path in json_paths:
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
        
        # Epochs are 1-indexed for the x-axis display
        epochs = list(range(1, len(train_loss) + 1))
        
        # Plot training loss (dashed line)
        plt.plot(
            epochs, 
            train_loss, 
            linestyle="--", 
            alpha=0.7, 
            label=f"{label_prefix} - Train"
        )
        
        # Plot validation loss (solid line, thicker to stand out)
        plt.plot(
            epochs, 
            val_loss, 
            linestyle="-", 
            linewidth=2, 
            label=f"{label_prefix} - Val"
        )
        
        # Optional: Drop a small marker pinpointing the best validation epoch
        if val_loss:
            best_val = min(val_loss)
            best_epoch = val_loss.index(best_val) + 1
            plt.scatter(
                best_epoch, 
                best_val, 
                s=40, 
                zorder=5, 
                label=f"Best Val: {best_val:.4f} (Ep {best_epoch})"
            )
            
        loaded_any = True

    if not loaded_any:
        print("No valid JSON metric files were loaded. Exiting window.")
        return

    # Graph Styling Parameters
    plt.title("Model Convergence Comparison (KL-Divergence Loss)", fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Epochs", fontsize=12)
    plt.ylabel("Loss Magnitude", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    
    # Place legend outside the main box so it doesn't cover up line paths
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True, shadow=True)
    plt.tight_layout()
    
    # Render interactive plot
    plt.show()

if __name__ == "__main__":
    plot_learning_curves(FILE_PATHS)