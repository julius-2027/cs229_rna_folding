"""
plot_comparisons.py
-------------------
Fully independent plotting script. No run_experiments imports, no CLI argument conflicts.
Adding a new model takes exactly one simple line.
"""

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import data as dt
import models as ms
from train_utils import set_seed

# ===========================================================================
# 1. Global Setup Constants
# ===========================================================================
DATA_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding/kinpfn_testing_set/parquet_parsing/test_val_dataset.parquet"
SEED = 42
N_BINS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===========================================================================
# 2. THE MODEL REGISTRY: Add new models here in 1 clean line
# Format: "Label": ( ms.ClassName, {kwargs}, "checkpoint_path" )
# ===========================================================================
MODELS_TO_PLOT = {
    "BiLSTM (RNA-FM)": (
        ms.DynamicEmbeddingHybridLSTM, 
        {"hidden_size": 64, "num_layers": 1, "static_feature_size": 3, "output_size": 50, "embedding_dim": 640, "mlp_hidden_size": 64, "bidirectional": True},
        "checkpoints/best_lstm_rna_fm_v1.pth"
    ),
    "Transformer (RNA-FM)": (
        ms.Embed_Transformer,
        {"embedding_dim": 640, "projection_dim": 64, "num_heads": 4, "num_layers": 1, "dropout": 0.5, "static_feature_size": 3, "output_size": 50, "mlp_hidden_size": 64},
        "checkpoints/best_transformer_rna_fm_v1.pth"
    ),
}

# ===========================================================================
# 3. Jensen-Shannon Divergence Calculation
# ===========================================================================
def calculate_js_divergence(p, q):
    m = 0.5 * (p + q)
    kl_pm = p * (np.log2(p + 1e-12) - np.log2(m + 1e-12))
    kl_qm = q * (np.log2(q + 1e-12) - np.log2(m + 1e-12))
    return 0.5 * (np.sum(kl_pm) + np.sum(kl_qm))

# ===========================================================================
# 4. Core Plotting Routine
# ===========================================================================
def plot_model_comparison(num_examples=3):
    print("Loading data partitions...")
    set_seed(SEED)
    handpicked_cols = ["gc_content", "mfe", "n_local_minima"]
    dataset_wrapper = dt.prepare_data(DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols)
    bin_centers = (dataset_wrapper.bin_edges[:-1] + dataset_wrapper.bin_edges[1:]) / 2
    
    val_loader = dt.make_dataloader(dataset_wrapper.val, batching="padded", batch_size=32, shuffle=False)
    sequences, structures, statics, targets, embeds, lengths,  = next(iter(val_loader))
    
    
    # Dynamic instantiation engine
    loaded_models = {}
    for name, (model_class, kwargs, checkpoint_path) in MODELS_TO_PLOT.items():
        try:
            model = model_class(**kwargs).to(DEVICE)
            checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
            
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            else:
                model.load_state_dict(checkpoint)
                
            model.eval()
            loaded_models[name] = model
            print(f"Loaded weights successfully for: {name}")
        except Exception as e:
            print(f"Skipping [{name}] registration step: {e}")

    num_models = len(loaded_models)
    if num_models == 0:
        print("Error: No models successfully built. Verify class parameters and paths.")
        return

    fig, axes = plt.subplots(num_examples, num_models, figsize=(4 * num_models, 3 * num_examples), sharex=True)
    if num_examples == 1: axes = np.expand_dims(axes, axis=0)
    if num_models == 1: axes = np.expand_dims(axes, axis=-1)

    print("Generating distribution visualizations...")
    for row_idx in range(num_examples):
        true_dist = targets[row_idx].numpy()
        
        for col_idx, (model_name, model) in enumerate(loaded_models.items()):
            ax = axes[row_idx, col_idx]
            
            with torch.no_grad():
                # Process across the stable batch size context
                all_logits = model(
                    sequences.to(DEVICE) if sequences is not None else None,
                    structures.to(DEVICE) if structures is not None else None,
                    statics.to(DEVICE),
                    embeds.to(DEVICE),
                    lengths.to(DEVICE) if lengths is not None else None
                )
                
                log_probs = F.log_softmax(all_logits, dim=-1)
                probs = torch.exp(log_probs)[row_idx].cpu().numpy()
            
            js_score = calculate_js_divergence(true_dist, probs)
            
            ax.plot(bin_centers, true_dist, label='True Target', color='black', alpha=0.5, linestyle=':')
            ax.plot(bin_centers, probs, label='Predicted', color='crimson' if "transformer" in model_name.lower() else 'royalblue')
            
            if row_idx == 0:
                ax.set_title(f"{model_name}\nSeq JS = {js_score:.3f}", fontsize=11, fontweight='bold')
            else:
                ax.set_title(f"Seq JS = {js_score:.3f}", fontsize=10)
                
            if col_idx == 0:
                ax.set_ylabel(f"RNA #{row_idx+1}\nProbability")
            if row_idx == num_examples - 1:
                ax.set_xlabel('ln(folding time)')
                
            ax.legend(loc='upper right', fontsize=8)
            ax.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    plot_model_comparison(num_examples=3)
