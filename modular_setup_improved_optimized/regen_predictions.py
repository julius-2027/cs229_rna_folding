"""
regen_predictions.py
-----------------------
Regenerates results/predictions_<name>_best_params_20ep.png for every real
trained model (everything in best_hyperparams.csv except the closed-form
naive_baseline/length_baseline rows, and always excluding
bilstm_rna_fm_proj), using the EXISTING checkpoints -- no retraining, since
only plot_prediction_grid's title metric changed (JS -> KL). Confirms the
same held-out test set used everywhere else (data.test from prepare_data).

Run with: python regen_predictions.py
"""

from pathlib import Path

import torch

from run_experiments_best_params import (
    build_model_configs, best_params_by_model, EXCLUDE_MODELS,
    DATA_PATH, BIN_EDGES_PATH, EMBEDDING_PATH, SEED, N_BINS, LOCAL_MINIMA_K, DEVICE, VERSION,
)
from data import prepare_data, make_dataloader
from train_utils import set_seed, evaluate_model, plot_prediction_grid

BASE_DIR = Path(__file__).resolve().parent
CLOSED_FORM_BASELINES = {"naive_baseline", "length_baseline"}


def run(name):
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

    set_seed(SEED)
    data = prepare_data(
        DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols,
        bin_edges_path=BIN_EDGES_PATH, data_mask=None,
        local_minima_k=LOCAL_MINIMA_K if all_local_data else None,
    )
    static_dim = data.train.static_features.shape[1]
    output_dim = data.train.targets.shape[1]
    bin_centers = (data.bin_edges[:-1] + data.bin_edges[1:]) / 2

    model_configs = build_model_configs(static_dim, output_dim)
    model = model_configs[name]["build"]().to(DEVICE)

    version_suffix = f"{name}_{VERSION}"
    checkpoint_path = BASE_DIR / "checkpoints" / f"best_{version_suffix}.pth"

    embeddings_dict = torch.load(EMBEDDING_PATH)
    test_loader = make_dataloader(
        data.test, batching=model.BATCHING, batch_size=best_params_by_model[name]["batch_size"],
        shuffle=False, embedding_dict=embeddings_dict,
    )

    print(f"{name}: test set size = {len(data.test.sequences)} (confirmed test-only)")
    eval_result = evaluate_model(model, test_loader, device=DEVICE, checkpoint_path=checkpoint_path)

    import matplotlib.pyplot as plt
    pred_fig = plot_prediction_grid(bin_centers, eval_result, n_examples=6)
    out_path = BASE_DIR / "results" / f"predictions_{version_suffix}.png"
    pred_fig.savefig(out_path)
    plt.close(pred_fig)
    print(f"  -> saved {out_path} (mean KL={eval_result['kl_mean']:.4f})")


if __name__ == "__main__":
    names = [n for n in best_params_by_model.keys() if n not in CLOSED_FORM_BASELINES and n not in EXCLUDE_MODELS]
    print(f"Regenerating predictions for: {names}")
    for name in names:
        run(name)
