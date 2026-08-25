"""
ablation_struct_lstm_rna_fm.py
--------------------------------
Causal/occlusion test for struct_lstm_rna_fm's already-trained checkpoint:
at inference time only (no retraining), zero out one input branch at a time
and measure how much test-set KL divergence degrades. This is a direct,
outcome-linked complement to the Integrated Gradients saliency numbers --
"how much does the model actually need this branch's real values" rather
than "how much does the gradient flow through this branch locally."

Three conditions, all on the held-out test set:
  1. baseline    -- real structure, real embeddings (no ablation)
  2. no_embed    -- embeddings zeroed out, structure kept real
  3. no_struct   -- structure zeroed out, embeddings kept real

Uses checkpoints/best_struct_lstm_rna_fm_best_params_20ep.pth (the tuned,
single-split run from run_experiments_best_params.py). Pure inference.

Run with: python ablation_struct_lstm_rna_fm.py
"""

import csv
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn

from run_experiments_best_params import build_model_configs
from data import prepare_data, make_dataloader
from train_utils import set_seed, _run_epoch

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
CHECKPOINT_PATH = BASE_DIR / "checkpoints" / "best_struct_lstm_rna_fm_best_params_20ep.pth"
OUTPUT_CSV = BASE_DIR / "ablation_struct_lstm_rna_fm.csv"

SEED = 42
N_BINS = 50
DEVICE = "cpu"
BATCH_SIZE = 64
MODEL_NAME = "struct_lstm_rna_fm"


class AblatedLoader:
    """Wraps a dataloader, zeroing out one branch of each batch on the fly."""

    def __init__(self, loader, ablate=None):
        self.loader = loader
        self.ablate = ablate  # None, "structure", or "embeddings"

    def __iter__(self):
        for sequences, structures, statics, targets, embeds, lengths in self.loader:
            if self.ablate == "structure":
                structures = torch.zeros_like(structures)
            elif self.ablate == "embeddings":
                embeds = torch.zeros_like(embeds)
            yield sequences, structures, statics, targets, embeds, lengths


def run():
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
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
    model.eval()

    print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
    embeddings_dict = torch.load(EMBEDDING_PATH)

    test_loader = make_dataloader(
        data.test, batching=model.BATCHING, batch_size=BATCH_SIZE,
        shuffle=False, embedding_dict=embeddings_dict,
    )

    criterion = nn.KLDivLoss(reduction="batchmean")

    conditions = {
        "baseline": None,
        "no_embed": "embeddings",
        "no_struct": "structure",
    }

    rows = []
    with torch.no_grad():
        for cond_name, ablate in conditions.items():
            loader = AblatedLoader(test_loader, ablate=ablate)
            test_kl = _run_epoch(model, loader, criterion, optimizer=None, device=DEVICE)
            print(f"{cond_name:10s} (ablate={str(ablate):10s}) test_kl = {test_kl:.4f}")
            rows.append({"condition": cond_name, "ablated_branch": ablate or "none", "test_kl": test_kl})

    baseline_kl = rows[0]["test_kl"]
    for r in rows:
        r["delta_vs_baseline"] = r["test_kl"] - baseline_kl

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["condition", "ablated_branch", "test_kl", "delta_vs_baseline"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "condition": r["condition"],
                "ablated_branch": r["ablated_branch"],
                "test_kl": f"{r['test_kl']:.4f}",
                "delta_vs_baseline": f"{r['delta_vs_baseline']:.4f}",
            })

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    run()
