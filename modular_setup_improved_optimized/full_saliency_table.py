"""
full_saliency_table.py
------------------------
Builds the full model x input-branch saliency table:

    model, params, pct_seq, pct_struct, pct_static, pct_embed

- glm_baseline / all_local_mlp: forward() only ever touches static_features
  (see models.py), so this is 100% static by construction -- no computation
  needed.
- mean_pool_mlp / lstm: forward() uses sequence + structure + static_features
  (embeddings unused) -- computed here via Integrated Gradients.
- lstm_rna_fm / struct_lstm_rna_fm / transformer_rna_fm / loc_rna_fm: reuses
  the Integrated Gradients results already computed by static_vs_embedding.py
  (static_vs_embedding_raw.json) rather than recomputing.

params counts are NOT recomputed -- taken from the user-provided table.

Run with: python full_saliency_table.py
"""

import csv
import json
from pathlib import Path

import pandas as pd
import torch

from run_experiments_best_params import build_model_configs
from data import prepare_data, make_dataloader
from train_utils import set_seed

BASE_DIR = Path(__file__).resolve().parent
MASK_PATH = None
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
STATIC_VS_EMBED_JSON = BASE_DIR / "static_vs_embedding_raw.json"
OUTPUT_CSV = BASE_DIR / "full_saliency_table.csv"

SEED = 42
N_BINS = 50
DEVICE = "cpu"
N_SAMPLES = 64
IG_STEPS = 32

# params NOT recomputed -- from the user-provided table
PARAMS = {
    "glm_baseline": 1200,
    "mean_pool_mlp": 3954,
    "all_local_mlp": 7666,
    "lstm": 26290,
    "bilstm": 49074,
    "lstm_rna_fm": 373170,
    "struct_lstm_rna_fm": 374706,
    "transformer_rna_fm": 94898,
    "loc_rna_fm": 1792818,
}

MODEL_ORDER = [
    "glm_baseline",
    "mean_pool_mlp",
    "all_local_mlp",
    "lstm",
    "bilstm",
    "lstm_rna_fm",
    "struct_lstm_rna_fm",
    "transformer_rna_fm",
    "loc_rna_fm",
]

TRIVIAL_STATIC_ONLY = {"glm_baseline", "all_local_mlp"}
SEQ_STRUCT_STATIC_MODELS = {"mean_pool_mlp", "lstm", "bilstm"}


def integrated_gradients_seq_struct_static(model, sequence, structure, static_features, lengths, steps=IG_STEPS):
    """Zero-baseline IG wrt sequence, structure, static_features (embeddings unused by these models)."""
    sequence = sequence.clone().detach()
    structure = structure.clone().detach()
    static_features = static_features.clone().detach()

    baseline_seq = torch.zeros_like(sequence)
    baseline_struct = torch.zeros_like(structure)
    baseline_static = torch.zeros_like(static_features)
    dummy_embed = torch.zeros(sequence.shape[0], sequence.shape[1], 640)

    with torch.no_grad():
        logits = model(sequence, structure, static_features, dummy_embed, lengths)
        target_bin = logits.argmax(dim=-1)

    grad_seq_sum = torch.zeros_like(sequence)
    grad_struct_sum = torch.zeros_like(structure)
    grad_static_sum = torch.zeros_like(static_features)

    for step in range(1, steps + 1):
        alpha = step / steps
        s_seq = (baseline_seq + alpha * (sequence - baseline_seq)).requires_grad_(True)
        s_struct = (baseline_struct + alpha * (structure - baseline_struct)).requires_grad_(True)
        s_static = (baseline_static + alpha * (static_features - baseline_static)).requires_grad_(True)

        logits = model(s_seq, s_struct, s_static, dummy_embed, lengths)
        log_probs = torch.log_softmax(logits, dim=-1)
        target_logp = log_probs.gather(1, target_bin.unsqueeze(1)).sum()

        grad_seq, grad_struct, grad_static = torch.autograd.grad(
            target_logp, [s_seq, s_struct, s_static], allow_unused=True
        )
        if grad_seq is not None:
            grad_seq_sum += grad_seq.detach()
        if grad_struct is not None:
            grad_struct_sum += grad_struct.detach()
        if grad_static is not None:
            grad_static_sum += grad_static.detach()

    attr_seq = (sequence - baseline_seq) * (grad_seq_sum / steps)
    attr_struct = (structure - baseline_struct) * (grad_struct_sum / steps)
    attr_static = (static_features - baseline_static) * (grad_static_sum / steps)
    return attr_seq, attr_struct, attr_static


def run_seq_struct_static_model(name, embeddings_dict):
    handpicked_cols = ["gc_content", "mfe", "n_local_minima"]

    set_seed(SEED)
    data = prepare_data(
        DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols,
        bin_edges_path=BIN_EDGES_PATH, data_mask=None,
    )
    static_dim = data.train.static_features.shape[1]
    output_dim = data.train.targets.shape[1]

    model_configs = build_model_configs(static_dim, output_dim)
    model = model_configs[name]["build"]().to(DEVICE)
    checkpoint_path = CHECKPOINT_DIR / f"best_{name}_best_params_20ep.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    test_loader = make_dataloader(
        data.test, batching=model.BATCHING, batch_size=N_SAMPLES,
        shuffle=False, embedding_dict=embeddings_dict,
    )
    sequence, structure, static_features, targets, embeddings, lengths = next(iter(test_loader))

    print(f"\n{'=' * 20} {name} {'=' * 20}")
    print(f"Running Integrated Gradients over {sequence.shape[0]} test examples, {IG_STEPS} steps...")

    attr_seq, attr_struct, attr_static = integrated_gradients_seq_struct_static(
        model, sequence, structure, static_features, lengths
    )

    seq_mass = attr_seq.abs().sum().item()
    struct_mass = attr_struct.abs().sum().item()
    static_mass = attr_static.abs().sum().item()
    total_mass = seq_mass + struct_mass + static_mass

    pct_seq = 100.0 * seq_mass / total_mass if total_mass > 0 else float("nan")
    pct_struct = 100.0 * struct_mass / total_mass if total_mass > 0 else float("nan")
    pct_static = 100.0 * static_mass / total_mass if total_mass > 0 else float("nan")

    print(f"  -> {pct_seq:.1f}% seq / {pct_struct:.1f}% struct / {pct_static:.1f}% static")

    return {
        "model": name,
        "pct_seq": pct_seq,
        "pct_struct": pct_struct,
        "pct_static": pct_static,
        "pct_embed": 0.0,
    }


if __name__ == "__main__":
    with open(STATIC_VS_EMBED_JSON) as f:
        embed_model_results = {r["model"]: r for r in json.load(f)}

    print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
    embeddings_dict = torch.load(EMBEDDING_PATH)

    rows = []
    for name in MODEL_ORDER:
        if name in TRIVIAL_STATIC_ONLY:
            rows.append({"model": name, "pct_seq": 0.0, "pct_struct": 0.0, "pct_static": 100.0, "pct_embed": 0.0})
        elif name in SEQ_STRUCT_STATIC_MODELS:
            rows.append(run_seq_struct_static_model(name, embeddings_dict))
        else:
            r = embed_model_results[name]
            rows.append({
                "model": name,
                "pct_seq": 0.0,
                "pct_struct": r["pct_structure"],
                "pct_static": r["pct_static"],
                "pct_embed": r["pct_embedding"],
            })
        rows[-1]["params"] = PARAMS[name]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "params", "pct_seq", "pct_struct", "pct_static", "pct_embed"])
        writer.writeheader()
        for r in rows:
            writer.writerow({
                "model": r["model"],
                "params": r["params"],
                "pct_seq": f"{r['pct_seq']:.2f}",
                "pct_struct": f"{r['pct_struct']:.2f}",
                "pct_static": f"{r['pct_static']:.2f}",
                "pct_embed": f"{r['pct_embed']:.2f}",
            })

    print(f"\nWrote {len(rows)} rows to {OUTPUT_CSV}")
    print(pd.DataFrame(rows)[["model", "params", "pct_seq", "pct_struct", "pct_static", "pct_embed"]].to_string(index=False))
