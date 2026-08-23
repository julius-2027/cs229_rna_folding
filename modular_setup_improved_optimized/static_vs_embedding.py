"""
static_vs_embedding.py
-----------------------
For each of the RNA-FM-embedding models (lstm_rna_fm, struct_lstm_rna_fm,
transformer_rna_fm, loc_rna_fm), computes what fraction of the model's
attribution mass falls on the static handpicked features (gc_content, mfe,
n_local_minima) vs. the RNA-FM embeddings (and, for struct_lstm_rna_fm, the
raw structure encoding it also consumes).

Uses Integrated Gradients (manual implementation, zero baseline, no extra
dependency) attributed to each model's predicted (argmax) output bin's
log-probability, averaged over a sample of the held-out test set. Loads each
model's checkpoint from checkpoints/best_<name>_best_params_20ep.pth (the
tuned, single-split runs from run_experiments_best_params.py).

Run with: python static_vs_embedding.py
"""

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import models as ms
from data import prepare_data, make_dataloader
from train_utils import set_seed

BASE_DIR = Path(__file__).resolve().parent
MASK_PATH = None
DATA_PATH = BASE_DIR / "synthetic+real_dataset_filtered_numpeaks.parquet"
EMBEDDING_PATH = BASE_DIR / "all_fm-rna_embeddings_filtered.pt"
BIN_EDGES_PATH = BASE_DIR / "bin_edges.npy"
CHECKPOINT_DIR = BASE_DIR / "checkpoints"
BEST_PARAMS_CSV = BASE_DIR / "best_hyperparams.csv"
OUTPUT_CSV = BASE_DIR / "static_vs_embedding.csv"

SEED = 42
N_BINS = 50
DEVICE = "cpu"
N_SAMPLES = 64   # how many held-out test examples to average attributions over
IG_STEPS = 32    # Riemann-sum steps for Integrated Gradients

MODELS = ["lstm_rna_fm", "struct_lstm_rna_fm", "transformer_rna_fm", "loc_rna_fm"]


def build_model(name, static_dim, output_dim):
    if name == "lstm_rna_fm":
        return ms.DynamicEmbeddingHybridLSTM(
            hidden_size=64, num_layers=1, static_feature_size=static_dim,
            output_size=output_dim, embedding_dim=640, mlp_hidden_size=64,
            bidirectional=True,
        )
    if name == "struct_lstm_rna_fm":
        return ms.DynamicEmbeddingHybridLSTMwithStruct(
            hidden_size=64, num_layers=1, struct_dim=3, static_feature_size=static_dim,
            output_size=output_dim, embedding_dim=640, mlp_hidden_size=64,
            bidirectional=True,
        )
    if name == "transformer_rna_fm":
        return ms.Embed_Transformer(
            embedding_dim=640, projection_dim=64, num_heads=4, num_layers=1,
            dropout=0.5, static_feature_size=static_dim, output_size=output_dim,
            mlp_hidden_size=64,
        )
    if name == "loc_rna_fm":
        return ms.RNALocLM(
            embedding_dim=640, cnn_channels=128, kernel_sizes=(3, 4, 5),
            lstm_hidden=128, lstm_layers=1, num_heads=8,
            static_feature_size=3, output_size=50, mlp_hidden_size=64, dropout=0.3,
        )
    raise ValueError(f"Unknown model: {name}")


def integrated_gradients(model, sequence, structure, static_features, embeddings, lengths, steps=IG_STEPS):
    """
    Zero-baseline Integrated Gradients wrt static_features, structure, and
    embeddings. Target is each example's predicted (argmax) bin's
    log-probability. Returns (attr_static, attr_struct, attr_embed), each
    already multiplied by (input - baseline) and same shape as the input.
    """
    static_features = static_features.clone().detach()
    structure = structure.clone().detach()
    embeddings = embeddings.clone().detach()

    baseline_static = torch.zeros_like(static_features)
    baseline_struct = torch.zeros_like(structure)
    baseline_embed = torch.zeros_like(embeddings)

    with torch.no_grad():
        logits = model(sequence, structure, static_features, embeddings, lengths)
        target_bin = logits.argmax(dim=-1)

    grad_static_sum = torch.zeros_like(static_features)
    grad_struct_sum = torch.zeros_like(structure)
    grad_embed_sum = torch.zeros_like(embeddings)

    for step in range(1, steps + 1):
        alpha = step / steps
        s_static = (baseline_static + alpha * (static_features - baseline_static)).requires_grad_(True)
        s_struct = (baseline_struct + alpha * (structure - baseline_struct)).requires_grad_(True)
        s_embed = (baseline_embed + alpha * (embeddings - baseline_embed)).requires_grad_(True)

        logits = model(sequence, s_struct, s_static, s_embed, lengths)
        log_probs = torch.log_softmax(logits, dim=-1)
        target_logp = log_probs.gather(1, target_bin.unsqueeze(1)).sum()

        grads = torch.autograd.grad(
            target_logp, [s_static, s_struct, s_embed], allow_unused=True
        )
        grad_static, grad_struct, grad_embed = grads
        if grad_static is not None:
            grad_static_sum += grad_static.detach()
        if grad_struct is not None:
            grad_struct_sum += grad_struct.detach()
        if grad_embed is not None:
            grad_embed_sum += grad_embed.detach()

    attr_static = (static_features - baseline_static) * (grad_static_sum / steps)
    attr_struct = (structure - baseline_struct) * (grad_struct_sum / steps)
    attr_embed = (embeddings - baseline_embed) * (grad_embed_sum / steps)
    return attr_static, attr_struct, attr_embed


def run_one_model(name):
    handpicked_cols = ["gc_content", "mfe", "n_local_minima"]
    data_mask = pd.read_parquet(MASK_PATH) if MASK_PATH is not None else None

    set_seed(SEED)
    data = prepare_data(
        DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols,
        bin_edges_path=BIN_EDGES_PATH, data_mask=data_mask,
    )
    static_dim = data.train.static_features.shape[1]
    output_dim = data.train.targets.shape[1]

    model = build_model(name, static_dim, output_dim).to(DEVICE)
    checkpoint_path = CHECKPOINT_DIR / f"best_{name}_best_params_20ep.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()

    print(f"Loading RNA-FM embedding dictionary from {EMBEDDING_PATH}...")
    embeddings_dict = torch.load(EMBEDDING_PATH)

    test_loader = make_dataloader(
        data.test, batching=model.BATCHING, batch_size=N_SAMPLES,
        shuffle=False, embedding_dict=embeddings_dict,
    )
    sequence, structure, static_features, targets, embeddings, lengths = next(iter(test_loader))
    sequence = sequence.to(DEVICE)
    structure = structure.to(DEVICE)
    static_features = static_features.to(DEVICE)
    embeddings = embeddings.to(DEVICE)
    lengths = lengths.to(DEVICE)

    print(f"\n{'=' * 20} {name} {'=' * 20}")
    print(f"Running Integrated Gradients over {sequence.shape[0]} test examples, {IG_STEPS} steps...")

    attr_static, attr_struct, attr_embed = integrated_gradients(
        model, sequence, structure, static_features, embeddings, lengths
    )

    static_mass = attr_static.abs().sum().item()
    struct_mass = attr_struct.abs().sum().item()
    embed_mass = attr_embed.abs().sum().item()

    non_static_mass = struct_mass + embed_mass
    total_mass = static_mass + non_static_mass
    pct_static = 100.0 * static_mass / total_mass if total_mass > 0 else float("nan")
    pct_embed = 100.0 * embed_mass / total_mass if total_mass > 0 else float("nan")
    pct_struct = 100.0 * struct_mass / total_mass if total_mass > 0 else float("nan")

    print(f"  static feature attribution mass: {static_mass:.6g}")
    if struct_mass > 0:
        print(f"  structure attribution mass:      {struct_mass:.6g}")
    print(f"  embedding attribution mass:      {embed_mass:.6g}")
    print(f"  -> {pct_static:.1f}% static / {pct_embed:.1f}% embedding"
          + (f" / {pct_struct:.1f}% structure" if struct_mass > 0 else ""))

    return {
        "model": name,
        "pct_static": pct_static,
        "pct_embedding": pct_embed,
        "pct_structure": pct_struct,
        "static_mass": static_mass,
        "struct_mass": struct_mass,
        "embed_mass": embed_mass,
        "n_samples": sequence.shape[0],
    }


if __name__ == "__main__":
    results = [run_one_model(name) for name in MODELS]

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "pct_static", "pct_embedding", "pct_structure"])
        writer.writeheader()
        for r in results:
            writer.writerow({
                "model": r["model"],
                "pct_static": f"{r['pct_static']:.2f}",
                "pct_embedding": f"{r['pct_embedding']:.2f}",
                "pct_structure": f"{r['pct_structure']:.2f}",
            })

    with open(BASE_DIR / "static_vs_embedding_raw.json", "w") as f:
        json.dump(results, f, indent=4)

    print(f"\nWrote {len(results)} rows to {OUTPUT_CSV}")
    print(pd.DataFrame(results)[["model", "pct_static", "pct_embedding", "pct_structure"]].to_string(index=False))
