import pandas as pd
import torch

# PARQUET_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding_clone/modular_setup_improved/synthetic+real_dataset.parquet"
# EMBEDDINGS_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding_clone/modular_setup_improved/all_fm-rna_embeddings.pt"

# FILTERED_PARQUET_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding_clone/modular_setup_improved/synthetic+real_dataset_filtered.parquet"
# FILTERED_EMBEDDINGS_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding_clone/modular_setup_improved/all_fm-rna_embeddings_filtered.pt"

PARQUET_PATH = "synthetic+real_dataset.parquet"
EMBEDDINGS_PATH = "all_fm-rna_embeddings.pt"

FILTERED_PARQUET_PATH = "synthetic+real_dataset_filtered.parquet"
FILTERED_EMBEDDINGS_PATH = "all_fm-rna_embeddings_filtered.pt"

CANONICAL_NUCLEOTIDES = set("AUGC")

df = pd.read_parquet(PARQUET_PATH)
embeddings = torch.load(EMBEDDINGS_PATH)

# Embedding dict keys are str(idx) of the dataframe's original index (see
# make_embeddings.py), so rows must be dropped by index rather than reset,
# to keep the remaining keys aligned with the remaining rows.
has_non_canonical = df["sequence"].apply(lambda seq: not set(seq.upper()) <= CANONICAL_NUCLEOTIDES)
bad_indices = df.index[has_non_canonical]

print(f"Found {len(bad_indices)} / {len(df)} rows with non-canonical nucleotides.")

for idx in bad_indices:
    key = str(idx)
    if key in embeddings:
        del embeddings[key]
    else:
        print(f"Warning: no embedding entry found for row index {idx}")

df = df.drop(index=bad_indices)

df.to_parquet(FILTERED_PARQUET_PATH)
torch.save(embeddings, FILTERED_EMBEDDINGS_PATH)

print(f"Wrote {len(df)} rows to {FILTERED_PARQUET_PATH}")
print(f"Wrote {len(embeddings)} embeddings to {FILTERED_EMBEDDINGS_PATH}")
