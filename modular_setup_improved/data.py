"""
data.py
-------
Data loading, label construction, feature engineering, and PyTorch
Dataset/Sampler/collate utilities shared by every model in models.py.

Design notes
------------
- All models consume the SAME `RNADataset` regardless of whether they need
  exact-length-bucketed batches (e.g. an LSTM without padding) or can handle
  variable-length / padded batches (e.g. a GLM that pools over the sequence,
  or any model that uses pack_padded_sequence).
- Two batching strategies are provided on top of that single Dataset:
    1. ExactLengthBatchSampler + collate_exact_length
       -> every batch contains sequences of identical length, no padding.
    2. Plain shuffling + collate_padded
       -> variable-length batches, zero-padded, with a `lengths` tensor
          so models can pack_padded_sequence / mask / mean-pool correctly.
- Sequence/structure encoding is a pluggable step (`encode_sequence`,
  `encode_structure`) so you can swap one-hot encoding for token ids etc.
  without touching anything else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Sampler

# ---------------------------------------------------------------------------
# Encoding maps (pluggable: swap these out for different representations)
# ---------------------------------------------------------------------------

NUC_MAP = {"A": [1, 0, 0, 0], "U": [0, 1, 0, 0], "G": [0, 0, 1, 0], "C": [0, 0, 0, 1]}
STRUCT_MAP = {".": [1, 0, 0], "(": [0, 1, 0], ")": [0, 0, 1]}


def encode_sequence(seq: str) -> np.ndarray:
    """One-hot encode an RNA sequence -> shape (len, 4)."""
    return np.array([NUC_MAP[c] for c in seq], dtype=np.float32)


def encode_structure(struct: str) -> np.ndarray:
    """One-hot encode a dot-bracket structure -> shape (len, 3)."""
    return np.array([STRUCT_MAP[c] for c in struct], dtype=np.float32)


# ---------------------------------------------------------------------------
# Loading + label construction
# ---------------------------------------------------------------------------

def load_dataset(parquet_path: str) -> pd.DataFrame:
    return pd.read_parquet(parquet_path)


def build_histogram_targets(df: pd.DataFrame, n_bins: int = 50,
                             fpts_col: str = "fpts", dist_col: str = "dist",
                             bin_edges: np.ndarray = None):
    """
    Build (or load precomputed) normalized log-fpt histograms as soft targets.

    If `dist_col` is already present (e.g. precomputed by test.py, which also
    drops `fpts_col`), those histograms are used directly and `bin_edges`
    must be supplied - it can't be recovered from histograms alone, so load
    the bin_edges.npy saved alongside them. Otherwise falls back to computing
    histograms on the fly from `fpts_col`, deriving bin_edges from the data
    if not given.

    Returns
    -------
    y : np.ndarray of shape (n_rows, n_bins)
    bin_edges : np.ndarray of shape (n_bins + 1,)
    """
    if dist_col in df.columns:
        if bin_edges is None:
            raise ValueError(
                f"'{dist_col}' is precomputed but no bin_edges were given - "
                f"pass the bin_edges.npy saved alongside it (see prepare_data's "
                f"bin_edges_path argument)."
            )
        y = np.stack(df[dist_col].to_numpy())
        return y, bin_edges

    if bin_edges is None:
        raise ValueError('need bins precomputed.')
        

def keep_handpicked_columns(df: pd.DataFrame, handpicked_cols: list = None,
                             fpts_col: str = "fpts", dist_col: str = "dist") -> pd.DataFrame:
    """Keep only the columns explicitly provided in handpicked_cols, plus mandatory keys."""
    if handpicked_cols is None:
        handpicked_cols = []

    # Combine lists WITHOUT mutating the input handpicked_cols list in-place
    target_col = dist_col
    mandatory_cols = ["sequence", "mfe_structure", target_col]
    all_needed_cols = list(set(handpicked_cols + mandatory_cols))


    # Slice the dataframe to only include columns that actually exist in the data
    valid_cols = [col for col in all_needed_cols if col in df.columns]

    return df[valid_cols].copy()


# ---------------------------------------------------------------------------
# Splitting + scaling
# ---------------------------------------------------------------------------

@dataclass
class DataBundle:
    """Everything a model/dataloader needs, for one split (train/val/test)."""
    sequences: list          # list of np.ndarray, variable length, encoded
    structures: list         # list of np.ndarray, variable length, encoded
    static_features: np.ndarray  # (n, n_static) scaled numeric features
    targets: np.ndarray      # (n, n_bins) histogram targets
    row_keys: list


@dataclass
class PreparedData:
    train: DataBundle
    val: DataBundle
    test: DataBundle
    bin_edges: np.ndarray
    scaler: StandardScaler
    handpicked_cols: list = field(default_factory=list)


def _bucket(df_split: pd.DataFrame, static_scaled: np.ndarray, y_split: np.ndarray) -> DataBundle:
    sequences = [encode_sequence(s) for s in df_split["sequence"].tolist()]
    structures = [encode_structure(s) for s in df_split["mfe_structure"].tolist()]
    row_keys = [str(idx) for idx in df_split.index.tolist()]
    return DataBundle(
        sequences=sequences,
        structures=structures,
        static_features=static_scaled.astype(np.float32),
        targets=y_split.astype(np.float32),
        row_keys=row_keys
    )


def prepare_data(
    parquet_path: str,
    handpicked_cols=None,
    n_bins: int = 50,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = 42,
    bin_edges_path: str = None,
) -> PreparedData:
    """
    Full pipeline: load -> build targets -> drop unused cols -> split -> scale -> encode.

    This is model-agnostic. Call once per experiment run (not once per model).

    bin_edges_path: required when parquet_path's dataframe has a precomputed
    'dist' column instead of raw 'fpts' (see test.py) - pass the bin_edges.npy
    saved alongside it.
    """

    df = load_dataset(parquet_path)
    bin_edges = np.load(bin_edges_path) if bin_edges_path else None
    y, bin_edges = build_histogram_targets(df, n_bins=n_bins, bin_edges=bin_edges)
    X = keep_handpicked_columns(df, handpicked_cols=handpicked_cols)

    #separate out train, test, val splits
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state
    )

    scaler = StandardScaler()
    handpicked_cols = list(handpicked_cols)
    static_train = scaler.fit_transform(X_train[handpicked_cols])
    static_val = scaler.transform(X_val[handpicked_cols])
    static_test = scaler.transform(X_test[handpicked_cols])

    train_bundle = _bucket(X_train, static_train, y_train)
    val_bundle = _bucket(X_val, static_val, y_val)
    test_bundle = _bucket(X_test, static_test, y_test)

    return PreparedData(
        train=train_bundle,
        val=val_bundle,
        test=test_bundle,
        bin_edges=bin_edges,
        scaler=scaler,
        handpicked_cols=handpicked_cols,
    )


# ---------------------------------------------------------------------------
# Dataset (shared by every model / batching strategy)
# ---------------------------------------------------------------------------

class RNADataset(Dataset):
    """
    Holds variable-length encoded sequences/structures + static features +
    targets. Works with either batching strategy below.
    """

    def __init__(self, bundle: DataBundle, embedding_dict: dict = None):
        self.sequences = bundle.sequences
        self.structures = bundle.structures
        self.static_features = bundle.static_features
        self.targets = bundle.targets
        self.row_keys = bundle.row_keys
        self.embeddings_dict = embedding_dict


    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        key = self.row_keys[idx]
        
        # Fetch the pre-computed embedding tensor -> Shape: (seq_len, 640)
        # Convert back from float16 to float32 for training math stability
        embedding = self.embeddings_dict[key].to(torch.float32)
        
        return (
            self.sequences[idx],
            self.structures[idx],
            self.static_features[idx],
            self.targets[idx],
            embedding,
        )   

# ---- Strategy 1: exact-length bucketing (no padding needed) --------------

class ExactLengthBatchSampler(Sampler):
    """Groups indices by exact sequence length so every batch is uniform length."""

    def __init__(self, dataset: RNADataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        length_to_indices = {}
        for idx in range(len(dataset)):
            length = len(dataset.sequences[idx])
            length_to_indices.setdefault(length, []).append(idx)
        self.batches = list(length_to_indices.values())

    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.batches)
        for batch in self.batches:
            yield batch

    def __len__(self):
        return len(self.batches)


def collate_exact_length(batch):
    seqs, structs, statics, targets, embeds = zip(*batch)
    
    # Pack them directly into tensors without padding loops
    padded_embeds = torch.stack(embeds) # Since uniform length, stack works perfectly
    padded_seqs = torch.from_numpy(np.array(seqs, dtype=np.float32))
    padded_structs = torch.from_numpy(np.array(structs, dtype=np.float32))
    
    statics = torch.from_numpy(np.array(statics, dtype=np.float32))
    targets = torch.from_numpy(np.array(targets, dtype=np.float32))
    lengths = torch.full((len(batch),), padded_seqs.shape[1], dtype=torch.long)
    
    return (
        padded_seqs,
        padded_structs,
        statics,
        targets,
        padded_embeds,
        lengths,
    )

# ---- Strategy 2: variable-length / padded batches -------------------------

def collate_padded(batch):
    """
    Zero-pads sequences/structures to the max length in the batch and
    returns a `lengths` tensor so models can pack_padded_sequence, mask,
    or mean-pool appropriately. Use this with a normal shuffling DataLoader
    for models that don't require uniform-length batches (e.g. a GLM that
    pools over valid positions, or any model using packed sequences).
    """

    seqs, structs, statics, targets, embeds = zip(*batch)
    lengths = torch.tensor([s.shape[0] for s in seqs], dtype=torch.long)
    max_len = int(lengths.max())

    seq_dim = seqs[0].shape[1]
    struct_dim = structs[0].shape[1]
    embed_dim = embeds[0].shape[1]

    padded_seqs = np.zeros((len(batch), max_len, seq_dim), dtype=np.float32)
    padded_structs = np.zeros((len(batch), max_len, struct_dim), dtype=np.float32)
    padded_embeds = np.zeros((len(batch), max_len, embed_dim), dtype=np.float32)
    
    for i, (s, st, e) in enumerate(zip(seqs, structs, embeds)):
        L = s.shape[0]
        padded_seqs[i, :L] = s
        padded_structs[i, :L] = st
        padded_embeds[i, :L] = e.numpy()

    statics = np.array(statics, dtype=np.float32)
    targets = np.array(targets, dtype=np.float32)


    return (
        torch.from_numpy(padded_seqs),
        torch.from_numpy(padded_structs),
        torch.from_numpy(statics),
        torch.from_numpy(targets),
        torch.from_numpy(padded_embeds),
        lengths,
    )


# ---------------------------------------------------------------------------
# Convenience: build a DataLoader given a strategy name
# ---------------------------------------------------------------------------

def make_dataloader(bundle: DataBundle, batching: str = "exact_length",
                     batch_size: int = 32, shuffle: bool = True, embedding_dict: dict = None):
    """
    batching: "exact_length" -> ExactLengthBatchSampler + collate_exact_length
                                 (batch_size is ignored; batches = all items of
                                 a given length)
              "padded"       -> standard shuffled DataLoader + collate_padded
    """
    dataset = RNADataset(bundle, embedding_dict=embedding_dict)
    if batching == "exact_length":
        sampler = ExactLengthBatchSampler(dataset, shuffle=shuffle)
        return torch.utils.data.DataLoader(
            dataset, batch_sampler=sampler, collate_fn=collate_exact_length
        )
    elif batching == "padded":
        return torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_padded
        )
    else:
        raise ValueError(f"Unknown batching strategy: {batching}")
