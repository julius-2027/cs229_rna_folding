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

from pathlib import Path

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

chars = ['A', 'U', 'G', 'C', 'K', 'X', 'D', 'Y', 'N', 'R', 'S', 'H', 'M', 'W']

# Create the one-hot dictionary
vocab_size = len(chars)
one_hot_dict = {
    char: [1 if i == idx else 0 for i in range(vocab_size)] 
    for idx, char in enumerate(chars)
}

#NUC_MAP = {"A": [1, 0, 0, 0], "U": [0, 1, 0, 0], "G": [0, 0, 1, 0], "C": [0, 0, 0, 1]}
NUC_MAP = one_hot_dict
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
                             fpts_col: str = "fpts", bin_edges: np.ndarray = None):
    """
    Build normalized log-fpt histograms as soft targets.

    If `bin_edges` is not given, it is derived from `df` itself (legacy
    behavior). Pass in bin_edges computed from a train split to reuse the
    same binning for val/test without leaking their fpts range into the edges.

    Returns
    -------
    y : np.ndarray of shape (n_rows, n_bins)
    bin_edges : np.ndarray of shape (n_bins + 1,)
    """
    if bin_edges is None:
        arr_fpts = np.array([row for row in df[fpts_col]])
        # remove zeros for log behavior
        nonzero = arr_fpts[arr_fpts != 0]
        log_all = np.log(nonzero)
        # !!!!! objective changes slightly with dataset. once we have dataset compiled this should be fine.#
        bin_edges = np.linspace(log_all.min(), log_all.max(), n_bins + 1)

    y = np.zeros((df.shape[0], n_bins))
    for i, row in df.iterrows():
        fpts = row[fpts_col]
        nz = fpts[fpts != 0]
        logfpts = np.log(nz)
        hist, _ = np.histogram(logfpts, bin_edges, density=True)
        hist /= np.sum(hist)
        y[i] = hist
    return y, bin_edges


def keep_handpicked_columns(df: pd.DataFrame, handpicked_cols: list = None) -> pd.DataFrame:
    """Keep only the columns explicitly provided in handpicked_cols, plus mandatory keys."""
    if handpicked_cols is None:
        handpicked_cols = []
        
    # Combine lists WITHOUT mutating the input handpicked_cols list in-place
    mandatory_cols = ["sequence", "mfe_structure", "fpts"]
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
    embeddings: dict = None  # row_key -> tensor, preloaded (see prepare_data_from_bundles).
                              # None means RNADataset must load an embedding_path itself (legacy path).


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
    random_state: int = 42
) -> PreparedData:
    """
    Full pipeline: load -> build targets -> drop unused cols -> split -> scale -> encode.

    This is model-agnostic. Call once per experiment run (not once per model).
    """
    
    df = load_dataset(parquet_path)
    y, bin_edges = build_histogram_targets(df, n_bins=n_bins)
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
# Merging multiple (parquet, embeddings) sources into one split-first dataset
# ---------------------------------------------------------------------------
#
# Each source's embeddings dict is keyed on that source's own dataframe index
# (see make_embeddings.py: `str(idx)` from `df.iterrows()`), so two sources
# both have keys "0", "1", ... that collide. Every row is re-keyed as
# "<source_label>::<original_index>" in both the merged dataframe (new
# `row_key` column) and the merged embeddings dict so nothing collides or
# gets silently overwritten.
#
# The workflow is: merge_sources -> split_merged -> save_split_bundle, once,
# to produce train.pt/val.pt/test.pt. Every later run just calls
# prepare_data_from_bundles on those 3 files - no re-parsing parquet, no
# re-running train_test_split, no reloading the full embeddings file per
# DataLoader (train/val/test each get only their own, already-loaded,
# already-subsetted embeddings dict).

def _make_row_key(source_label: str, orig_key: str) -> str:
    return f"{source_label}::{orig_key}"


def merge_sources(parquet_paths: list, embedding_paths: list, source_labels: list = None):
    """
    Merge N (parquet, embeddings) pairs produced independently (e.g. by
    separate make_embeddings.py runs on different datasets) into one
    dataframe + one embeddings dict.

    ALL original columns from every source are kept (no dropping) so nothing
    is lost for downstream analysis - only two columns are added:
    `source_dataset` (the label for that row's origin) and `row_key` (the
    globally-unique embedding lookup key). If the sources have different
    columns, the missing ones are NaN for the other source's rows; that's
    fine unless you put such a column in `handpicked_cols` for scaling later
    (prepare_data_from_bundles will raise if it finds NaNs there).

    Returns
    -------
    merged_df : pd.DataFrame, all rows from all sources, all original columns
    merged_embeddings : dict[str, torch.Tensor], keyed by the new row_key
    """
    if source_labels is None:
        source_labels = [f"source{i}" for i in range(len(parquet_paths))]
    assert len(parquet_paths) == len(embedding_paths) == len(source_labels), \
        "parquet_paths, embedding_paths, and source_labels must be the same length"

    dfs = []
    merged_embeddings = {}
    for parquet_path, embedding_path, label in zip(parquet_paths, embedding_paths, source_labels):
        df = load_dataset(parquet_path).copy()
        embeddings = torch.load(embedding_path)

        orig_keys = [str(idx) for idx in df.index]
        new_keys = [_make_row_key(label, k) for k in orig_keys]

        missing = [k for k in orig_keys if k not in embeddings]
        if missing:
            raise KeyError(
                f"{len(missing)} row(s) in {parquet_path} have no matching embedding in "
                f"{embedding_path} (e.g. missing key {missing[0]!r}). Are these really the "
                f"parquet + .pt pair that were used together in make_embeddings.py?"
            )

        df["source_dataset"] = label
        df["row_key"] = new_keys
        dfs.append(df)

        for orig_key, new_key in zip(orig_keys, new_keys):
            merged_embeddings[new_key] = embeddings[orig_key]

    merged_df = pd.concat(dfs, axis=0, ignore_index=True)
    return merged_df, merged_embeddings


def split_merged(merged_df: pd.DataFrame, embeddings: dict,
                  test_size: float = 0.15, val_size: float = 0.15,
                  random_state: int = 42):
    """
    Split BEFORE any filtering. Filtering by rna_type/source_dataset/etc.
    should always happen after this, on the already-split dataframes
    (see prepare_data_from_bundles), so it can never move a row across
    splits or leak train rows into val/test.

    Returns
    -------
    dict with keys "train"/"val"/"test" -> (split_df, split_embeddings)
    """
    train_df, test_df = train_test_split(
        merged_df, test_size=test_size, random_state=random_state
    )
    train_df, val_df = train_test_split(
        train_df, test_size=val_size, random_state=random_state
    )

    splits = {}
    for name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split_df = split_df.reset_index(drop=True)
        split_embeddings = {k: embeddings[k] for k in split_df["row_key"]}
        splits[name] = (split_df, split_embeddings)
    return splits


def save_split_bundle(path: str, df_split: pd.DataFrame, embeddings_split: dict):
    """Write one split's dataframe + its own embeddings dict to a single binary file."""
    torch.save({"df": df_split, "embeddings": embeddings_split}, path)


def load_split_bundle(path: str):
    """Load a bundle written by save_split_bundle. Returns (df, embeddings)."""
    bundle = torch.load(path)
    return bundle["df"], bundle["embeddings"]


def build_merged_split_bundles(parquet_paths: list, embedding_paths: list, out_dir: str,
                                source_labels: list = None, test_size: float = 0.15,
                                val_size: float = 0.15, random_state: int = 42):
    """
    One-time preprocessing entry point: merge N (parquet, embedding) source
    pairs, split into train/val/test, and write each split to its own .pt
    file (out_dir/train.pt, val.pt, test.pt) containing that split's full
    dataframe (every original column preserved) plus its own embeddings dict.

    Run this once whenever the source data changes; every training run after
    that should load the 3 output files via prepare_data_from_bundles instead
    of re-merging/re-splitting/re-parsing the original parquet + .pt files.
    """
    merged_df, merged_embeddings = merge_sources(parquet_paths, embedding_paths, source_labels)
    splits = split_merged(merged_df, merged_embeddings, test_size, val_size, random_state)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, (split_df, split_embeddings) in splits.items():
        path = out_dir / f"{name}.pt"
        save_split_bundle(path, split_df, split_embeddings)
        paths[name] = str(path)
        print(f"Wrote {name} split: {len(split_df)} rows -> {path}")
    return paths


def prepare_data_from_bundles(train_path: str, val_path: str, test_path: str,
                               handpicked_cols: list = None, n_bins: int = 50,
                               filter_fn=None) -> PreparedData:
    """
    Load pre-merged, pre-split train/val/test bundles (see
    build_merged_split_bundles) into a PreparedData ready for make_dataloader.

    filter_fn, if given, is applied independently to each split's dataframe,
    e.g. `lambda df: df[df["rna_type"] == "tRNA"]` or
    `lambda df: df[df["source_dataset"] == "bigger"]`. Because train/val/test
    assignment was already fixed when the bundles were built, filtering here
    can never move a row across splits or leak rows between them.

    bin_edges and the static-feature scaler are fit on the TRAIN split only
    and reused for val/test, so no information about val/test ever reaches
    a fitting step.
    """
    handpicked_cols = list(handpicked_cols) if handpicked_cols else []

    raw = {}
    for name, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
        df, embeddings = load_split_bundle(path)
        if filter_fn is not None:
            df = filter_fn(df)
            embeddings = {k: embeddings[k] for k in df["row_key"]}
        raw[name] = (df.reset_index(drop=True), embeddings)

    _, bin_edges = build_histogram_targets(raw["train"][0], n_bins=n_bins)

    scaler = StandardScaler()
    if handpicked_cols:
        scaler.fit(raw["train"][0][handpicked_cols])

    bundles = {}
    for name, (df, embeddings) in raw.items():
        if handpicked_cols:
            static_cols = df[handpicked_cols]
            bad_cols = static_cols.columns[static_cols.isnull().any()].tolist()
            if bad_cols:
                raise ValueError(
                    f"handpicked_cols {bad_cols} contain NaNs in the '{name}' split after "
                    f"merging sources - likely one source parquet doesn't have this column. "
                    f"Drop it from handpicked_cols or impute before calling prepare_data_from_bundles."
                )
            static = scaler.transform(static_cols).astype(np.float32)
        else:
            static = np.zeros((len(df), 0), dtype=np.float32)

        y, _ = build_histogram_targets(df, n_bins=n_bins, bin_edges=bin_edges)

        bundles[name] = DataBundle(
            sequences=[encode_sequence(s) for s in df["sequence"].tolist()],
            structures=[encode_structure(s) for s in df["mfe_structure"].tolist()],
            static_features=static,
            targets=y.astype(np.float32),
            row_keys=df["row_key"].tolist(),
            embeddings=embeddings,
        )

    return PreparedData(
        train=bundles["train"],
        val=bundles["val"],
        test=bundles["test"],
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

    def __init__(self, bundle: DataBundle, embedding_path: str = "modular_setup/fm-rna_embeddings.pt"):
        self.sequences = bundle.sequences
        self.structures = bundle.structures
        self.static_features = bundle.static_features
        self.targets = bundle.targets
        self.row_keys = bundle.row_keys

        if bundle.embeddings is not None:
            # Already loaded for this split (e.g. via prepare_data_from_bundles) -
            # avoids re-reading the whole embeddings file for every DataLoader.
            self.embeddings_dict = bundle.embeddings
        else:
            print(f"Loading RNA-FM embedding dictionary from {embedding_path}...")
            self.embeddings_dict = torch.load(embedding_path)


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
                     batch_size: int = 32, shuffle: bool = True, embedding_path="fm-rna_embeddings_subset.pt"):
    """
    batching: "exact_length" -> ExactLengthBatchSampler + collate_exact_length
                                 (batch_size is ignored; batches = all items of
                                 a given length)
              "padded"       -> standard shuffled DataLoader + collate_padded
    """
    dataset = RNADataset(bundle, embedding_path=embedding_path)
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
