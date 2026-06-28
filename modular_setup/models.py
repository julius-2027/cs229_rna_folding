"""
models.py
---------
All model architectures share a single forward signature so train_utils.py
never needs to know which model it's training:

    logits = model(sequence, structure, static_features, lengths)

- `sequence`, `structure`: (batch, seq_len, feat_dim) float tensors, possibly
  zero-padded if batching == "padded", or exact-length if batching ==
  "exact_length" (in which case every row's true length == seq_len).
- `static_features`: (batch, n_static) float tensor.
- `lengths`: (batch,) long tensor of true (unpadded) sequence lengths.
  Models that don't need it (e.g. a GLM using only static features) can
  simply ignore the argument.

Each model declares which batching strategy it expects via the class
attribute `BATCHING`, so the experiment runner knows how to build its
DataLoader without you having to remember per-model quirks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence


class BaseModel(nn.Module):
    """Marker base class. Subclasses must set BATCHING and implement forward."""

    # "exact_length" -> requires uniform-length batches (no padding)
    # "padded"       -> handles variable-length / padded batches fine
    BATCHING = "padded"

    def forward(self, sequence, structure, static_features, embeddings,lengths=None):
        raise NotImplementedError


class GLMBaseline(BaseModel):
    """
    Simple generalized-linear-model-style baseline: ignores sequence/structure
    entirely and predicts the output distribution from the handpicked static
    features alone (gc_content, mfe, n_local_minima, ...). Handles arbitrary
    batch composition since it never looks at sequence length.
    """

    BATCHING = "padded"  # irrelevant here, but "padded" avoids exact-length bucketing overhead

    def __init__(self, static_feature_size: int, output_size: int):
        super().__init__()
        self.linear = nn.Linear(static_feature_size, output_size)

    def forward(self, sequence, structure, static_features, embeddings,lengths=None):
        return self.linear(static_features)


class MeanPoolMLP(BaseModel):
    """
    A slightly richer variable-length-friendly baseline: mean-pools the
    sequence/structure encodings over valid (unpadded) positions, concatenates
    with static features, and passes through an MLP. Works fine on padded,
    variable-length batches because pooling is masked by `lengths`.
    """

    BATCHING = "padded"

    def __init__(self, static_feature_size: int, output_size: int,
                 seq_struct_dim: int = 7, hidden_size: int = 64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(seq_struct_dim + static_feature_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        )

    def forward(self, sequence, structure, static_features, embeddings, lengths=None):
        x = torch.cat([sequence, structure], dim=-1)  # (batch, seq_len, feat)
        if lengths is None:
            pooled = x.mean(dim=1)
        else:
            mask = (
                torch.arange(x.size(1), device=x.device)[None, :] < lengths[:, None]
            ).unsqueeze(-1)  # (batch, seq_len, 1)
            summed = (x * mask).sum(dim=1)
            pooled = summed / lengths.clamp(min=1).unsqueeze(-1).to(x.dtype)
        combined = torch.cat([pooled, static_features], dim=-1)
        return self.mlp(combined)


class DynamicHybridLSTM(BaseModel):
    """
    BiLSTM over the concatenated sequence+structure encoding, combined with
    static handpicked features via an MLP head.

    Works with EITHER batching strategy:
      - "exact_length": pass lengths=None (or a uniform-length tensor) and the
        LSTM runs over the full, unpadded batch directly.
      - "padded": pass the `lengths` tensor and the model will
        pack_padded_sequence internally so padding doesn't corrupt the
        final hidden state.

    Default BATCHING is "exact_length" because that's the more efficient
    option when available (no padding/packing overhead), but it will run
    correctly under "padded" too.
    """

    BATCHING = "padded"

    def __init__(self, hidden_size: int, num_layers: int, static_feature_size: int,
                 output_size: int, seq_struct_dim: int = 7, mlp_hidden_size: int = 64,
                 bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=seq_struct_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        mlp_input_size = hidden_size + static_feature_size
        if bidirectional:
            mlp_input_size += hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, output_size),
        )

    def forward(self, sequence, structure, static_features, embeddings,lengths=None):
        x = torch.cat([sequence, structure], dim=-1)

        use_packing = lengths is not None and not torch.all(lengths == lengths[0])
        if use_packing:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, cell) = self.lstm(packed)
        else:
            _, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]
        if self.bidirectional:
            last_hidden_fwd = hidden[-2]
            last_hidden = torch.cat([last_hidden_fwd, last_hidden], dim=-1)

        combined = torch.cat([last_hidden, static_features], dim=-1)
        return self.mlp(combined)

class DynamicEmbeddingHybridLSTM(BaseModel):
    """
    BiLSTM over the concatenated sequence+structure encoding, combined with
    static handpicked features via an MLP head.

    Works with EITHER batching strategy:
      - "exact_length": pass lengths=None (or a uniform-length tensor) and the
        LSTM runs over the full, unpadded batch directly.
      - "padded": pass the `lengths` tensor and the model will
        pack_padded_sequence internally so padding doesn't corrupt the
        final hidden state.

    Default BATCHING is "exact_length" because that's the more efficient
    option when available (no padding/packing overhead), but it will run
    correctly under "padded" too.
    """

    BATCHING = "padded"

    def __init__(self, hidden_size: int, num_layers: int, static_feature_size: int,
                 output_size: int, embedding_dim: int = 640, mlp_hidden_size: int = 64,
                 bidirectional: bool = True):
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
        )

        mlp_input_size = hidden_size + static_feature_size
        if bidirectional:
            mlp_input_size += hidden_size

        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, output_size),
        )

    def forward(self, sequence, structs, static_features, embeddings, lengths=None):
        x = embeddings
        use_packing = lengths is not None and not torch.all(lengths == lengths[0])
        if use_packing:
            packed = pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hidden, cell) = self.lstm(packed)
        else:
            _, (hidden, cell) = self.lstm(x)

        last_hidden = hidden[-1]
        if self.bidirectional:
            last_hidden_fwd = hidden[-2]
            last_hidden = torch.cat([last_hidden_fwd, last_hidden], dim=-1)

        combined = torch.cat([last_hidden, static_features], dim=-1)
        return self.mlp(combined)