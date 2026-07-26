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
from symtable import Class

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
    #     print("input shape:", x.shape, "lengths shape:", lengths.shape,
    #   "lengths max:", lengths.max().item(), "lengths min:", lengths.min().item())
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

class DynamicEmbeddingHybridLSTM_proj(BaseModel):

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
                 bidirectional: bool = True, projection_dim: int = 64):
        super().__init__()
        self.bidirectional = bidirectional

        self.projection = nn.Linear(embedding_dim, projection_dim)

        self.lstm = nn.LSTM(
            input_size=projection_dim,
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
        x = self.projection(x)  
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
    

class Embed_Transformer(nn.Module):
    """
    A simple transformer encoder that takes in embeddings and outputs a fixed-size representation.
    """
    BATCHING = "padded"
    def __init__(self, embedding_dim: int = 640, projection_dim: int = 128, num_heads: int = 4, num_layers: int = 1, dropout: float = 0.3, static_feature_size: int = 3, output_size: int = 50, mlp_hidden_size: int = 64):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.input_proj = nn.Linear(embedding_dim, projection_dim)
        self.pos_embedding = nn.Embedding(200, projection_dim) # Assuming max sequence length 
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=projection_dim, 
            nhead=num_heads, 
            dim_feedforward=projection_dim * 2, 
            dropout=dropout, 
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Late Fusion Prediction Head (MLP)
        mlp_input_size = projection_dim + static_feature_size
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, output_size),
        )
    def forward(self, sequence, structs, static_features, embeddings, lengths=None):
        
        x = embeddings
        batch_size, seq_len, _ = x.shape
        x = self.input_proj(x)  # Project embeddings to a lower dimension

        positions = torch.arange(0, seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x = x + self.pos_embedding(positions)
        x_trans = self.transformer(x)
        if lengths is not None:
            mask = torch.arange(seq_len, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
            mask = mask.unsqueeze(-1).float() # (Batch, Seq_Len, 1)
            # Sum up valid positions and divide by sequence lengths for a clean mean-pool
            pooled = (x_trans * mask).sum(dim=1) / lengths.unsqueeze(-1).float()
        else:
            pooled = x_trans.mean(dim=1) # Fallback flat mean-pool if lengths missing
            
        # Step 5: Late Fusion with Tabular Static Features
        combined = torch.cat([pooled, static_features], dim=-1)
        
        # Step 6: Map directly to output target distribution dimensions (50 bins)
        return self.mlp(combined)

class RNALocLM(BaseModel):
    """
    RNA-FM embeddings -> multi-kernel TextCNN -> BiLSTM -> multi-head
    self-attention -> FC, matching the RNALoc-LM architecture (Yan et al.
    2025, PMC11978386), but with the FC head sized to predict the 50 output
    bins instead of a localization class.
    """

    BATCHING = "padded"

    def __init__(self, embedding_dim: int = 640, cnn_channels: int = 128,
                 kernel_sizes: tuple = (3, 4, 5), lstm_hidden: int = 128,
                 lstm_layers: int = 1, num_heads: int = 8,
                 static_feature_size: int = 3, output_size: int = 50,
                 mlp_hidden_size: int = 64, dropout: float = 0.3):
        super().__init__()
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, cnn_channels, kernel_size=k, padding=k // 2)
            for k in kernel_sizes
        ])
        self.cnn_relu = nn.ReLU()
        cnn_out_dim = cnn_channels * len(kernel_sizes)

        self.lstm = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
        )
        lstm_out_dim = lstm_hidden * 2

        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_out_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(lstm_out_dim + static_feature_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_size, output_size),
        )

    def forward(self, sequence, structs, static_features, embeddings, lengths=None):
        x = embeddings  # (batch, seq_len, embedding_dim)
        seq_len = x.size(1)

        key_padding_mask = None
        if lengths is not None:
            key_padding_mask = (
                torch.arange(seq_len, device=x.device).unsqueeze(0) >= lengths.unsqueeze(1)
            )

        conv_in = x.transpose(1, 2)  # (batch, embedding_dim, seq_len)
        conv_outs = []
        for conv in self.convs:
            out = self.cnn_relu(conv(conv_in))  # (batch, cnn_channels, seq_len')
            if out.size(-1) > seq_len:
                out = out[:, :, :seq_len]
            elif out.size(-1) < seq_len:
                out = nn.functional.pad(out, (0, seq_len - out.size(-1)))
            conv_outs.append(out)
        cnn_features = torch.cat(conv_outs, dim=1).transpose(1, 2)  # (batch, seq_len, cnn_out_dim)

        use_packing = lengths is not None and not torch.all(lengths == lengths[0])
        if use_packing:
            packed = pack_padded_sequence(
                cnn_features, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            lstm_out, _ = self.lstm(packed)
            lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True, total_length=seq_len)
        else:
            lstm_out, _ = self.lstm(cnn_features)

        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out, key_padding_mask=key_padding_mask
        )

        if lengths is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            pooled = (attn_out * mask).sum(dim=1) / lengths.clamp(min=1).unsqueeze(-1).float()
        else:
            pooled = attn_out.mean(dim=1)

        combined = torch.cat([pooled, static_features], dim=-1)
        return self.fc(combined)


    