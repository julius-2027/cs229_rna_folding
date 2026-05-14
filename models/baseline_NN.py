import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── 1. Load ──────────────────────────────────────────────────────────────────

def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)


# ── 2. Feature extraction ────────────────────────────────────────────────────

N_MINIMA = 20

# Scalar columns with no nulls (or trivially filled)
SCALAR_COLS = [
    "length", "gc_content",
    "freq_A", "freq_U", "freq_G", "freq_C",
    "freq_AA", "freq_AU", "freq_AG", "freq_AC",
    "freq_UA", "freq_UU", "freq_UG", "freq_UC",
    "freq_GA", "freq_GU", "freq_GG", "freq_GC",
    "freq_CA", "freq_CU", "freq_CG", "freq_CC",
    "mfe", "n_local_minima",
]

# Per-minimum numeric cols (nullable → fill with 0, mask separately)
PER_MIN_NUM = ["energy", "bp_dist", "tree_dist"]


def encode_structure(s) -> np.ndarray:
    """
    Dot-bracket string → fixed-length 3-channel summary:
      ch0: fraction of '('
      ch1: fraction of ')'
      ch2: fraction of '.'
    Returns zeros for null/empty.
    """
    if not isinstance(s, str) or len(s) == 0:
        return np.zeros(3, dtype=np.float32)
    arr = np.array(list(s))
    n = len(arr)
    return np.array([
        (arr == '(').sum() / n,
        (arr == ')').sum() / n,
        (arr == '.').sum() / n,
    ], dtype=np.float32)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    """
    Returns float32 array of shape (N, D) where D is:
      - len(SCALAR_COLS)                          = 24
      - encode_structure(mfe_structure)            =  3
      - per minimum: energy, bp_dist, tree_dist,
                     structure encoding (3),
                     present_mask (1)             =  7  × 20 = 140
    Total: 24 + 3 + 140 = 167
    """
    N = len(df)
    parts = []

    # Scalar features
    scalars = df[SCALAR_COLS].fillna(0.0).values.astype(np.float32)
    parts.append(scalars)  # (N, 24)

    # MFE structure encoding
    mfe_struct = np.stack(
        df["mfe_structure"].apply(encode_structure).values
    )  # (N, 3)
    parts.append(mfe_struct)

    # Per-minimum features
    for i in range(1, N_MINIMA + 1):
        e_col    = f"min_{i}_energy"
        bp_col   = f"min_{i}_bp_dist"
        tr_col   = f"min_{i}_tree_dist"
        st_col   = f"min_{i}_structure"

        present  = df[e_col].notna().values.astype(np.float32).reshape(N, 1)
        energy   = df[e_col].fillna(0.0).values.astype(np.float32).reshape(N, 1)
        bp_dist  = df[bp_col].fillna(0.0).values.astype(np.float32).reshape(N, 1)
        tr_dist  = df[tr_col].fillna(0.0).values.astype(np.float32).reshape(N, 1)
        struct   = np.stack(df[st_col].apply(encode_structure).values)  # (N, 3)

        parts.append(energy)    # 1
        parts.append(bp_dist)   # 1
        parts.append(tr_dist)   # 1
        parts.append(struct)    # 3
        parts.append(present)   # 1  ← tells model "this minimum exists"

    return np.concatenate(parts, axis=1)  # (N, 167)


# ── 3. Bin edges ─────────────────────────────────────────────────────────────

def compute_global_bins(df: pd.DataFrame, n_bins: int = 50) -> np.ndarray:
    all_fpts = np.concatenate(df["fpts"].values)
    all_fpts = all_fpts[all_fpts > 0]
    log_fpts = np.log10(all_fpts)
    return np.linspace(log_fpts.min(), log_fpts.max(), n_bins + 1)


def fpt_to_histogram(fpts, bins: np.ndarray) -> np.ndarray:
    arr = np.asarray(fpts, dtype=np.float64)
    arr = arr[arr > 0]
    counts, _ = np.histogram(np.log10(arr), bins=bins)
    return counts.astype(np.float32) / max(counts.sum(), 1)


# ── 4. Dataset ───────────────────────────────────────────────────────────────

class FPTDataset(Dataset):
    def __init__(self, df: pd.DataFrame, bins: np.ndarray):
        self.X = extract_features(df)
        self.y = np.stack([fpt_to_histogram(r, bins) for r in df["fpts"].values])

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


# ── 5. Model ──────────────────────────────────────────────────────────────────

class FPTNet(nn.Module):
    """
    Simple MLP with residual connections.
    in_dim → 512 → 512 → 256 → n_bins (softmax)
    """
    def __init__(self, in_dim: int, hidden: int = 512, n_bins: int = 50):
        super().__init__()

        self.input_proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )

        self.res_block1 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        self.res_block2 = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden),
            nn.LayerNorm(hidden),
        )

        self.head = nn.Sequential(
            nn.GELU(),
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Linear(256, n_bins),
        )

    def forward(self, x):
        x = self.input_proj(x)
        x = F.gelu(x + self.res_block1(x))
        x = F.gelu(x + self.res_block2(x))
        return F.softmax(self.head(x), dim=-1)


# ── 6. Normalisation ─────────────────────────────────────────────────────────

def compute_stats(X: np.ndarray):
    mean = X.mean(axis=0)
    std  = X.std(axis=0) + 1e-8
    return mean, std


# ── 7. Training ───────────────────────────────────────────────────────────────

def kl_loss(pred, target, eps=1e-8):
    return F.kl_div((pred + eps).log(), target + eps, reduction="batchmean")


def train(
    parquet_path: str,
    n_bins: int   = 50,
    epochs: int   = 100,
    batch_size    = 64,
    lr: float     = 1e-3,
    val_split     = 0.1,
    device: str   = "cpu",
):
    df   = load_parquet(parquet_path)
    bins = compute_global_bins(df, n_bins)

    # shuffle & split
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    n_val    = max(1, int(len(df) * val_split))
    df_val   = df.iloc[:n_val]
    df_train = df.iloc[n_val:]

    # normalise features using train stats only
    X_train_raw = extract_features(df_train)
    mean, std   = compute_stats(X_train_raw)

    class NormDataset(FPTDataset):
        def __init__(self, df, bins, mean, std):
            super().__init__(df, bins)
            self.X = ((self.X - mean) / std).astype(np.float32)

    train_ds = NormDataset(df_train, bins, mean, std)
    val_ds   = NormDataset(df_val,   bins, mean, std)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=2)

    in_dim = train_ds.X.shape[1]
    model  = FPTNet(in_dim=in_dim, n_bins=n_bins).to(device)
    opt    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best_val = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for X, y in train_dl:
            X, y = X.to(device), y.to(device)
            opt.zero_grad()
            loss = kl_loss(model(X), y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += loss.item() * len(X)
        train_loss /= len(train_ds)

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, y in val_dl:
                X, y = X.to(device), y.to(device)
                val_loss += kl_loss(model(X), y).item() * len(X)
        val_loss /= len(val_ds)
        sched.step()

        if val_loss < best_val:
            best_val = val_loss
            torch.save({
                "model": model.state_dict(),
                "bins": bins, "mean": mean, "std": std, "in_dim": in_dim,
            }, "fpt_best.pt")

        print(f"Epoch {epoch:3d}/{epochs}  train={train_loss:.4f}  val={val_loss:.4f}"
              + ("  ✓" if val_loss == best_val else ""))

    print(f"\nBest val KL: {best_val:.4f} — saved to fpt_best.pt")
    return model, bins, mean, std


# ── 8. Inference helper ───────────────────────────────────────────────────────

def predict(model, df: pd.DataFrame, mean, std, bins, device="cpu"):
    model.eval()
    X = (extract_features(df) - mean) / std
    X = torch.tensor(X, dtype=torch.float32).to(device)
    with torch.no_grad():
        return model(X).cpu().numpy()  # (N, 50) predicted histograms


# ── 9. Entrypoint ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    path   = sys.argv[1] if len(sys.argv) > 1 else "data.parquet"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    train(path, device=device)