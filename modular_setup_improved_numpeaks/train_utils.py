"""
train_utils.py
---------------
Model-agnostic training loop, evaluation, seeding, and plotting helpers.
Works with any model that follows the BaseModel forward signature from
models.py: forward(sequence, structure, static_features, lengths).
"""

from __future__ import annotations

import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
#from scipy.spatial.distance import jensenshannon
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # binary classification

import os

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    """One pass over `loader`. If optimizer is given, trains; else evaluates."""
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    n_examples = 0

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for sequences, structures, statics, targets, embeds, lengths in loader:
            sequences = sequences.to(device)
            structures = structures.to(device)
            statics = statics.to(device)
            targets = targets.to(device)
            embeds = embeds.to(device)
            lengths = lengths.to(device)

            if is_train:
                optimizer.zero_grad()

            logits = model(sequences, structures, statics, embeds, lengths)
            # log_probs = torch.log_softmax(logits, dim=-1)
            # loss = criterion(log_probs, targets)
            loss = criterion(logits.squeeze(-1), targets) # BCEWithLogitsLoss does sigmoid and then calculates binary cross-entropy loss (BCE)

            if is_train:
                loss.backward()
                optimizer.step()

            batch_size = targets.shape[0]
            total_loss += loss.item() * batch_size
            n_examples += batch_size

    return total_loss / max(n_examples, 1)


def train_model(
    model,
    train_loader,
    val_loader,
    epochs: int = 100,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    best_checkpoint_path: str | None = None,
    last_checkpoint_path: str | None = None,
    device: str = "cpu",
    verbose: bool = True,
    print_every: int = 5,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    min_lr: float = 1e-6,
    initial_best_val_loss: float = math.inf,
    loss_weight_value: float = 1
):
    """
    Generic training loop. Loss is KLDivLoss since targets are soft
    probability distributions (the binned log-fpt histograms).

    `checkpoint_path`: where to save the best (lowest val loss) model state
    dict. Pass a model-specific path (e.g. f"checkpoints/best_{name}.pth")
    so multiple models being compared don't clobber each other.

    Returns
    -------
    history : dict with keys "train_loss", "val_loss" (lists, one per epoch)
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor,
        patience=scheduler_patience, min_lr=min_lr,
    )

    if os.path.exists(last_checkpoint_path):
        checkpoint = torch.load(last_checkpoint_path)
        if (isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint):
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print('Optimizer successfully restored.')
        if (isinstance(checkpoint, dict) and 'scheduler_state_dict' in checkpoint):
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            print('Scheduler successfully restored.')
    
    #criterion = nn.KLDivLoss(reduction="batchmean")
    #criterion = nn.BCEWithLogitsLoss(reduction="mean") # binary classification

    criterion = nn.BCEWithLogitsLoss(reduction="mean", pos_weight=torch.tensor([loss_weight_value])) # re-weight the minority class

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = initial_best_val_loss

    if best_checkpoint_path is not None:
        Path(best_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    if last_checkpoint_path is not None:
        Path(last_checkpoint_path).parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        train_loss = _run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss = _run_epoch(model, val_loader, criterion, None, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        new_lr = optimizer.param_groups[0]["lr"]

        checkpoint_dict = {'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'scheduler_state_dict': scheduler.state_dict()
                            }
        
        torch.save(checkpoint_dict, last_checkpoint_path)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            if best_checkpoint_path is not None:
                torch.save(checkpoint_dict, best_checkpoint_path)

        if verbose:
            if new_lr != current_lr:
                print(f"Epoch {epoch + 1} | LR dropped: {current_lr:.2e} -> {new_lr:.2e}")
            if improved:
                print(f"Epoch {epoch + 1} | Validation loss improved -> saved checkpoint")
            if epoch % print_every == 0 or epoch == epochs - 1:
                print(
                    f"Epoch {epoch + 1} | Train: {train_loss:.4f} | "
                    f"Val: {val_loss:.4f} | LR: {current_lr:.2e}"
                )

    return history


def evaluate_model(model, loader, device: str = "cpu", checkpoint_path: str | None = None):
    """
    Runs the model over `loader`, returning softmax probabilities, targets,
    and per-example Jensen-Shannon divergence.

    If `checkpoint_path` is given, loads those weights before evaluating
    (use this to load the best-val checkpoint rather than whatever state
    the model happens to be in after training).
    """
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif isinstance(checkpoint, dict):
            model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for sequences, structures, statics, targets, embeds, lengths in loader:
            sequences = sequences.to(device)
            structures = structures.to(device)
            statics = statics.to(device)
            embeds = embeds.to(device)
            lengths = lengths.to(device)

            logits = model(sequences, structures, statics, embeds, lengths)
            #probs = torch.softmax(logits, dim=-1)
            probs = torch.sigmoid(logits) # binary classification

            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.numpy())

    all_preds = np.concatenate(all_preds, axis=0)
    all_preds_rounded = np.round(all_preds)
    all_targets = np.concatenate(all_targets, axis=0)

    # js_scores = np.array(
    #     [jensenshannon(all_targets[i], all_preds[i]) for i in range(len(all_targets))]
    # )

    # binary classification
    acc = accuracy_score(all_targets, all_preds_rounded)
    precision = precision_score(all_targets, all_preds_rounded, pos_label=1)
    recall = recall_score(all_targets, all_preds_rounded, pos_label=1)
    f1 = f1_score(all_targets, all_preds_rounded, pos_label=1)

    return {
        "preds": all_preds,
        "targets": all_targets,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_loss_curve(history, ax=None, label_prefix=""):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.plot(history["train_loss"], label=f"{label_prefix}Train")
    ax.plot(history["val_loss"], label=f"{label_prefix}Val")
    ax.set_xlabel("Epoch")
    #ax.set_ylabel("KL Divergence")
    ax.set_ylabel("Cross-Entropy Loss") # binary classification
    ax.legend()
    return ax


# def plot_prediction_grid(bin_centers, eval_result, n_examples: int = 6, ncols: int = 3):
#     import matplotlib.pyplot as plt

#     y_true = eval_result["targets"]
#     y_pred = eval_result["preds"]
#     js_scores = eval_result["js_scores"]

#     nrows = (n_examples + ncols - 1) // ncols
#     fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
#     axes = np.array(axes).reshape(-1)
#     for i, ax in enumerate(axes[:n_examples]):
#         ax.plot(bin_centers, y_true[i], label="True")
#         ax.plot(bin_centers, y_pred[i], label="Predicted")
#         ax.set_title(f"JS={js_scores[i]:.3f}")
#         ax.set_xlabel("ln(folding time)")
#         ax.set_ylabel("Probability")
#         ax.legend()
#     for ax in axes[n_examples:]:
#         ax.axis("off")
#     plt.tight_layout()
#     return fig
