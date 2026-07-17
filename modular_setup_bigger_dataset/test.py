import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import jensenshannon

import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, Sampler
from torch.utils.data import Dataset, DataLoader
from data import collate_padded

class MockRNADataset(Dataset):
    def __init__(self):
        # Let's make 5 items. 
        # Sequences will have varying lengths: 3, 3, 5, 5, 2
        self.sequences = [
            np.ones((3, 4)),  # length 3, 4 features
            np.ones((3, 4)) * 2, 
            np.ones((5, 4)) * 3,  # length 5
            np.ones((5, 4)) * 4,
            np.ones((2, 4)) * 5   # length 2
        ]
        # Matching structure dimensions
        self.structures = [np.zeros((len(s), 2)) for s in self.sequences]
        self.static_features = [np.array([1.0, 2.0]) for _ in self.sequences]
        self.targets = [np.array([0.1, 0.2, 0.3]) for _ in self.sequences]

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.structures[idx], self.static_features[idx], self.targets[idx]

# Instantiate it
dataset = MockRNADataset()

print("\n=== TESTING STRATEGY 2: PADDED BATCHES ===")

# Use a standard DataLoader, forcing a batch of 3 mixed lengths
loader_padded = DataLoader(dataset, batch_size=3, shuffle=False, collate_fn=collate_padded)

# Grab just the first batch to inspect
first_batch = next(iter(loader_padded))
seqs, structs, statics, targets, lengths = first_batch

print(f"Padded Sequence Shape: {seqs.shape}") # Should scale to the max length in this batch (5)
print(f"Reported original lengths: {lengths.tolist()}") 

# Let's print the actual values of the first sample (which originally had length 3)
# It should be 1.0 for the first 3 steps, and 0.0 for the remaining 2 steps (padded)
print("\nInspecting first sample features across time steps to check for trailing zeros:")
for step in range(seqs.shape[0]):
    print(f" -> Step {step}: {seqs[0, step].tolist()}")