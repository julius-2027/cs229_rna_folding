import numpy as np
import torch
import data
from data import make_dataloader, prepare_data

SEED = 42
N_BINS = 50
DATA_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding/kinpfn_testing_set/parquet_parsing/test_val_dataset.parquet"

class SequenceLengthMeanPredictor:
    """
    A pure statistical baseline model. 
    It ignores static features and sequence characters. It simply groups 
    training examples by sequence length, calculates the mean target histogram 
    for each length group, and uses that group mean as the prediction.
    """
    def __init__(self, output_dim: int):
        self.output_dim = output_dim
        # Maps sequence_length -> mean_histogram_tensor
        self.length_means = {}
        # Global fallback mean if we encounter an unseen test sequence length
        self.global_mean = None

    def fit(self, train_loader):
        """Accumulates targets grouped by sequence length from the training set."""
        length_totals = {}
        length_counts = {}
        all_targets = []

        # Iterate through the entire train dataloader once
        for _, _, _, targets, lengths in train_loader:
            # Convert tensors to numpy for easy manipulation
            targets_np = targets.cpu().numpy()
            lengths_np = lengths.cpu().numpy()
            
            for target, length in zip(targets_np, lengths_np):
                length = int(length)
                all_targets.append(target)
                
                if length not in length_totals:
                    length_totals[length] = np.zeros(self.output_dim)
                    length_counts[length] = 0
                    
                length_totals[length] += target
                length_counts[length] += 1

        # Calculate conditional means per length
        for length in length_totals:
            self.length_means[length] = torch.tensor(
                length_totals[length] / length_counts[length], 
                dtype=torch.float32
            )
            
        # Calculate global fallback mean
        self.global_mean = torch.tensor(
            np.mean(all_targets, axis=0), 
            dtype=torch.float32
        )
        print(f"Mean baseline compiled. Learned averages for {len(self.length_means)} unique sequence lengths.")

    def predict(self, lengths_tensor):
        """Generates predictions based entirely on sequence lengths."""
        batch_size = lengths_tensor.shape[0]
        predictions = torch.zeros((batch_size, self.output_dim))
        
        for i, length in enumerate(lengths_tensor):
            length_item = int(length)
            if length_item in self.length_means:
                predictions[i] = self.length_means[length_item]
            else:
                predictions[i] = self.global_mean
                
        return predictions

handpicked_cols = ["gc_content", "mfe", "n_local_minima"]
data = prepare_data(DATA_PATH, n_bins=N_BINS, random_state=SEED, handpicked_cols=handpicked_cols)
static_dim = data.train.static_features.shape[1]
output_dim = data.train.targets.shape[1]
bin_centers = (data.bin_edges[:-1] + data.bin_edges[1:]) / 2

print(f"Train/Val/Test sizes: {len(data.train.sequences)}/{len(data.val.sequences)}/{len(data.test.sequences)}")
print(f"Static feature dim: {static_dim}, output dim: {output_dim}")

# Assuming you already ran prepare_data and have your test_loader
test_loader = make_dataloader(data.test, batching="padded", batch_size=32, shuffle=False)

# 1. Instantiate and "Train" (Fit) the baseline using the training data
output_dim = 50
mean_baseline = SequenceLengthMeanPredictor(output_dim=output_dim)
# We can use train_loader to calculate the lengths statistics
train_loader = make_dataloader(data.train, batching="padded", batch_size=32, shuffle=False)
mean_baseline.fit(train_loader)

# 2. Evaluate performance manually over the test set
from scipy.spatial.distance import jensenshannon

js_divergences = []

with torch.no_grad():
    for _, _, _, targets, lengths in test_loader:
        # Get predictions based entirely on the lengths of this batch
        preds = mean_baseline.predict(lengths)
        
        # Calculate JS Divergence metrics for each item in the batch
        for p, t in zip(preds.numpy(), targets.numpy()):
            # jensenshannon returns the distance; square it to get the divergence if required
            js_dist = jensenshannon(p, t)
            if not np.isnan(js_dist):
                js_divergences.append(js_dist)

print("\n--- Pure Mean Baseline Performance ---")
print(f"Test JS Mean:   {np.mean(js_divergences):.4f}")
print(f"Test JS Median: {np.median(js_divergences):.4f}")