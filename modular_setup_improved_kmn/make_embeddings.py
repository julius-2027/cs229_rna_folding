import torch
import fm
import pandas as pd
import os

# Weber
#DATASET_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding/kinpfn_testing_set/parquet_parsing/test_val_dataset.parquet"

# generic
#DATASET_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding_clone/kinfold_simulations/feature_extraction/synthetic+real_dataset_no_keys.parquet"
DATASET_PATH = "../kinfold_simulations/feature_extraction/synthetic+real_dataset.parquet"

OUTPUT_PATH = 'all_fm-rna_embeddings.pt'

# Load RNA-FM
model, alphabet = fm.pretrained.rna_fm_t12('../RNA-FM_pretrained/RNA-FM_pretrained.pth')
batch_converter = alphabet.get_batch_converter()
model.eval()

# Move model to GPU if available for a 10x speedup
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
model = model.to(device)
print(f"Running RNA-FM extraction on device: {device}")

df = pd.read_parquet(DATASET_PATH).reset_index(drop=True)

# CRITICAL: Pair the exact Dataframe string Index Key with the sequence 
data = [(str(idx), row['sequence']) for idx, row in df.iterrows()]
lengths = df['length'].tolist()

BATCH_SIZE = 16  # Increased from 4 since we are discarding tracking graphs with no_grad
embedding_dict = {}

num_sequences = len(data)

print(f"Starting extraction for {num_sequences} sequences...")

for i in range(0, num_sequences, BATCH_SIZE):
    batch = data[i : i + BATCH_SIZE]
    batch_lengths = lengths[i : i + BATCH_SIZE]
    
    # labels now contain your exact, true dataframe string indices
    labels, strs, tokens = batch_converter(batch)
    tokens = tokens.to(device)
    
    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
        
    # Shape: (Batch_Size, Padded_Token_Len, 640)
    token_embedding = results["representations"][12]
    
    for idx, true_length in enumerate(batch_lengths):
        row_key = labels[idx]
        
        # FIX: Slice from 1 to true_length + 1 to bypass <cls> at index 0 
        # and isolate ONLY the actual sequence nucleotides!
        true_embedding = token_embedding[idx, 1 : true_length + 1, :].cpu()
        
        # Compress to float16 to save huge amounts of SSD memory space
        embedding_dict[row_key] = true_embedding.to(torch.float16)
        
    print(f"Processed {min(i + BATCH_SIZE, num_sequences)}/{num_sequences}")

# Save as a clean lookup map matching what your RNADataset expects
torch.save(embedding_dict, OUTPUT_PATH)
print(f"Done! Saved unified lookup dictionary to {OUTPUT_PATH}")