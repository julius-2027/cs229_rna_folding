import pandas as pd
import torch

# ── paths ─────────────────────────────────────────────────────────────────


def main():
    synthetic = pd.read_parquet("/Users/weberlin/src/cs229/cs229_rna_folding_clone/kinpfn_testing_set/parquet_parsing/test_val_dataset.parquet")
    real = pd.read_parquet("/Users/weberlin/src/cs229/cs229_rna_folding_clone/kinfold_simulations/SLURM/dataset_with_fpts_subset.parquet")
    synthetic['dataset'] = 'synthetic'
    real['dataset'] = 'rna_central'

    # Concatenate the DataFrames and reset the index
    combined_df = pd.concat([synthetic, real], ignore_index=True).reset_index()
    
    combined_df.to_parquet("synthetic+real_dataset.parquet", index=True)

if __name__ == "__main__":
    main()
