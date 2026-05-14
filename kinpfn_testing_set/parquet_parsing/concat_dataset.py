import pandas as pd

#fills new columns with NaNs due to outer join, input file1, file2, and output file
def concat_parquet_files(file1_path: str, file2_path: str, output_path: str) -> None:
    df1 = pd.read_parquet(file1_path)
    df2 = pd.read_parquet(file2_path)
    combined = pd.concat([df1, df2], ignore_index=True)
    combined.to_parquet(output_path, index=False)
    print(f"Combined {len(df1)} + {len(df2)} rows → {len(combined)} rows saved to {output_path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python solution.py <file1.parquet> <file2.parquet> <output.parquet>")
        sys.exit(1)
    concat_parquet_files(sys.argv[1], sys.argv[2], sys.argv[3])