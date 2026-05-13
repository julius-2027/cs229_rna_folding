import random
import pandas as pd

random.seed(67)
input_file = "rna_unfiltered.parquet"
count = 250
output_file = f"sampled_{count}x.parquet"

df = pd.read_parquet(input_file)

# Convert T -> U for RNA tools
df['sequence'] = df['sequence'].str.replace('T', 'U', regex=False)

df['length'] = df['sequence'].str.len()
df = df[df['length'].between(15, 100)]

sampled = (
    df.groupby('length', group_keys=False)
    .apply(lambda g: g.sample(min(len(g), count), random_state=67))
    .reset_index(drop=True)
)

sampled.to_parquet(output_file, index=False)
print(f"Wrote {len(sampled)} sequences to {output_file}")
print(sampled[['rnacentral_id', 'rna_type', 'sequence', 'length']].head())