import pandas as pd

df = pd.read_parquet('/Users/weberlin/src/cs229/cs229_rna_folding_clone/kinfold_simulations/SLURM/dataset_with_fpts_subset.parquet')
print(df.iloc[:, 3].value_counts())
"""
RNA types + distribution: 
tRNA             2568
piRNA            1864
miRNA            1683
snoRNA            862
snRNA             480
ncRNA             219
Y_RNA             206
sRNA              164
rRNA              138
lncRNA             74
antisense_RNA      26
scRNA              13
vault_RNA          11
ribozyme            1
scaRNA              1
pre_miRNA           1
"""