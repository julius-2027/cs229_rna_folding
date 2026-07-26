import pandas as pd
import numpy as np

"""
Build normalized log-fpt histograms as soft targets.

Returns
-------
y : np.ndarray of shape (n_rows, n_bins)
bin_edges : np.ndarray of shape (n_bins + 1,)
"""
n_bins = 50
df = pd.read_parquet('../kinfold_simulations/feature_extraction/synthetic+real_dataset.parquet')
arr_fpts = np.array([row for row in df['fpts']])
# remove zeros for log behavior
nonzero = arr_fpts[arr_fpts != 0]
log_all = np.log(nonzero)
# !!!!! objective changes slightly with dataset. once we have dataset compiled this should be fine.#
bin_edges = np.linspace(log_all.min(), log_all.max(), n_bins + 1)

dists = []
for _, row in df.iterrows():
    fpts = row['fpts']
    nz = fpts[fpts != 0]
    logfpts = np.log(nz)
    hist, _ = np.histogram(logfpts, bin_edges, density=True)
    hist /= np.sum(hist)
    dists.append(hist)

# Assign the whole column at once (as a list of arrays) so pandas boxes each
# histogram as one object-dtype cell instead of trying to broadcast an array
# into a single scalar cell via .loc, which is what raised the ValueError.
df['dist'] = dists
#df.drop(columns=['fpts'], inplace=True)
df.to_parquet('synthetic+real_dataset.parquet', index=True)
np.save('bin_edges.npy', bin_edges)


# """
# RNA types + distribution: 
# tRNA             2568
# piRNA            1864
# miRNA            1683
# snoRNA            862
# snRNA             480
# ncRNA             219
# Y_RNA             206
# sRNA              164
# rRNA              138
# lncRNA             74
# antisense_RNA      26
# scRNA              13
# vault_RNA          11
# ribozyme            1
# scaRNA              1
# pre_miRNA           1
# """