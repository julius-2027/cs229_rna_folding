import torch
import fm
import pandas as pd

DATASET_PATH = "/Users/weberlin/src/cs229/cs229_rna_folding/kinpfn_testing_set/parquet_parsing/test_val_dataset.parquet"

model, alphabet = fm.pretrained.rna_fm_t12()
batch_converter = alphabet.get_batch_converter()
model.eval()

df = pd.read_parquet(DATASET_PATH)
data = [(i, seq) for i, seq in enumerate(df['sequence'].tolist())]
lengths = df['length'].tolist()

def pool_embeddings(token_embeddings, lengths):
    B, T, D = token_embeddings.shape
    seq_embeds = token_embeddings[:, 1:-1, :]
    mask = torch.zeros(B, T-2, dtype=torch.bool)
    for i, l in enumerate(lengths):
        mask[i, :l] = True
    mask = mask.unsqueeze(-1)
    masked = seq_embeds * mask
    mean_pooled = masked.sum(dim=1) / mask.sum(dim=1)
    seq_embeds_for_max = seq_embeds.masked_fill(~mask, float('-inf'))
    max_pooled = seq_embeds_for_max.max(dim=1).values
    cls = token_embeddings[:, 0, :]
    return torch.cat([cls, mean_pooled, max_pooled], dim=1)  # (B, 1920)

BATCH_SIZE = 4  # start here, increase to 8 or 16 if stable
all_pooled = []
all_ids = []

for i in range(0, len(data), BATCH_SIZE):
    batch = data[i:i+BATCH_SIZE]
    batch_lengths = lengths[i:i+BATCH_SIZE]
    
    labels, strs, tokens = batch_converter(batch)
    
    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
    
    token_embeddings = results["representations"][12]
    pooled = pool_embeddings(token_embeddings, batch_lengths)
    
    all_pooled.append(pooled)
    all_ids.extend(labels)
    
    print(f"processed {min(i+BATCH_SIZE, len(data))}/{len(data)}")

torch.save({
    'embeddings': torch.cat(all_pooled, dim=0),  # (N, 1920)
    'labels': df['fpts'].tolist(),
    'ids': all_ids,
}, 'fm-rna_embeddings.pt')

print("done!")