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

BATCH_SIZE = 4 
all_ids = []

token_embeddings = []
for i in range(0, 2 * BATCH_SIZE, BATCH_SIZE):
    batch = data[i:i+BATCH_SIZE]
    batch_lengths = lengths[i:i+BATCH_SIZE]
    print(batch_lengths)
    labels, strs, tokens = batch_converter(batch)
    
    with torch.no_grad():
        results = model(tokens, repr_layers=[12])
    token_embedding = results["representations"][12]
    for idx, true_length in enumerate(batch_lengths):
        # RNA-FM typically adds a <cls> token at index 0 and an <eos> token at the end. keep cls
        true_embedding = token_embedding[idx, 0:true_length+1, :].cpu() 
        token_embeddings.append(true_embedding)
    all_ids.extend(labels)
    
    print(f"processed {min(i+BATCH_SIZE, len(data))}/{len(data)}")

torch.save({
    'embeddings': token_embeddings,  # (N, 1920)
    'fpts': df['fpts'].tolist(),
    'ids': all_ids,
}, 'fm-rna_embeddings.pt')

print("done!")