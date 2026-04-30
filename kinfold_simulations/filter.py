import random
from collections import defaultdict
from Bio import SeqIO
random.seed(67)
input_file = "RNA_unfiltered.fasta"
output_file = "sampled_4x.fastanew"

# Store sequences by length
seqs_by_length = defaultdict(list)

for record in SeqIO.parse(input_file, "fasta"):
    length = len(record.seq)
    if 50 <= length <= 100:
        seqs_by_length[length].append(record)

# Sample up to 4 per length
sampled_records = []

for length in range(50, 101):
    records = seqs_by_length[length]

    if len(records) >= 4:
        sampled = random.sample(records, 4)
    else:
        sampled = records  # take all if fewer than 4

    sampled_records.extend(sampled)

# Write output
SeqIO.write(sampled_records, output_file, "fasta")

print(f"Wrote {len(sampled_records)} sequences to {output_file}")