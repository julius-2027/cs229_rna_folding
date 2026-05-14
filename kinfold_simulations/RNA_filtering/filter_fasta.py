import random
from collections import defaultdict
from Bio import SeqIO
random.seed(67)
input_file = "RNA_unfiltered.fasta"
#TAXONOMY:"9606" AND entry_type:"Sequence" AND so_rna_type_name:"NcRNA" AND length:[15 TO 100]

output_file = "sampled_4x.fasta"
count = 250
# Store sequences by length
seqs_by_length = defaultdict(list)

for record in SeqIO.parse(input_file, "fasta"):
    length = len(record.seq)
    if 15 <= length <= 100:
        seqs_by_length[length].append(record)

# Sample up to 4 per length
sampled_records = []

for length in range(15, 101):
    records = seqs_by_length[length]

    if len(records) >= count:
        sampled = random.sample(records, count)
    else:
        sampled = records

    sampled_records.extend(sampled)

# Write output
SeqIO.write(sampled_records, output_file, "fasta")

print(f"Wrote {len(sampled_records)} sequences to {output_file}")