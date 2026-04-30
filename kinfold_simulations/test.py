import RNA

# Test 1: Fold a simple sequence
sequence = "GGGAAACCCUU"
structure, mfe = RNA.fold(sequence)
print(f"Sequence:  {sequence}")
print(f"Structure: {structure}")
print(f"MFE:       {mfe:.2f} kcal/mol")

# Test 2: Partition function
fc = RNA.fold_compound(sequence)
structure, energy = fc.mfe()
prob = fc.pf()
print(f"\nPartition function energy: {prob[1]:.2f} kcal/mol")

print("\nViennaRNA is working correctly!")

import subprocess

result = subprocess.run(
    ["/Users/weberlin/miniconda3/bin/Kinfold", "--num", "10", "--time", "100"],
    input="GGGAAACCC\n(((...)))\n",
    capture_output=True,
    text=True
)
print(result.stdout)
print(result.stderr)


