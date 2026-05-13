import RNA
import pandas as pd
import numpy as np
import subprocess
from itertools import product

# ── constants ─────────────────────────────────────────────────────────────────
INPUT_PARQUET = "sampled_250x.parquet"
OUTPUT_PARQUET = "test_10_features.parquet"   # overwrite in place
SUBOPT_DELTA   = 200                      # 2.0 kcal/mol window (deka-calories)
MAX_MINIMA     = 20                       # fixed max local minima per sequence
NUCLEOTIDES    = ["A", "U", "G", "C"]
DINUCLEOTIDES  = ["".join(p) for p in product(NUCLEOTIDES, repeat=2)]  # all 16


# ── helpers ───────────────────────────────────────────────────────────────────

def gc_content(seq: str) -> float:
    return (seq.count("G") + seq.count("C")) / len(seq)


def nucleotide_frequencies(seq: str) -> dict:
    n = len(seq)
    return {nt: seq.count(nt) / n for nt in NUCLEOTIDES}


def dinucleotide_frequencies(seq: str) -> dict:
    n = len(seq) - 1
    if n <= 0:
        return {di: 0.0 for di in DINUCLEOTIDES}
    return {di: sum(seq[i:i+2] == di for i in range(n)) / n for di in DINUCLEOTIDES}


def get_mfe_and_structure(seq: str):
    md = RNA.md()
    md.uniq_ML = 1
    fc = RNA.fold_compound(seq, md)
    structure, mfe = fc.mfe()
    return fc, structure, mfe


def get_subopt_structures(fc, mfe_structure: str, delta: int = SUBOPT_DELTA):
    """Return list of (structure, energy, bp_distance_to_mfe) sorted by energy."""
    results = fc.subopt(delta)
    subopt = []
    for sol in results:
        if sol.structure is None:
            continue
        if sol.structure == mfe_structure:
            continue                      # skip the MFE itself
        dist = RNA.bp_distance(mfe_structure, sol.structure)
        subopt.append((sol.structure, sol.energy, dist))
    subopt.sort(key=lambda x: x[1])      # sort by energy ascending
    return subopt


def tree_distance(struct1: str, struct2: str) -> int:
    """Call RNAdistance and parse the 'f: N' output."""
    inp = f"{struct1}\n{struct2}\n"
    result = subprocess.run(
        ["RNAdistance"],
        input=inp,
        capture_output=True,
        text=True
    )
    line = result.stdout.strip()         # e.g. "f: 12"
    try:
        return int(line.split(":")[1].strip())
    except (IndexError, ValueError):
        return np.nan


def pad(lst, length=MAX_MINIMA, fill=np.nan):
    return lst[:length] + [fill] * max(0, length - len(lst))


# ── per-sequence feature extraction ───────────────────────────────────────────

def extract_features(seq: str) -> dict:
    features = {}

    # GC content
    features["gc_content"] = gc_content(seq)

    # Nucleotide frequencies
    for nt, freq in nucleotide_frequencies(seq).items():
        features[f"freq_{nt}"] = freq

    # Dinucleotide frequencies
    for di, freq in dinucleotide_frequencies(seq).items():
        features[f"freq_{di}"] = freq

    # MFE and MFE dot-bracket
    fc, mfe_struct, mfe = get_mfe_and_structure(seq)
    features["mfe"] = mfe
    features["mfe_structure"] = mfe_struct

    # Suboptimal structures (local minima proxy)
    subopt = get_subopt_structures(fc, mfe_struct)
    features["n_local_minima"] = len(subopt)

    # Padded per-minima columns
    structures = pad([s[0] for s in subopt])
    energies   = pad([s[1] for s in subopt])
    bp_dists   = pad([s[2] for s in subopt])

    for i in range(MAX_MINIMA):
        features[f"min_{i+1}_structure"] = structures[i]
        features[f"min_{i+1}_energy"]    = energies[i]
        features[f"min_{i+1}_bp_dist"]   = bp_dists[i]

    # Tree distance (RNAdistance) from MFE to each local minimum
    for i, sol in enumerate(subopt[:MAX_MINIMA]):
        features[f"min_{i+1}_tree_dist"] = tree_distance(mfe_struct, sol[0])
    # Pad remaining tree distances
    for i in range(len(subopt), MAX_MINIMA):
        features[f"min_{i+1}_tree_dist"] = np.nan

    return features


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_parquet(INPUT_PARQUET)
    print(f"Loaded {len(df)} sequences from {INPUT_PARQUET}")

    feature_rows = []
    for idx, row in df.iterrows():
        seq = row["sequence"]
        if idx % 100 == 0:
            print(f"  Processing {idx}/{len(df)}...")
        try:
            features = extract_features(seq)
        except Exception as e:
            print(f"  WARNING: failed on index {idx} ({seq[:20]}...): {e}")
            features = {}
        feature_rows.append(features)

    feat_df = pd.DataFrame(feature_rows)
    out_df  = pd.concat([df.reset_index(drop=True), feat_df], axis=1)

    out_df.to_parquet(OUTPUT_PARQUET, index=False)
    print(f"\nDone. Wrote {len(out_df)} rows to {OUTPUT_PARQUET}")
    print(f"Columns: {out_df.columns.tolist()}")


if __name__ == "__main__":
    main()
