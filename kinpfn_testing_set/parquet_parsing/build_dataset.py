"""
build_dataset.py

Walks the directory tree:
    root/
        short/
            15/  16/ ... 30/
        middle/
            31/  ...
        long/
            ...

Each leaf folder contains CSV-like files (no header) where every row is:
    dot_bracket, mfe, fpt, label, sequence
and all rows in a file share the same sequence (only fpt changes).

Outputs a single parquet with:
    - sequence features (GC, nt freqs, dinuc freqs, MFE, MFE structure)
    - suboptimal structure features (energy, bp_dist, tree_dist) padded to MAX_MINIMA
    - fpts: list of 1000 first passage times
"""

import os
import RNA
import pandas as pd
import numpy as np
import subprocess
from itertools import product
from pathlib import Path

# ── config ────────────────────────────────────────────────────────────────────
ROOT_DIR       = "/Users/weberlin/src/cs229/cs229_rna_folding/neps_validation_set/all_seq_valid"  # change to your root folder path
OUTPUT_PARQUET = "val_dataset.parquet"
SUBOPT_DELTA   = 200             # 2.0 kcal/mol (deka-calories)
MAX_MINIMA     = 20
N_FPTS         = 1000

NUCLEOTIDES   = ["A", "U", "G", "C"]
DINUCLEOTIDES = ["".join(p) for p in product(NUCLEOTIDES, repeat=2)]


# ── FPT file parser ───────────────────────────────────────────────────────────

def parse_fpt_file(filepath: Path) -> dict:
    """
    Parse a single FPT file. Returns dict with:
        sequence, mfe_from_file, dot_bracket_from_file, fpts (list of floats)
    Returns None if file is malformed.
    """
    fpts = []
    sequence = None
    mfe_from_file = None
    dot_bracket_from_file = None

    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 5:
                continue
            dot_bracket_from_file = parts[0].strip()
            mfe_from_file         = float(parts[1].strip())
            fpt                   = float(parts[2].strip())
            # parts[3] is label (X1), skip
            sequence              = parts[4].strip()
            fpts.append(fpt)

    if sequence is None or len(fpts) == 0:
        return None

    return {
        "sequence":            sequence,
        "mfe_from_file":       mfe_from_file,
        "dot_bracket_from_file": dot_bracket_from_file,
        "fpts":                fpts,
    }


# ── feature helpers ───────────────────────────────────────────────────────────

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


def get_subopt_structures(fc, mfe_structure: str):
    results = fc.subopt(SUBOPT_DELTA)
    subopt = []
    for sol in results:
        if sol.structure is None or sol.structure == mfe_structure:
            continue
        dist = RNA.bp_distance(mfe_structure, sol.structure)
        subopt.append((sol.structure, sol.energy, dist))
    subopt.sort(key=lambda x: x[1])
    return subopt


def tree_distance(struct1: str, struct2: str) -> float:
    inp = f"{struct1}\n{struct2}\n"
    result = subprocess.run(
        ["RNAdistance"],
        input=inp,
        capture_output=True,
        text=True
    )
    line = result.stdout.strip()   # "f: 12"
    try:
        return int(line.split(":")[1].strip())
    except (IndexError, ValueError):
        return np.nan


def pad(lst, length=MAX_MINIMA, fill=np.nan):
    return lst[:length] + [fill] * max(0, length - len(lst))


# ── per-sequence feature extraction ───────────────────────────────────────────

def extract_features(seq: str) -> dict:
    features = {}

    features["gc_content"] = gc_content(seq)

    for nt, freq in nucleotide_frequencies(seq).items():
        features[f"freq_{nt}"] = freq

    for di, freq in dinucleotide_frequencies(seq).items():
        features[f"freq_{di}"] = freq

    fc, mfe_struct, mfe = get_mfe_and_structure(seq)
    features["mfe"]           = mfe
    features["mfe_structure"] = mfe_struct

    subopt = get_subopt_structures(fc, mfe_struct)
    features["n_local_minima"] = len(subopt)

    structures = pad([s[0] for s in subopt])
    energies   = pad([s[1] for s in subopt])
    bp_dists   = pad([s[2] for s in subopt])

    for i in range(MAX_MINIMA):
        features[f"min_{i+1}_structure"] = structures[i]
        features[f"min_{i+1}_energy"]    = energies[i]
        features[f"min_{i+1}_bp_dist"]   = bp_dists[i]

    for i, sol in enumerate(subopt[:MAX_MINIMA]):
        features[f"min_{i+1}_tree_dist"] = tree_distance(mfe_struct, sol[0])
    for i in range(len(subopt), MAX_MINIMA):
        features[f"min_{i+1}_tree_dist"] = np.nan

    return features


# ── directory walker ──────────────────────────────────────────────────────────

def iter_fpt_files(root: str):
    """
    Yield all files under root, regardless of nesting depth.
    Skips hidden files and directories.
    """
    for dirpath, dirnames, filenames in os.walk(root):
        # skip hidden dirs in-place
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for fname in filenames:
            if fname.startswith("."):
                continue
            yield Path(dirpath) / fname


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    root = Path(ROOT_DIR)
    if not root.exists():
        raise FileNotFoundError(f"ROOT_DIR '{ROOT_DIR}' does not exist.")

    all_files = sorted(iter_fpt_files(str(root)))
    print(f"Found {len(all_files)} files under {root}")

    rows = []
    for i, filepath in enumerate(all_files):
        if i % 500 == 0:
            print(f"  [{i}/{len(all_files)}] {filepath}")

        parsed = parse_fpt_file(filepath)
        if parsed is None:
            print(f"  WARNING: skipping malformed file {filepath}")
            continue

        seq = parsed["sequence"]

        try:
            features = extract_features(seq)
        except Exception as e:
            print(f"  WARNING: feature extraction failed for {filepath}: {e}")
            continue

        row = {
            "filepath":             str(filepath),
            "sequence":             seq,
            "length":               len(seq),
            "mfe_from_file":        parsed["mfe_from_file"],
            "dot_bracket_from_file": parsed["dot_bracket_from_file"],
            "fpts":                 parsed["fpts"],
            **features,
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(OUTPUT_PARQUET, index=False)

    print(f"\nDone. Wrote {len(df)} rows → {OUTPUT_PARQUET}")
    print(f"Columns ({len(df.columns)}): {df.columns.tolist()}")


if __name__ == "__main__":
    main()
