"""
tree_edit_distance.py
------------------------
Tree edit distance between two RNA secondary structures in dot-bracket
notation, via ViennaRNA's RNAdistance CLI (same approach used elsewhere in
this repo, e.g. kinfold_simulations/feature_extraction/add_features.py).

RNAdistance represents each structure as a tree and reports several distance
metrics; "f:" is the full tree edit distance (the one used throughout this
project).

Usage:
    python tree_edit_distance.py "((..))" "(....)"

Or import directly:
    from tree_edit_distance import tree_distance
    tree_distance("((..))", "(....)")
"""

import subprocess
import sys

import numpy as np


def tree_distance(struct1: str, struct2: str) -> int:
    """Full tree edit distance between two dot-bracket structures."""
    inp = f"{struct1}\n{struct2}\n"
    result = subprocess.run(
        ["RNAdistance"],
        input=inp,
        capture_output=True,
        text=True,
    )
    line = result.stdout.strip()  # e.g. "f: 12"
    try:
        return int(line.split(":")[1].strip())
    except (IndexError, ValueError):
        return np.nan


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python tree_edit_distance.py <dot_bracket_1> <dot_bracket_2>")
        sys.exit(1)

    struct1, struct2 = sys.argv[1], sys.argv[2]
    print(tree_distance(struct1, struct2))
