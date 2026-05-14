import os
import RNA
import pandas as pd
import numpy as np
import subprocess
from itertools import product
from pathlib import Path
# just double checking that the MFE from file and from viennaRNA are the same
INPUT_PARQUET = "test.parquet"
df = pd.read_parquet(INPUT_PARQUET)
print(df[df['length'] == 140][['n_local_minima', 'min_4_energy']])