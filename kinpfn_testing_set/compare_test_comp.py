import os
import RNA
import pandas as pd
import numpy as np
import subprocess
from itertools import product
from pathlib import Path
# just double checking that the MFE from file and from viennaRNA are the same
INPUT_PARQUET = "dataset.parquet"
df = pd.read_parquet(INPUT_PARQUET)
print(df[['mfe', 'mfe_from_file']].head())