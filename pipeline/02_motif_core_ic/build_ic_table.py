"""data/ic_motifs_data.parquet (Fig. 1D-G): the per-motif summaries of hit_core_ic.py
({PAPER,ALL_BPNET}/{dir}/hit_core_ic_summary_counts_0p7.csv) of all 2,282 models, concatenated. CPU job."""
import glob
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'modules'))
import paths

files = (glob.glob(f'{paths.PAPER}/*/*/hit_core_ic_summary_counts_0p7.csv')
         + glob.glob(f'{paths.ALL_BPNET}/*/*/hit_core_ic_summary_counts_0p7.csv'))
summ = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
assert summ.drop_duplicates(['bpnet_set', 'dir']).shape[0] == len(files) == 2282, len(files)
paths.DATA.mkdir(exist_ok=True)
summ.to_parquet(paths.DATA / 'ic_motifs_data.parquet')
print(summ.shape, 'models', len(files))
