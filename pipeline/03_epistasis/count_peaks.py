import sys, glob, gzip, pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
import os
rows = []
for s in ['paper', 'all']:
    os.environ['BPNET_SET'] = s
    import importlib, bpnet_sets; importlib.reload(bpnet_sets)
    base, meta = bpnet_sets.load()
    for r in meta.itertuples():
        d = f'{base}/{r.dir}'
        pk = glob.glob(f'{d}/training_and_test_regions__*/peaks.all_input_regions.*.bed.gz')
        tsv = glob.glob(f'{d}/sequence_motifs_instances__*/counts/*lambda_0p7/*.tsv')
        h5 = glob.glob(f'{d}/models_*/fold_*/*.h5')
        n = sum(1 for _ in gzip.open(pk[0])) if pk else -1
        rows.append(dict(set=s, dir=r.dir, n_peaks=n, n_tsv=len(tsv), n_h5=len(h5)))
df = pd.DataFrame(rows); df.to_csv(_Path(__file__).resolve().parent / 'tasks' / 'peak_counts.csv', index=False)
print(df.groupby('set').n_peaks.describe()); print(df.groupby('set').n_peaks.sum())
print(df[(df.n_tsv != 1) | (df.n_h5 != 5) | (df.n_peaks < 0)])
