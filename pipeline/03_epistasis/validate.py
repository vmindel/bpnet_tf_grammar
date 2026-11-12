"""Open every motif_epistasis parquet: rows vs peak count, NaNs, the derived columns."""
import os, sys, numpy as np, pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
HERE = _Path(__file__).resolve().parent
pc = pd.read_csv(HERE / 'tasks' / 'peak_counts.csv').set_index(['set', 'dir']).n_peaks
OUT = os.environ.get('OUT', 'motif_epistasis')
rows = []
for s in ['paper', 'all']:
    os.environ['BPNET_SET'] = s
    import importlib, bpnet_sets; importlib.reload(bpnet_sets)
    base, meta = bpnet_sets.load()
    for r in meta.itertuples():
        f = f'{base}/{r.dir}/{OUT}_{r.tag}.parquet'
        if not os.path.exists(f):
            continue
        x = pd.read_parquet(f)
        ref = pd.read_parquet(f'{base}/{r.dir}/motif_epistasis_{r.tag}.parquet', columns=['chrom', 'start', 'end', 'wt', 'shuffled'])
        same = len(ref) == len(x) and (ref[['chrom', 'start', 'end']].values == x[['chrom', 'start', 'end']].values).all() \
            and np.allclose(ref.wt, x.wt) and np.allclose(ref.shuffled, x.shuffled)
        rows.append(dict(same_as_ref=same, set=s, dir=r.dir, n=len(x), n_bed=pc[(s, r.dir)], nan=int(x.isna().sum().sum()),
                         add_err=float(np.abs(x.additive - (x.p0_only + x.non_only - x.none)).max()),
                         frac_p0=(x.n_p0 > 0).mean(), med_int=x.interaction.median(), med_c_p0=x.c_p0.median()))
v = pd.DataFrame(rows)
v['dropped'] = v.n_bed - v.n
print(v.groupby('set')[['n', 'dropped', 'nan', 'add_err', 'frac_p0', 'med_int', 'med_c_p0']].describe().T.round(4).to_string())
print('no direct hits:', (v.frac_p0 == 0).sum(), 'of', len(v))
print('bad:', v[(v.nan > 0) | (v.dropped < 0) | (v.dropped > 0.01 * v.n_bed) | (v.add_err > 1e-4) | ~v.same_as_ref].to_string())
v.to_csv(HERE / f'validate_{OUT}.csv', index=False)
