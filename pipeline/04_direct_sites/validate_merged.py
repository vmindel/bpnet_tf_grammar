"""Check every direct_merged_sites output against itself and against the original direct_sites outputs.
Per experiment: NaNs, site rows == 2 * sum(n_direct), n_direct_raw == original n_direct, sum(n_merged) per peak ==
n_direct_raw, peak table values identical to the original, unmerged sites identical to the original, merged sites
of a peak don't overlap each other. -> validate_merged.csv. Run on a CPU node: bsub -q short ... validate_merged.py"""
import os
import numpy as np
import pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
ROOT = str(paths.ROOT)
HERE = _Path(__file__).resolve().parent

rows = []
for s, base in [('paper', f'{ROOT}/PAPER'), ('all', f'{ROOT}/ALL_BPNET')]:
    for d in open(HERE / 'tasks' / f'tasks_{s}_direct.txt').read().split():
        tag = d.replace('/', '_')
        f = {k: f'{base}/{d}/{k}_{tag}.parquet' for k in
             ['direct_sites_peaks', 'direct_sites_sites', 'direct_merged_sites_peaks', 'direct_merged_sites_sites']}
        r = dict(set=s, dir=d, has_merged=all(os.path.exists(f[k]) for k in f))
        if r['has_merged']:
            po, so = pd.read_parquet(f['direct_sites_peaks']), pd.read_parquet(f['direct_sites_sites'])
            pm_, sm = pd.read_parquet(f['direct_merged_sites_peaks']), pd.read_parquet(f['direct_merged_sites_sites'])
            r.update(n_peaks=len(pm_), n_raw=int(po.n_direct.sum()), n_merged_sites=int(pm_.n_direct.sum()),
                     n_multi=int((sm.n_merged > 1).sum() // 2),
                     nan_peaks=int(pm_.isna().sum().sum()), nan_sites=int(sm.isna().sum().sum()),
                     site_rows_ok=len(sm) == 2 * pm_.n_direct.sum(),
                     raw_ok=bool((pm_.n_direct_raw.values == po.n_direct.values).all()))
            cols = [c for c in po.columns if c != 'n_direct']
            r['peaks_same'] = bool(po[cols].equals(pm_[cols]))
            w = sm[sm.condition == 'WT']
            nm = w.groupby('peak_name').n_merged.sum().reindex(pm_.peak_name, fill_value=0).values
            r['sum_n_merged_ok'] = bool((nm == pm_.n_direct_raw.values).all())
            ws = w.sort_values(['peak_name', 'site_start'])
            same_pk = ws.peak_name.values[1:] == ws.peak_name.values[:-1]
            r['merged_overlap'] = int((same_pk & (ws.site_start.values[1:] < ws.site_end.values[:-1])).sum())
            key = ['peak_name', 'condition', 'site_start', 'site_end', 'site_name', 'strand']
            single = sm[sm.n_merged == 1].merge(so[~so.overlaps.astype(bool)], on=key, suffixes=('', '_o'))
            r['singles_matched'] = len(single) == (sm.n_merged == 1).sum() == (~so.overlaps.astype(bool)).sum()
            r['singles_maxdiff'] = float((single.prediction - single.prediction_o).abs().max()) if len(single) else 0.0
        rows.append(r)

v = pd.DataFrame(rows)
v.to_csv(HERE / 'validate_merged.csv', index=False)
x = v[v.has_merged]
print(f'{len(v)} experiments in task lists, {len(x)} with merged outputs')
for k in ['site_rows_ok', 'raw_ok', 'peaks_same', 'sum_n_merged_ok', 'singles_matched']:
    print(f'{k}: {int(x[k].astype(bool).sum())} / {len(x)}')
print('NaNs:', int(x.nan_peaks.sum()), int(x.nan_sites.sum()), '| merged sites overlapping:', int(x.merged_overlap.sum()))
print(f'singles max |diff|: {x.singles_maxdiff.max():.2e}')
print(f'raw sites {x.n_raw.sum()} -> merged {x.n_merged_sites.sum()} ({x.n_multi.sum()} from >1 raw site)')
print('share of raw sites in a merged site:', f'{(x.n_raw.sum() - x.n_merged_sites.sum() + x.n_multi.sum()) / x.n_raw.sum():.3f}')
