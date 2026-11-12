"""Check every direct_sites output against itself and against motif_epistasis_direct_{tag}.parquet (same donors).
Per experiment: rows, NaNs, sites rows == 2 * sum(n_direct), n_direct == n_p0, max |diff| of WT / KOALL_WT /
KOALL_DiSh vs wt / non_only / shuffled, max |diff| of bp_masked = 0 sites vs their reference.
-> validate.csv. Run on a CPU node: bsub -q short ... validate.py"""
import os, sys, glob
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
    tasks = open(HERE / 'tasks' / f'tasks_{s}_direct.txt').read().split()
    for d in tasks:
        tag = d.replace('/', '_')
        fp, fs = f'{base}/{d}/direct_sites_peaks_{tag}.parquet', f'{base}/{d}/direct_sites_sites_{tag}.parquet'
        r = dict(set=s, dir=d, has_peaks=os.path.exists(fp), has_sites=os.path.exists(fs))
        if r['has_peaks'] and r['has_sites']:
            pk, st = pd.read_parquet(fp), pd.read_parquet(fs)
            r.update(n_peaks=len(pk), n_site_rows=len(st), n_sites=int(pk.n_direct.sum()),
                     peaks_with_sites=int((pk.n_direct > 0).sum()),
                     nan_peaks=int(pk.isna().sum().sum()), nan_sites=int(st.isna().sum().sum()),
                     site_rows_ok=len(st) == 2 * pk.n_direct.sum())
            z = st[st.bp_masked == 0]
            r['edge_sites'] = len(z) // 2
            r['edge_maxdiff'] = float((z.prediction - z.prediction_ref).abs().max()) if len(z) else 0.0
            ep = f'{base}/{d}/motif_epistasis_direct_{tag}.parquet'
            if os.path.exists(ep):
                e = pd.read_parquet(ep, columns=['chrom', 'start', 'end', 'n_p0', 'wt', 'non_only', 'shuffled'])
                same = len(e) == len(pk) and (e[['chrom', 'start', 'end']].values ==
                                              pk[['peak_chrom', 'peak_start', 'peak_end']].values).all()
                r['coords_match_epi'] = bool(same)
                if same:
                    r['n_direct_eq_n_p0'] = bool((pk.n_direct.values == e.n_p0.values).all())
                    for mine, theirs in [('WT', 'wt'), ('KOALL_WT', 'non_only'), ('KOALL_DiSh', 'shuffled')]:
                        r[f'maxdiff_{mine}'] = float(np.abs(pk[mine].values - e[theirs].values).max())
        rows.append(r)

v = pd.DataFrame(rows)
v.to_csv(HERE / 'validate.csv', index=False)
done = v[v.has_peaks & v.has_sites]
print(f'{len(v)} experiments in task lists, {len(done)} with both outputs')
print('NaNs:', int(done.nan_peaks.sum()), int(done.nan_sites.sum()), '| site rows wrong:', int((~done.site_rows_ok).sum()))
print('coords mismatch vs epistasis:', int((done.coords_match_epi == False).sum()),
      '| no epistasis file:', int(done.coords_match_epi.isna().sum()),
      '| n_direct != n_p0:', int((done.n_direct_eq_n_p0 == False).sum()))
for k in ['maxdiff_WT', 'maxdiff_KOALL_WT', 'maxdiff_KOALL_DiSh', 'edge_maxdiff']:
    print(f'{k}: median {done[k].median():.2e}, max {done[k].max():.2e}, n > 1e-2: {int((done[k] > 1e-2).sum())}')
print('experiments with no direct site:', int((done.n_sites == 0).sum()))
print('total peaks', int(done.n_peaks.sum()), 'sites', int(done.n_sites.sum()))
