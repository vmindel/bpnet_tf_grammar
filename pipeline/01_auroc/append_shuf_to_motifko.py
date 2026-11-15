"""Append the dinucleotide-shuffle predictions of testset_auc_shuf.py (pred_shuf_ctl, pred_shuf_noctl) to
testset_predictions_motifko_{tag}.parquet. No new predictions: the peaks (label == 1) of
testset_predictions_shuf_{tag}.parquet are in the same order as the motifko rows (both follow the mix parquet); rows
are asserted to match on chrom, start, end, fold, and pred_ctl / pred_noctl to agree within 1e-2.
Model set: env BPNET_SET (see bpnet_sets.py). Runs every experiment in one CPU job; skips files that already have them."""
import os, sys
import numpy as np
import pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
import bpnet_sets

BASE, meta = bpnet_sets.load()
meta = meta[meta.dir.isin(sys.argv[1:])] if sys.argv[1:] else meta
KEY, NEW = ['chrom', 'start', 'end', 'fold'], ['pred_shuf_ctl', 'pred_shuf_noctl']
n = {'added': 0, 'skipped': 0}
for row in meta.itertuples():
    d, tag = f'{BASE}/{row.dir}', row.tag
    dest = f'{d}/testset_predictions_motifko_{tag}.parquet'
    ko = pd.read_parquet(dest)
    if all(c in ko for c in NEW):
        n['skipped'] += 1
        continue
    shuf = pd.read_parquet(f'{d}/testset_predictions_shuf_{tag}.parquet')
    shuf = shuf[shuf.label == 1].reset_index(drop=True)
    assert len(shuf) == len(ko) and (shuf[KEY].values == ko[KEY].values).all(), f'{row.dir}: rows differ'
    for c in ['pred_ctl', 'pred_noctl']:
        dmax = np.abs(shuf[c].values - ko[c].values).max()
        assert dmax < 1e-2, f'{row.dir}: {c} differs by {dmax}'
    ko.assign(**{c: shuf[c].values for c in NEW}).to_parquet(dest + '.tmp', index=False)
    os.replace(dest + '.tmp', dest)
    n['added'] += 1
print(bpnet_sets.SET, n, 'of', len(meta))
print('DONE', flush=True)
