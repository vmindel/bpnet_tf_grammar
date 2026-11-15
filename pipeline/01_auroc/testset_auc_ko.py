"""Test-set AUROC / AUPRC for every perturbation, per fold, on three peak sets. CPU only, no new predictions.

Positives: the test peaks of testset_predictions_motifko_{tag}.parquet (held-out fold-k model per peak), restricted to
  peaks=all   every test peak
  peaks=p0    peaks with >=1 pattern_0 hit (n_p0 > 0)
  peaks=oth   peaks with >=1 other positive-pattern hit (n_other > 0)
Negatives: all test nonpeaks of the fold (testset_predictions_shuf_{tag}.parquet, label 0), never filtered, scored in the
form each condition needs:
  cond     peaks                           negatives
  WT       real                            real                 (all peaks: = auroc_ctl / auroc_noctl)
  p0ko     pattern_0 shuffled              real
  p0rand   pattern_0 random bases          real
  othko    other patterns shuffled         real
  shufpk   dinuc shuffled                  real                 (= auroc_shufpk_*)
  shufall  dinuc shuffled                  dinuc shuffled       (= auroc_shufall_*)
  mix      identity mix                    identity mix         (= auroc_mix for ctl)
each with the region's own control (ctl) and with none (noctl). Mixed negatives come from
testset_predictions_mix_{tag}.parquet: pred_mix_ctl, and pred_mix_noctl = pred_noctl[mix_source] (row within the fold).
A fold with < 3 positives in a peak set is skipped.

Output per experiment, in {BASE}/{dir}/ (model set: env BPNET_SET, see bpnet_sets.py):
  testset_auc_ko_{tag}.csv   peaks, fold, ctl, cond, n_pos, n_neg, auroc, auprc, baseline
Existing outputs are skipped. Usage: testset_auc_ko.py [DIR ...]  (no args: LSF array slice LSB_JOBINDEX of N_TASKS)."""
import os, sys, time
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
import bpnet_sets

BASE, meta = bpnet_sets.load()  # env BPNET_SET: paper (default) or all
CTLS = ['ctl', 'noctl']
CONDS = {  # cond -> (peak column, negative column); '{}' = ctl / noctl
    'WT': ('{}', '{}'), 'p0ko': ('p0ko_{}', '{}'), 'p0rand': ('p0rand_{}', '{}'), 'othko': ('othko_{}', '{}'),
    'shufpk': ('shuf_{}', '{}'), 'shufall': ('shuf_{}', 'shuf_{}'), 'mix': ('mix_{}', 'mix_{}'),
}
SUBSETS = {'all': lambda p: p, 'p0': lambda p: p[p.n_p0 > 0], 'oth': lambda p: p[p.n_other > 0]}
MIN_POS = 3


def negatives(d, tag):
    sh = pd.read_parquet(f'{d}/testset_predictions_shuf_{tag}.parquet')
    mx = pd.read_parquet(f'{d}/testset_predictions_mix_{tag}.parquet')
    key = ['chrom', 'start', 'end', 'fold', 'label']
    assert len(sh) == len(mx) and (sh[key].values == mx[key].values).all(), 'shuf and mix rows differ'
    out = []
    for k, f in sh.groupby('fold', sort=True):
        m = mx[mx.fold == k]
        out.append(f.assign(pred_mix_ctl=m.pred_mix_ctl.values, pred_mix_noctl=f.pred_noctl.values[m.mix_source.values]))
    return pd.concat(out).query('label == 0')


def run(row):
    d, tag = f'{BASE}/{row.dir}', row.tag
    dest = f'{d}/testset_auc_ko_{tag}.csv'
    if os.path.exists(dest):
        print('skip (exists)', dest, flush=True)
        return
    t0 = time.time()
    ko, neg = pd.read_parquet(f'{d}/testset_predictions_motifko_{tag}.parquet'), negatives(d, tag)
    rows = []
    for k in range(5):
        ng = neg[neg.fold == k]
        for peaks, sub in SUBSETS.items():
            p = sub(ko[ko.fold == k])
            if len(p) < MIN_POS:
                continue
            y = np.r_[np.ones(len(p)), np.zeros(len(ng))]
            for ctl in CTLS:
                for cond, (pc, nc) in CONDS.items():
                    s = np.r_[p[f'pred_{pc.format(ctl)}'].values, ng[f'pred_{nc.format(ctl)}'].values]
                    rows.append(dict(peaks=peaks, fold=k, ctl=ctl, cond=cond, n_pos=len(p), n_neg=len(ng),
                                     auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s), baseline=y.mean()))
    res = pd.DataFrame(rows)
    res.to_csv(dest + '.tmp', index=False)
    os.replace(dest + '.tmp', dest)
    print(f'DONE {row.tf_cells} {time.time() - t0:.0f}s', flush=True)


meta = bpnet_sets.select(meta, sys.argv[1:])
print('node', os.uname().nodename, '| TFs', len(meta), flush=True)
failed = []
for row in meta.itertuples():
    try:
        run(row)
    except Exception as e:
        failed.append(row.tf_cells)
        print('FAILED', row.tf_cells, repr(e), flush=True)
print('FAILED', failed, flush=True)
print('ALLDONE', flush=True)
