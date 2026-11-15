"""One row per BPNet model (PAPER 440 + ALL_BPNET 1,842): metadata, the AUROC/AUPRC/ctl_share summary
({set}/testset_auc_summary.csv) and the pairwise Pearson correlations between all prediction columns of
testset_predictions_motifko_{tag}.parquet (test-set peaks, held-out fold model per peak).
  set                  PAPER | ALL_OTHER
  r_{a}__{b}           Pearson r over the peaks of each test fold, averaged over the 5 folds; a/b = the pred_* column
                       names without 'pred_'
  rp0_{a}__{b}         the same over peaks with >=1 pattern_0 hit only (a fold with < 3 such peaks is left out of
                       the mean; NaN if every fold is)
  encode_{pearsonr,spearmanr}_{peaks,all}   ENCODE observed vs predicted log counts (performance_metrics.counts
                       csv), peaks / peaks_and_nonpeaks, mean over the 5 folds
  n_peaks, n_peaks_p0, n_peaks_other   peaks, and peaks with >=1 pattern_0 / other-pattern hit (the ko columns
                       equal WT on the rest)
  {auroc,auprc}_{cond}_{ctl}_{pk}pk     from testset_auc_ko_{tag}.csv (testset_auc_ko.py), mean over folds:
                       cond WT|p0ko|p0rand|othko|shufpk|shufall|mix, ctl ctl|noctl, pk all|p0|oth = positives are all
                       test peaks / peaks with pattern_0 / peaks with other patterns (negatives always all nonpeaks).
                       The allpk WT / mix_ctl / shufpk / shufall columns are asserted equal to the stored ones.
  auprc_baseline_{pk}pk, n_folds_{pk}pk  positive fraction; folds used (a fold with < 3 positives is skipped)
  n_pos_{pk}pk, n_neg_{pk}pk  peaks / nonpeaks behind those AUCs (summed over the folds used)
  n_nonpeaks           test nonpeaks (ambiguous ones dropped), over all 5 folds
Output: data/bpnet_big_table.parquet (repository). CPU job."""
import glob, warnings
from itertools import combinations
import numpy as np
import pandas as pd
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
import bpnet_sets

warnings.filterwarnings('ignore', 'Mean of empty slice')

META = ['tf_cells', 'tf_name', 'cell_line', 'experiment_accession', 'bpnet_models_accession', 'bpnet_model_status',
        'dir', 'tag']
rows = []
for s, label in [('paper', 'PAPER'), ('all', 'ALL_OTHER')]:
    bpnet_sets.SET = s
    base, meta = bpnet_sets.load()
    summ = pd.read_csv(f'{base}/testset_auc_summary.csv')
    assert summ.tf_cells.is_unique and set(summ.tf_cells) == set(meta.tf_cells), f'{s}: summary vs meta'
    for r in meta.itertuples():
        p = pd.read_parquet(f'{base}/{r.dir}/testset_predictions_motifko_{r.tag}.parquet')
        pc = [c for c in p if c.startswith('pred_')]
        assert len(pc) == 12, (r.dir, pc)
        rec = {'set': label, **{c: getattr(r, c) for c in META}, 'n_peaks': len(p),
               'n_peaks_p0': int((p.n_p0 > 0).sum()), 'n_peaks_other': int((p.n_other > 0).sum())}
        cm, cq = [], []  # per-fold correlation matrices -> mean over folds
        for k, f in p.groupby('fold'):
            q = f[f.n_p0 > 0]
            cm.append(np.corrcoef(f[pc].values.T))
            cq.append(np.corrcoef(q[pc].values.T) if len(q) >= 3 else np.full(cm[-1].shape, np.nan))
        assert len(cm) == 5, r.dir
        cm, cq = np.mean(cm, axis=0), np.nanmean(cq, axis=0) if np.isfinite(cq).any() else np.full(cm[0].shape, np.nan)
        for i, j in combinations(range(len(pc)), 2):
            rec[f'r_{pc[i][5:]}__{pc[j][5:]}'] = cm[i, j]
            rec[f'rp0_{pc[i][5:]}__{pc[j][5:]}'] = cq[i, j]
        [perf] = glob.glob(f'{base}/{r.dir}/model_performance_metrics__*/performance_metrics.counts.*.csv')
        perf = pd.read_csv(perf)
        for m in ['pearsonr', 'spearmanr']:
            for reg, nm in [('peaks', 'peaks'), ('peaks_and_nonpeaks', 'all')]:
                v = perf[(perf.metric_type == m) & (perf.regions_input == reg)]
                assert v.fold.nunique() == 5 == len(v), (r.dir, m, reg)
                rec[f'encode_{m}_{nm}'] = v.metric_value.mean()
        ka = pd.read_csv(f'{base}/{r.dir}/testset_auc_ko_{r.tag}.csv')
        fm = ka.groupby(['peaks', 'ctl', 'cond'])[['auroc', 'auprc']].mean()
        for (pk, ctl, cond), v in fm.iterrows():
            rec[f'auroc_{cond}_{ctl}_{pk}pk'], rec[f'auprc_{cond}_{ctl}_{pk}pk'] = v.auroc, v.auprc
        for pk, v in ka.groupby('peaks'):
            rec[f'auprc_baseline_{pk}pk'], rec[f'n_folds_{pk}pk'] = v.baseline.mean(), v.fold.nunique()
            per_fold = v.drop_duplicates('fold')
            rec[f'n_pos_{pk}pk'], rec[f'n_neg_{pk}pk'] = int(per_fold.n_pos.sum()), int(per_fold.n_neg.sum())
        rec['n_nonpeaks'] = int(ka[ka.peaks == 'all'].drop_duplicates('fold').n_neg.sum())
        rows.append(rec)
    print(s, len(meta), flush=True)

big = pd.DataFrame(rows)
summ = pd.concat([pd.read_csv(f'{bpnet_sets.ROOT}/{d}/testset_auc_summary.csv').assign(set=l)
                  for d, l in [('PAPER', 'PAPER'), ('ALL_BPNET', 'ALL_OTHER')]])
dup = [c for c in summ if c in big and c not in ('tf_cells', 'set')]
assert (big.merge(summ[['set', 'tf_cells'] + dup], on=['set', 'tf_cells'], suffixes=('', '_s'))
        .pipe(lambda m: all((m[c].astype(str) == m[c + '_s'].astype(str)).all() for c in dup))), dup
big = big.merge(summ.drop(columns=dup), on=['set', 'tf_cells'], how='left', validate='1:1')
assert len(big) == 440 + 1842 and big.auroc_ctl.notna().all()
assert (big.n_pos_allpk == big.n_peaks).all()
for new, old in [('auroc_WT_ctl_allpk', 'auroc_ctl'), ('auroc_WT_noctl_allpk', 'auroc_noctl'), ('auroc_mix_ctl_allpk', 'auroc_mix'),
                 ('auprc_WT_ctl_allpk', 'auprc_ctl'), ('auprc_mix_ctl_allpk', 'auprc_mix')] + \
                [(f'{m}_{s}_{c}_allpk', f'{m}_{s}_{c}') for m in ['auroc', 'auprc'] for s in ['shufpk', 'shufall'] for c in ['ctl', 'noctl']]:
    dmax = (big[new] - big[old]).abs().max()
    assert dmax < 1e-6, f'{new} vs stored {old}: {dmax}'
paths.DATA.mkdir(exist_ok=True)
big.to_parquet(paths.DATA / 'bpnet_big_table.parquet', index=False)
print(big.shape, big.set.value_counts().to_dict())
print(list(big.columns))
print('DONE', flush=True)
