"""ENCODE-style counts AUROC/AUPRC (peaks vs GC-matched nonpeaks) with NO control passed to the model,
to compare with ENCODE's reported (with-control) values.

Per fold k, the fold-k model is run on that fold's test regions from the "training and test regions" tar
(peaks.testset.fold_k: 2114 bp centred on the summit; nonpeaks.testset.fold_k: already 2114 bp) with an all-zero
control; the score is the predicted log counts. As ENCODE does, nonpeaks overlapping the experiment's
pooled-pseudoreplicate "IDR ranked peaks" are dropped (overlap with the full 2114 bp window reproduces ENCODE's
auprc_baseline exactly on A549/ATF3; baseline_absdiff reports the check for every fold).

Output per experiment, in {BASE}/{dir}/ (PAPER: PAPER/{cell}/{TF}/, tag {cell}_{TF}; model set: env BPNET_SET, see bpnet_sets.py):
  testset_predictions_{cell}_{TF}.parquet  per-region: chrom, start, end, fold, label, ambiguous, pred_noctl
  testset_auc_{cell}_{TF}.csv              per fold: n_pos, n_neg, auprc_baseline, auroc_noctl, auprc_noctl,
                                           ENCODE's reported encode_auroc / encode_auprc / encode_auprc_baseline,
                                           baseline_absdiff = |auprc_baseline - encode_auprc_baseline|.
Existing outputs are skipped. Usage: testset_auc.py [DIR ...]   (no args: LSF array slice LSB_JOBINDEX of N_TASKS)."""
import os, sys, glob, time
import numpy as np
import pandas as pd
import torch
import pymisha as pm
from sklearn.metrics import roc_auc_score, average_precision_score
from bpnetlite.bpnet import BasePairNet
from cherimoya import ControlWrapper, LogCountWrapper
from tangermeme.utils import one_hot_encode
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
import bpnet_sets

BASE, meta = bpnet_sets.load()  # env BPNET_SET: paper (default) or all
IN_WINDOW, N_FOLDS, BATCH = 2114, 5, 256
pm.gsetroot(paths.MISHA_HG38)


def read_bed(path, is_peak):
    b = pd.read_csv(path, sep='\t', header=None, usecols=[0, 1, 9], names=['chrom', 'start', 'summit'])
    centre = b.start + b.summit if is_peak else b.start + IN_WINDOW // 2
    return pd.DataFrame({'chrom': b.chrom, 'start': centre - IN_WINDOW // 2, 'end': centre + IN_WINDOW // 2,
                         'label': int(is_peak)})


def misha_order(df):
    """Rows of df reordered to match pm.gintervals (which sorts), plus the sequences in that order."""
    iv = pm.gintervals(df.chrom.tolist(), df.start.tolist(), df.end.tolist())
    key = ['chrom', 'start', 'end']
    iv = iv[key].assign(_k=iv.groupby(key).cumcount())
    out = iv.merge(df.assign(_k=df.groupby(key).cumcount()), on=key + ['_k'], how='left', validate='1:1')
    assert len(out) == len(df) and out.label.notna().all()
    seqs = [s.upper() for s in pm.gseq_extract(iv[key])]
    return out.drop(columns='_k'), seqs


def overlaps(regions, peaks):
    """True where a region overlaps any peak interval (half-open)."""
    hit = np.zeros(len(regions), bool)
    for c, idx in regions.groupby('chrom').groups.items():
        p = peaks[peaks.chrom == c].sort_values('start')
        if p.empty:
            continue
        ps, pe_max = p.start.values, np.maximum.accumulate(p.end.values)
        r = regions.loc[idx]
        k = np.searchsorted(ps, r.end.values, side='left')  # peaks[:k] start before region end
        hit[regions.index.get_indexer(idx)] = (k > 0) & (pe_max[np.maximum(k - 1, 0)] > r.start.values)
    return hit


@torch.no_grad()
def predict_noctl(w, X):
    return torch.cat([w(X[i:i + BATCH].cuda().float(), None).flatten().cpu() for i in range(0, len(X), BATCH)]).numpy()


def encode_reported(d):
    [f] = glob.glob(f'{d}/model_performance_metrics__*/performance_metrics.counts.*.csv')
    m = pd.read_csv(f)
    m = m[m.metric_type.isin(['auroc', 'auprc', 'auprc_baseline'])]
    return m.pivot_table(index='fold', columns='metric_type', values='metric_value').add_prefix('encode_')


def run(row):
    d, tag = f'{BASE}/{row.dir}', row.tag
    dest_pred, dest_auc = f'{d}/testset_predictions_{tag}.parquet', f'{d}/testset_auc_{tag}.csv'
    if os.path.exists(dest_auc):
        print('skip (exists)', dest_auc, flush=True)
        return
    t0 = time.time()
    [rdir] = glob.glob(f'{d}/training_and_test_regions__{row.training_regions_accession}')
    enc = encode_reported(d)
    folds = [pd.concat([read_bed(glob.glob(f'{rdir}/fold_{k}/{kind}.testset.fold_{k}.*.bed*')[0], kind == 'peaks')
                        for kind in ['peaks', 'nonpeaks']], ignore_index=True).assign(fold=k) for k in range(N_FOLDS)]
    # ranked-peak file: if the experiment has several, take the one that best reproduces ENCODE's auprc_baseline
    best = None
    for acc in row.idr_ranked_peaks_accession.split(';'):
        ranked = pd.read_csv(f'{d}/chipseq_idr_ranked_peaks__{acc}.bed.gz', sep='\t', header=None,
                             usecols=[0, 1, 2], names=['chrom', 'start', 'end'])
        amb = [(f.label == 0).values & overlaps(f, ranked) for f in folds]
        err = sum(abs(f.label.sum() / (~a).sum() - enc.encode_auprc_baseline[k]) for k, (f, a) in enumerate(zip(folds, amb)))
        if best is None or err < best[0]:
            best = (err, acc, amb)
    _, ranked_acc, amb = best
    preds, rows = [], []
    for k, fold in enumerate(folds):
        fold['ambiguous'] = amb[k]
        fold, seqs = misha_order(fold)  # pm.gintervals sorts: keep labels aligned with the sequences
        assert all(len(s) == IN_WINDOW for s in seqs)
        X = torch.stack([one_hot_encode(s, alphabet=['A', 'C', 'G', 'T'], ignore=['N']) for s in seqs])
        [ckpt] = glob.glob(f'{d}/models__{row.bpnet_models_accession}/fold_{k}/model.fold_{k}.*.h5')
        w = LogCountWrapper(ControlWrapper(BasePairNet.from_bpnet(ckpt).cuda().eval()))
        fold['pred_noctl'] = predict_noctl(w, X)
        preds.append(fold)
        sub = fold[~fold.ambiguous]
        rows.append(dict(fold=k, n_pos=int(sub.label.sum()), n_neg=int((sub.label == 0).sum()),
                         auprc_baseline=sub.label.mean(), auroc_noctl=roc_auc_score(sub.label, sub.pred_noctl),
                         auprc_noctl=average_precision_score(sub.label, sub.pred_noctl)))
        print(f'  {row.tf_cells} fold_{k} n={len(fold)}', flush=True)

    auc = pd.DataFrame(rows).join(enc, on='fold')
    auc['baseline_absdiff'] = (auc.auprc_baseline - auc.encode_auprc_baseline).abs()
    auc.insert(0, 'tf_cells', row.tf_cells)
    auc.insert(1, 'model_status', row.bpnet_model_status)
    auc.insert(2, 'idr_ranked_peaks', ranked_acc)
    pd.concat(preds, ignore_index=True).to_parquet(dest_pred + '.tmp', index=False)
    os.replace(dest_pred + '.tmp', dest_pred)
    auc.to_csv(dest_auc + '.tmp', index=False)
    os.replace(dest_auc + '.tmp', dest_auc)
    print(f'DONE {row.tf_cells} {time.time() - t0:.0f}s', flush=True)
    print(auc.round(3).to_string(), flush=True)


meta = bpnet_sets.select(meta, sys.argv[1:])
print('node', os.uname().nodename, '| GPU', torch.cuda.get_device_name(0), '| TFs', len(meta), flush=True)
failed = []
for row in meta.itertuples():
    try:
        run(row)
    except Exception as e:
        failed.append(row.tf_cells)
        print('FAILED', row.tf_cells, repr(e), flush=True)
print('FAILED', failed, flush=True)
print('ALLDONE', flush=True)
