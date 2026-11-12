"""Counts AUROC/AUPRC WITH the control but with the sequences mixed between regions: does the control alone
separate peaks from background?

Uses the test regions and labels of testset_auc.py (testset_predictions_{cell}_{TF}.parquet; ambiguous nonpeaks
dropped). Per fold k, the fold-k model scores the fold's regions with each region's own control
(observed control profile bigWigs the model was trained with) and
  pred_ctl      its own sequence       -> should reproduce ENCODE's reported auroc/auprc (check: ctl_absdiff)
  pred_mix_ctl  the sequence of region mix_source[i], a derangement (seed 0) over ALL scored regions of the fold,
                peaks and nonpeaks together, so sequence carries no label information.
If the control dominates, auroc_mix stays close to auroc_ctl; if the sequence does, it falls toward 0.5.

Output per experiment, in {BASE}/{dir}/ (PAPER: PAPER/{cell}/{TF}/, tag {cell}_{TF}; model set: env BPNET_SET, see bpnet_sets.py):
  testset_predictions_mix_{cell}_{TF}.parquet  chrom, start, end, fold, label, pred_ctl, pred_mix_ctl, mix_source
                                              (mix_source = row, within the same fold, whose sequence was used)
  testset_auc_mix_{cell}_{TF}.csv              per fold: auroc/auprc for ctl and mix, ENCODE's reported values.
Existing outputs are skipped. Usage: testset_auc_mix.py [DIR ...]   (no args: LSF array slice LSB_JOBINDEX of N_TASKS)."""
import os, sys, glob, time
import numpy as np
import pandas as pd
import torch
import bbi
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


def sequences(df):
    """Sequences in the row order of df, which must already be in pm.gintervals (sorted) order."""
    key = ['chrom', 'start', 'end']
    iv = pm.gintervals(df.chrom.tolist(), df.start.tolist(), df.end.tolist())
    assert (iv[key].reset_index(drop=True) == df[key].reset_index(drop=True)).all().all(), 'row order != misha order'
    return [s.upper() for s in pm.gseq_extract(iv[key])]


def control_tracks(row, df):
    """(N, 2, 2114) control in model channel order: plus, minus."""
    ctl = BASE + f'/{row.cell_line}/Control/observed_control_profile_{{}}__{{}}.bigWig'
    tr = [bbi.stackup(ctl.format(s, a), df.chrom, df.start, df.end, bins=-1, missing=0, oob=0)
          for s, a in [('plus', row.control_plus_bigwig_accession), ('minus', row.control_minus_bigwig_accession)]]
    return torch.from_numpy(np.nan_to_num(np.stack(tr, axis=1)).astype(np.float32))


def derangement(n):
    g = torch.Generator().manual_seed(0)
    while True:
        perm = torch.randperm(n, generator=g)
        if n < 2 or (perm != torch.arange(n)).all():
            return perm


@torch.no_grad()
def predict(w, X, C, perm):
    ctl, mix = [], []
    for i in range(0, len(X), BATCH):
        sl = slice(i, i + BATCH)
        c = C[sl].cuda()
        ctl.append(w(X[sl].cuda().float(), c).flatten().cpu())
        mix.append(w(X[perm[sl]].cuda().float(), c).flatten().cpu())
    return torch.cat(ctl).numpy(), torch.cat(mix).numpy()


def encode_reported(d):
    [f] = glob.glob(f'{d}/model_performance_metrics__*/performance_metrics.counts.*.csv')
    m = pd.read_csv(f)
    m = m[m.metric_type.isin(['auroc', 'auprc'])]
    return m.pivot_table(index='fold', columns='metric_type', values='metric_value').add_prefix('encode_')


def run(row):
    d, tag = f'{BASE}/{row.dir}', row.tag
    dest_pred, dest_auc = f'{d}/testset_predictions_mix_{tag}.parquet', f'{d}/testset_auc_mix_{tag}.csv'
    if os.path.exists(dest_auc):
        print('skip (exists)', dest_auc, flush=True)
        return
    t0 = time.time()
    regions = pd.read_parquet(f'{d}/testset_predictions_{tag}.parquet')
    enc = encode_reported(d)
    preds, rows = [], []
    for k in range(N_FOLDS):
        fold = regions[(regions.fold == k) & ~regions.ambiguous][['chrom', 'start', 'end', 'fold', 'label']]
        fold = fold.reset_index(drop=True)
        seqs = sequences(fold)
        assert all(len(s) == IN_WINDOW for s in seqs)
        X = torch.stack([one_hot_encode(s, alphabet=['A', 'C', 'G', 'T'], ignore=['N']) for s in seqs])
        C = control_tracks(row, fold)
        perm = derangement(len(fold))
        [ckpt] = glob.glob(f'{d}/models__{row.bpnet_models_accession}/fold_{k}/model.fold_{k}.*.h5')
        w = LogCountWrapper(ControlWrapper(BasePairNet.from_bpnet(ckpt).cuda().eval()))
        fold['pred_ctl'], fold['pred_mix_ctl'] = predict(w, X, C, perm)
        fold['mix_source'] = perm.numpy()
        preds.append(fold)
        y = fold.label
        rows.append(dict(fold=k, n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
                         mix_pos_from_pos=float(y.values[perm.numpy()][y.values == 1].mean()),
                         auroc_ctl=roc_auc_score(y, fold.pred_ctl), auprc_ctl=average_precision_score(y, fold.pred_ctl),
                         auroc_mix=roc_auc_score(y, fold.pred_mix_ctl),
                         auprc_mix=average_precision_score(y, fold.pred_mix_ctl)))
        print(f'  {row.tf_cells} fold_{k} n={len(fold)}', flush=True)

    auc = pd.DataFrame(rows).join(enc, on='fold')
    auc['ctl_absdiff'] = (auc.auroc_ctl - auc.encode_auroc).abs()
    auc.insert(0, 'tf_cells', row.tf_cells)
    auc.insert(1, 'model_status', row.bpnet_model_status)
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
