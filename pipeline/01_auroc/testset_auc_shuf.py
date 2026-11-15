"""Counts AUROC/AUPRC on dinucleotide-shuffled sequences, with and without the control: how much of the score is left
once motifs are destroyed but the dinucleotide (and GC) composition of each region is kept?

Uses the test regions, labels and real-sequence predictions of testset_auc.py (pred_noctl) and testset_auc_mix.py
(pred_ctl): the same fold-k regions with ambiguous nonpeaks dropped, in the same row order as
testset_predictions_mix_{tag}.parquet. Per fold k, each region's 2114 bp sequence is dinucleotide shuffled once
with the same call as predict_paper.py's dinuc condition (tangermeme, random_state=42; the few regions with N, flagged in
has_N, are shuffled with N as a fifth letter) and the fold-k model scores it
  pred_shuf_ctl    shuffled sequence, the region's own control
  pred_shuf_noctl  shuffled sequence, all-zero control
Two comparisons per condition (ctl / noctl):
  shufpk   shuffled peaks vs real nonpeaks   (auroc_shufpk_ctl, auroc_shufpk_noctl, ...)
  shufall  shuffled peaks vs shuffled nonpeaks (auroc_shufall_ctl, auroc_shufall_noctl, ...)
With the control, a control-driven model keeps its AUROC; without it, only sequence composition is left.

Output per experiment, in {BASE}/{dir}/ (model set: env BPNET_SET, see bpnet_sets.py):
  testset_predictions_shuf_{tag}.parquet  chrom, start, end, fold, label, has_N, pred_ctl, pred_noctl, pred_shuf_ctl, pred_shuf_noctl
  testset_auc_shuf_{tag}.csv              per fold: auroc/auprc for real and shuffled sequences, ± control.
Existing outputs are skipped. Usage: testset_auc_shuf.py [DIR ...]   (no args: LSF array slice LSB_JOBINDEX of N_TASKS)."""
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
from tangermeme.ersatz import dinucleotide_shuffle
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


def shuffled(seqs):
    """(N, 4, 2114) one-hot of each sequence dinucleotide shuffled once, as predict_paper.py does it. predict_paper drops
    peaks with N; tangermeme rejects them, so sequences with N are shuffled with N as a fifth letter (then all-zero)."""
    X = torch.stack([one_hot_encode(s, alphabet=['A', 'C', 'G', 'T']) for s in seqs]).to(torch.int8)
    has_n = torch.tensor(['N' in s for s in seqs])
    out = X.clone()
    if (~has_n).any():
        out[~has_n] = dinucleotide_shuffle(X[~has_n].float(), n=1, random_state=42).squeeze(1).to(torch.int8)
    if has_n.any():
        X5 = torch.stack([one_hot_encode(s, alphabet=['A', 'C', 'G', 'T', 'N'], ignore=[]) for s, h in zip(seqs, has_n) if h])
        out[has_n] = dinucleotide_shuffle(X5.float(), n=1, random_state=42).squeeze(1)[:, :4].to(torch.int8)
    return out


@torch.no_grad()
def predict(w, X, C):
    out = []
    for i in range(0, len(X), BATCH):
        c = None if C is None else C[i:i + BATCH].cuda()
        out.append(w(X[i:i + BATCH].cuda().float(), c).flatten().cpu())
    return torch.cat(out).numpy()


def metrics(y, score, name):
    return {f'auroc_{name}': roc_auc_score(y, score), f'auprc_{name}': average_precision_score(y, score)}


def run(row):
    d, tag = f'{BASE}/{row.dir}', row.tag
    dest_pred, dest_auc = f'{d}/testset_predictions_shuf_{tag}.parquet', f'{d}/testset_auc_shuf_{tag}.csv'
    if os.path.exists(dest_auc):
        print('skip (exists)', dest_auc, flush=True)
        return
    t0 = time.time()
    real = pd.read_parquet(f'{d}/testset_predictions_{tag}.parquet')
    mix = pd.read_parquet(f'{d}/testset_predictions_mix_{tag}.parquet')
    key = ['chrom', 'start', 'end', 'label']
    preds, rows = [], []
    for k in range(N_FOLDS):
        fold = mix[mix.fold == k][['chrom', 'start', 'end', 'fold', 'label', 'pred_ctl']].reset_index(drop=True)
        noctl = real[(real.fold == k) & ~real.ambiguous].reset_index(drop=True)
        assert (fold[key] == noctl[key]).all().all(), 'mix and testset rows differ'
        fold['pred_noctl'] = noctl.pred_noctl.values
        seqs = sequences(fold)
        fold['has_N'] = ['N' in q for q in seqs]
        assert all(len(s) == IN_WINDOW for s in seqs)
        X = shuffled(seqs)
        C = control_tracks(row, fold)
        [ckpt] = glob.glob(f'{d}/models__{row.bpnet_models_accession}/fold_{k}/model.fold_{k}.*.h5')
        w = LogCountWrapper(ControlWrapper(BasePairNet.from_bpnet(ckpt).cuda().eval()))
        fold['pred_shuf_ctl'] = predict(w, X, C)
        fold['pred_shuf_noctl'] = predict(w, X, None)
        preds.append(fold)
        y, pk = fold.label.values, fold.label.values == 1
        r = dict(fold=k, n_pos=int(pk.sum()), n_neg=int((~pk).sum()))
        for cond in ['ctl', 'noctl']:
            r.update(metrics(y, fold[f'pred_{cond}'], cond))
            r.update(metrics(y, np.where(pk, fold[f'pred_shuf_{cond}'], fold[f'pred_{cond}']), f'shufpk_{cond}'))
            r.update(metrics(y, fold[f'pred_shuf_{cond}'], f'shufall_{cond}'))
        rows.append(r)
        print(f'  {row.tf_cells} fold_{k} n={len(fold)}', flush=True)

    auc = pd.DataFrame(rows)
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
