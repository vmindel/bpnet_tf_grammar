"""Counts predictions on the test-set PEAKS with Fi-NeMo motif hits knocked out, ± control: what is left of each peak's
prediction without its pattern_0 sites, or without its other motifs?

Regions and models are those of the AUROC runs (testset_auc.py / testset_auc_mix.py): each fold's test peaks
(peaks.testset.fold_k, 2114 bp centred on the summit), each scored once by its held-out fold-k model, so every peak of
the experiment gets exactly one value per condition (fold = which model). Nonpeaks are not mutated.

Motifs: ENCODE's Fi-NeMo hits, counts head, lambda 0.7 (Fi-NeMo's default), sequence_motifs_instances__{acc}/counts/
seq_motifs_instances.counts.lambda_0p7/*.bed. Hits are assigned to a peak by overlap with its 2114 bp window (clipped to
it; identical windows counted once). pattern_0 = pos_patterns ..._pattern_0; others = every other pos_patterns hit;
neg_patterns are never touched. Knockouts shuffle the bases inside the hit windows (mononucleotide permutation, so the
peak's composition is kept; retried so each window changes when it can):
  p0ko   every pattern_0 window shuffled, even where it overlaps other patterns
  othko  every other positive pattern's window shuffled, with the pattern_0 positions frozen (only the remaining bases of
         each window are permuted), so pattern_0 sites are byte-identical to the wild type
  p0rand every pattern_0 window replaced by uniform random bases (i.i.d. A/C/G/T, redrawn until the window differs);
         unlike p0ko this also changes the window's composition, so a GC-rich motif cannot survive as its own shuffle.
         Its own RNG stream (seed + 1), so adding it did not change the p0ko / othko sequences.
A peak without such hits keeps its wild-type sequence (n_p0 / n_other = 0). The shuffle's RNG is seeded per peak
(chrom, start), so the sequences can be regenerated exactly; they are not stored.

Columns (log counts; ctl = the region's own control from the observed control bigWigs, noctl = all-zero control):
  pred_ctl, pred_noctl       wild type: the stored AUROC predictions (recomputed here and checked, max diff logged)
  pred_mix_ctl               sequence of mix_source (derangement over the fold's peaks AND nonpeaks, seed 0), from
                             testset_predictions_mix_{tag}.parquet
  pred_mix_noctl             = pred_noctl[mix_source]: with a zero control the output depends on the sequence only
  pred_p0ko_ctl/_noctl, pred_othko_ctl/_noctl, pred_p0rand_ctl/_noctl   the knockouts
plus n_p0 / n_other (hit windows in the peak) and bp_p0ko / bp_othko / bp_p0rand (bases that differ from the wild type).

Output per experiment, in {BASE}/{dir}/ (model set: env BPNET_SET, see bpnet_sets.py):
  testset_predictions_motifko_{tag}.parquet
Existing outputs are skipped; an existing file that lacks a knockout (e.g. p0rand, added 2026-09-30) gets only
its missing columns added, after checking that its rows are the same peaks. Usage: testset_motif_ko.py [DIR ...]   (no args: LSF array slice LSB_JOBINDEX of N_TASKS)."""
import os, sys, glob, time, re, zlib
import numpy as np
import pandas as pd
import torch
import bbi
import pymisha as pm
from bpnetlite.bpnet import BasePairNet
from cherimoya import ControlWrapper, LogCountWrapper
from tangermeme.utils import one_hot_encode
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
import bpnet_sets

BASE, meta = bpnet_sets.load()  # env BPNET_SET: paper (default) or all
IN_WINDOW, N_FOLDS, BATCH, SEED = 2114, 5, 256, 42
HITS = 'sequence_motifs_instances__{}/counts/seq_motifs_instances.counts.lambda_0p7/*.bed'
PATTERN = re.compile(r'^(pos|neg)_patterns\..*_pattern_(\d+)$')
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


def read_hits(d, acc):
    """Fi-NeMo hits: chrom, start, end, kind (p0 / other); neg_patterns dropped."""
    [bed] = glob.glob(f'{d}/' + HITS.format(acc))
    h = pd.read_csv(bed, sep='\t', header=None, usecols=[0, 1, 2, 3], names=['chrom', 'start', 'end', 'motif'])
    m = h.motif.str.extract(PATTERN)
    assert m[0].notna().all(), f'unparsed motif names in {bed}: {h.motif[m[0].isna()].unique()[:3]}'
    h = h[m[0] == 'pos'].assign(kind=np.where(m.loc[m[0] == 'pos', 1] == '0', 'p0', 'other'))
    return h.drop(columns='motif').sort_values(['chrom', 'start']).reset_index(drop=True)


def windows_in(peaks, hits):
    """Per peak: (p0 windows, other windows) as sorted unique (start, end) relative to the peak, clipped to it."""
    out = [([], []) for _ in range(len(peaks))]
    wmax = int((hits.end - hits.start).max()) if len(hits) else 0
    for c, g in hits.groupby('chrom'):
        hs, he, hk = g.start.values, g.end.values, g.kind.values
        for i in np.flatnonzero(peaks.chrom.values == c):
            s = peaks.start.values[i]
            lo, hi = np.searchsorted(hs, s - wmax), np.searchsorted(hs, s + IN_WINDOW)
            sets = ({}, {})
            for j in range(lo, hi):
                a, b = max(hs[j], s) - s, min(he[j], s + IN_WINDOW) - s
                if b > a:
                    sets[hk[j] != 'p0'][(a, b)] = None
            out[i] = (sorted(sets[0]), sorted(sets[1]))
    return out


def permute(seq, pos, rng, max_tries=100):
    """seq (list) with the bases at positions pos permuted among themselves; retried so they change when they can."""
    orig = [seq[p] for p in pos]
    for _ in range(max_tries):
        new = rng.permutation(orig)
        if list(new) != orig:
            break
    for p, x in zip(pos, new):
        seq[p] = x


def knockouts(seq, p0, other, rng):
    """(p0ko, othko) sequences, see the module docstring."""
    ko = list(seq)
    for a, b in p0:
        permute(ko, list(range(a, b)), rng)
    frozen = np.zeros(len(seq), bool)
    for a, b in p0:
        frozen[a:b] = True
    oth = list(seq)
    for a, b in other:
        pos = [p for p in range(a, b) if not frozen[p]]
        if pos:
            permute(oth, pos, rng)
    oth = ''.join(oth)
    assert all(oth[a:b] == seq[a:b] for a, b in p0), 'othko touched a pattern_0 site'
    return ''.join(ko), oth


def randomized(seq, p0, rng, max_tries=100):
    """seq with every pattern_0 window replaced by uniform random bases (each window redrawn until it differs)."""
    out = list(seq)
    for a, b in p0:
        for _ in range(max_tries):
            new = rng.choice(list('ACGT'), b - a)
            if ''.join(new) != seq[a:b]:
                break
        out[a:b] = new
    return ''.join(out)


KOS = ['p0ko', 'othko', 'p0rand']


@torch.no_grad()
def predict(w, seqs, C):
    out = []
    for i in range(0, len(seqs), BATCH):
        X = torch.stack([one_hot_encode(s, alphabet=['A', 'C', 'G', 'T'], ignore=['N']) for s in seqs[i:i + BATCH]])
        c = None if C is None else C[i:i + BATCH].cuda()
        out.append(w(X.cuda().float(), c).flatten().cpu())
    return torch.cat(out).numpy()


def run(row):
    d, tag = f'{BASE}/{row.dir}', row.tag
    dest = f'{d}/testset_predictions_motifko_{tag}.parquet'
    old = pd.read_parquet(dest) if os.path.exists(dest) else None
    todo = [n for n in KOS if old is None or f'pred_{n}_ctl' not in old]
    if not todo:
        print('skip (exists)', dest, flush=True)
        return
    t0 = time.time()
    real = pd.read_parquet(f'{d}/testset_predictions_{tag}.parquet')
    mix = pd.read_parquet(f'{d}/testset_predictions_mix_{tag}.parquet')
    hits = read_hits(d, row.motif_instances_accession)
    key = ['chrom', 'start', 'end', 'label']
    out, diffs = [], []
    for k in range(N_FOLDS):
        fold = mix[mix.fold == k].reset_index(drop=True)
        noctl = real[(real.fold == k) & ~real.ambiguous].reset_index(drop=True)
        assert (fold[key] == noctl[key]).all().all(), 'mix and testset rows differ'
        fold['pred_noctl'] = noctl.pred_noctl.values
        fold['pred_mix_noctl'] = fold.pred_noctl.values[fold.mix_source.values]
        pk = fold[fold.label == 1].reset_index(drop=True)
        seqs = sequences(pk)
        assert all(len(s) == IN_WINDOW for s in seqs)
        wins = windows_in(pk, hits)
        p0ko, othko, p0rand = [], [], []
        for s, (c, st), (p0, oth) in zip(seqs, zip(pk.chrom, pk.start), wins):
            rng = np.random.default_rng([SEED, zlib.crc32(c.encode()), int(st)])
            a, b = knockouts(s, p0, oth, rng)
            p0ko.append(a)
            othko.append(b)
            p0rand.append(randomized(s, p0, np.random.default_rng([SEED + 1, zlib.crc32(c.encode()), int(st)])))
        pk['n_p0'] = [len(w[0]) for w in wins]
        pk['n_other'] = [len(w[1]) for w in wins]
        pk['bp_p0ko'] = [sum(x != y for x, y in zip(s, t)) for s, t in zip(seqs, p0ko)]
        pk['bp_othko'] = [sum(x != y for x, y in zip(s, t)) for s, t in zip(seqs, othko)]
        pk['bp_p0rand'] = [sum(x != y for x, y in zip(s, t)) for s, t in zip(seqs, p0rand)]
        C = control_tracks(row, pk)
        [ckpt] = glob.glob(f'{d}/models__{row.bpnet_models_accession}/fold_{k}/model.fold_{k}.*.h5')
        w = LogCountWrapper(ControlWrapper(BasePairNet.from_bpnet(ckpt).cuda().eval()))
        diffs.append((np.abs(predict(w, seqs, C) - pk.pred_ctl.values).max(),
                      np.abs(predict(w, seqs, None) - pk.pred_noctl.values).max()))
        for name, ss in [(n, {'p0ko': p0ko, 'othko': othko, 'p0rand': p0rand}[n]) for n in todo]:
            pk[f'pred_{name}_ctl'] = predict(w, ss, C)
            pk[f'pred_{name}_noctl'] = predict(w, ss, None)
        out.append(pk)
        del C, w
        torch.cuda.empty_cache()
        print(f'  {row.tf_cells} fold_{k} peaks={len(pk)} with_p0={(pk.n_p0 > 0).sum()} with_other={(pk.n_other > 0).sum()}'
              f' | wt check max|diff| ctl {diffs[-1][0]:.2g} noctl {diffs[-1][1]:.2g}', flush=True)

    res = pd.concat(out, ignore_index=True)
    dmax = np.max(diffs)
    assert dmax < 1e-2, f'recomputed wild type differs from the stored AUROC predictions by {dmax}'
    new = [c for n in todo for c in (f'bp_{n}', f'pred_{n}_ctl', f'pred_{n}_noctl')]
    if old is not None:  # add the missing knockouts to the existing file: same peaks, same (re-generated) knockouts
        same = ['chrom', 'start', 'end', 'fold', 'n_p0', 'n_other', 'bp_p0ko', 'bp_othko']
        assert len(old) == len(res) and (old[same].values == res[same].values).all(), 'existing file rows differ'
        res = old.assign(**{c: res[c].values for c in new})
    else:
        res = res[['chrom', 'start', 'end', 'fold', 'n_p0', 'n_other'] + [f'bp_{n}' for n in KOS] +
                  ['pred_ctl', 'pred_noctl', 'pred_mix_ctl', 'pred_mix_noctl'] +
                  [f'pred_{n}_{c}' for n in KOS for c in ['ctl', 'noctl']]]
    res.to_parquet(dest + '.tmp', index=False)
    os.replace(dest + '.tmp', dest)
    print(f'DONE {row.tf_cells} peaks={len(res)} added={todo} wt_maxdiff={dmax:.2g} {time.time() - t0:.0f}s', flush=True)
    print(res.drop(columns=['chrom', 'start', 'end']).describe().loc[['mean', '50%']].round(3).to_string(), flush=True)


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
