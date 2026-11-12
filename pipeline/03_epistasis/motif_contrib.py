"""proper.ipynb as a script: per experiment, the `contrib` table of the shuffle-donor motif knockout (notebook's last cell),
for ALL peaks of the experiment.

Peaks: training_and_test_regions__*/peaks.all_input_regions.*.bed.gz, 2114 bp around the interval centre (as in the
notebook), peaks containing N dropped. Models: the 5 fold .h5, CountWrapper(ControlWrapper(.)) from bpnetlite, i.e.
NO control (all-zero control track). Every prediction is the mean over the 5 folds.

Motifs (env MOTIFS), hits overlapping the central 2*FLANK bp window of each peak; others = minus the primary bases:
  bigbed (default)  ENCODE's annotated instances, sequence_motifs_instances__*.bigBed, rows whose head_type contains
                    'counts', pos_patterns only. primary ("p0") = category_type_simple == 'direct'; others = every other
                    pos hit (indirect, non-canonical, unvalidated). Skipped: experiments without a bigBed (162
                    archived models) and the 47 old-format bigBeds without category_type_simple.
                    Output {BASE}/{dir}/motif_epistasis_direct_{tag}.parquet. wt and shuffled do not depend on the
                    hits, so they are taken from motif_epistasis_{tag}.parquet when it exists (same peaks, checked).
  finemo            Fi-NeMo counts hits, lambda 0.7 (.tsv); primary = names ending in pattern_0 (pos only, see
                    P0_POS_ONLY). Output {BASE}/{dir}/motif_epistasis_{tag}.parquet (run 2026-10-18).
Donor: N_SHUF dinucleotide shuffles of the central window (tangermeme, seed SEED + peak row, so chunking does not
change them). Conditions (each averaged over the N_SHUF shuffles; shuffle k is the donor in every condition):
  shuffled  whole central window shuffled
  none      pattern_0 and other motif bases taken from the shuffle
  p0_only   other motif bases from the shuffle -> pattern_0 intact
  non_only  pattern_0 bases from the shuffle   -> others intact
  additive  = p0_only + non_only - none (per shuffle);  interaction = wt - additive
A peak with nothing to replace in a condition gets wt there (exactly what the notebook's prediction would be).

Output (name: see MOTIFS), one row per peak (row = peak index after the N filter):
  chrom start end  rank (1 = highest wt)  n_p0 n_other (hits)  bp_p0 bp_other (bases replaced)
  wt shuffled none p0_only non_only additive interaction c_p0 c_non_p0 and *_sd (sd over shuffles)
Existing outputs are skipped. Usage: BPNET_SET=paper|all [MOTIFS=bigbed|finemo] motif_contrib.py [DIR ...]
(no args: LSF array slice)."""
import os, sys, glob, time
import bbi
import numpy as np
import pandas as pd
import torch
import pymisha as pm
from bpnetlite.bpnet import BasePairNet, CountWrapper, ControlWrapper
from tangermeme.utils import one_hot_encode
from tangermeme.ersatz import dinucleotide_shuffle
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths
import bpnet_sets

BASE, meta = bpnet_sets.load()
L, FLANK, N_SHUF, SEED = 2114, 300, 20, 0
MID = L // 2
OFF = MID - FLANK
CHUNK, BATCH = 64, 1024          # peaks per shuffle chunk, sequences per forward pass
P0_POS_ONLY = os.environ.get('P0_POS_ONLY', '1') == '1'
MOTIFS = os.environ.get('MOTIFS', 'bigbed')
OUT = {'bigbed': 'motif_epistasis_direct', 'finemo': 'motif_epistasis'}[MOTIFS]
pm.gsetroot(paths.MISHA_HG38)
CONDS = ['shuffled', 'none', 'p0_only', 'non_only']


def load_peaks(d):
    [bed] = glob.glob(f'{d}/training_and_test_regions__*/peaks.all_input_regions.*.bed.gz')
    peaks = pm.gintervals_force_range(pm.gintervals_normalize(pm.gintervals_import_bed(bed), L))
    seqs = pd.Series(pm.gseq_extract(peaks)).str.upper()
    ok = (~seqs.str.contains('N') & (seqs.str.len() == L)).values
    return peaks[ok].reset_index(drop=True)[['chrom', 'start', 'end']], seqs[ok].reset_index(drop=True)


def read_hits(d):
    """All hits of the experiment: chrom start end motif_name hit_strand category is_p0 (see MOTIFS)."""
    if MOTIFS == 'finemo':
        [tsv] = glob.glob(f'{d}/sequence_motifs_instances__*/counts/*lambda_0p7/*.tsv')
        h = (pd.read_csv(tsv, sep='\t', usecols=['chr', 'start', 'end', 'motif_name', 'strand'])
             .rename(columns={'chr': 'chrom', 'strand': 'hit_strand'}))
        h['category'] = h.motif_name.str.split('_').str[0]
        h['is_p0'] = h.motif_name.str.endswith('_pattern_0') & (h.motif_name.str.startswith('pos_') | (not P0_POS_ONLY))
    else:
        [bb] = glob.glob(f'{d}/sequence_motifs_instances__*.bigBed')
        with bbi.open(bb) as f:
            h = pd.concat([f.fetch_intervals(c, 0, -1) for c in f.chromsizes], ignore_index=True)
        if 'category_type_simple' not in h:
            return None                             # old-format bigBed, no direct/indirect annotation
        h = h[h.head_type.str.contains('counts') & h.name_tfmodisco.str.startswith('pos_')]
        h = h.rename(columns={'name_tfmodisco': 'motif_name', 'strand': 'hit_strand', 'category_type_simple': 'category'})
        h['is_p0'] = h.category == 'direct'
    h = h[['chrom', 'start', 'end', 'motif_name', 'hit_strand', 'category', 'is_p0']]
    return h.astype({'start': int, 'end': int}).reset_index(drop=True)


def load_hits(hits, peaks):
    """Hits overlapping each peak's central window, deduplicated, window-relative start/end, is_p0."""
    hits = hits[hits.chrom.isin(set(peaks.chrom))]  # bigBeds can hold contigs the misha genome lacks
    win = peaks.assign(example_idx=np.arange(len(peaks)))
    win = pm.gintervals_normalize(win, 2 * FLANK)[['chrom', 'start', 'end', 'example_idx']]
    ov = pm.gintervals_neighbors(win, hits, maxneighbors=10_000, mindist=0, maxdist=0)
    if ov is None:                                  # no hit in any window: every condition = wt
        return pd.DataFrame({'example_idx': np.zeros(0, int), 'start': np.zeros(0, int), 'end': np.zeros(0, int),
                             'motif_name': pd.Series(dtype=str), 'strand': pd.Series(dtype=str),
                             'category': pd.Series(dtype=str), 'is_p0': np.zeros(0, bool)})
    ov.columns = ['chrom', 'win_start', 'win_end', 'example_idx', 'hit_chrom', 'hit_start', 'hit_end'] + list(ov.columns[7:])
    s = pd.DataFrame({'example_idx': ov.example_idx.astype(int),
                      'start': ov.hit_start - ov.win_start, 'end': ov.hit_end - ov.win_start,
                      'motif_name': ov.motif_name, 'strand': ov.hit_strand, 'category': ov.category,
                      'is_p0': ov.is_p0.astype(bool)}).drop_duplicates()
    return s.reset_index(drop=True)


def masks(hits, n):
    """(n, L) bool masks: pattern_0 bases, and other-motif bases not covered by pattern_0 (= notebook's free segments)."""
    m_p0, m_oth = np.zeros((n, L), bool), np.zeros((n, L), bool)
    for i, a, b, p in zip(hits.example_idx, OFF + hits.start, OFF + hits.end, hits.is_p0):
        (m_p0 if p else m_oth)[i, max(a, 0):min(b, L)] = True
    return torch.from_numpy(m_p0), torch.from_numpy(m_oth & ~m_p0)


@torch.no_grad()
def predict(models, X):
    """Mean over folds of predicted log counts, X (N, 4, L) on CPU -> (N,) numpy."""
    out = []
    for i in range(0, len(X), BATCH):
        x = X[i:i + BATCH].cuda().float()
        out.append(torch.stack([m(x).flatten() for m in models]).mean(0).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def run(row):
    d, tag = f'{BASE}/{row.dir}', row.tag
    dest = f'{d}/{OUT}_{tag}.parquet'
    if os.path.exists(dest):
        print('skip (exists)', dest, flush=True)
        return
    if MOTIFS == 'bigbed' and not glob.glob(f'{d}/sequence_motifs_instances__*.bigBed'):
        print('skip (no bigBed)', row.dir, flush=True)
        return
    all_hits = read_hits(d)
    if all_hits is None:
        print('skip (old-format bigBed, no category_type_simple)', row.dir, flush=True)
        return
    t0 = time.time()
    folds = sorted(glob.glob(f'{d}/models_*/fold_*/*.h5'))
    assert len(folds) == 5, folds
    models = [CountWrapper(ControlWrapper(BasePairNet.from_bpnet(f))).cuda().eval() for f in folds]
    peaks, seqs = load_peaks(d)
    N = len(peaks)
    X = torch.stack([one_hot_encode(s) for s in seqs]).to(torch.int8)       # (N, 4, L)
    prev = f'{d}/motif_epistasis_{tag}.parquet'                              # hit-independent wt / shuffled, if done
    reuse = MOTIFS == 'bigbed' and os.path.exists(prev)
    if reuse:
        prev = pd.read_parquet(prev, columns=['chrom', 'start', 'end', 'wt', 'shuffled', 'shuffled_sd'])
        assert (prev[['chrom', 'start', 'end']].values == peaks[['chrom', 'start', 'end']].values).all(), 'peaks differ'
        wt = prev.wt.to_numpy()
    else:
        wt = predict(models, X)
    conds = [k for k in CONDS if not (reuse and k == 'shuffled')]
    hits = load_hits(all_hits, peaks)
    m_p0, m_oth = masks(hits, N)
    cond_mask = {'shuffled': None, 'none': m_p0 | m_oth, 'p0_only': m_oth, 'non_only': m_p0}
    cats = hits.groupby(['category', 'is_p0']).size()
    print(f'  {row.tf_cells} hits in windows by (category, primary): {cats.to_dict()}', flush=True)

    p = {k: np.repeat(wt[:, None], N_SHUF, 1) for k in conds}               # (N, N_SHUF), wt where nothing replaced
    for c0 in range(0, N, CHUNK):
        sl = slice(c0, min(c0 + CHUNK, N))
        Xs = dinucleotide_shuffle(X[sl], start=OFF, end=MID + FLANK, n=N_SHUF,
                                  random_state=SEED + c0).to(torch.int8)     # (n, N_SHUF, 4, L), seed SEED + peak row
        for k in conds:
            if k == 'shuffled':
                rows, Xk = np.arange(sl.stop - sl.start), Xs
            else:
                rows = np.flatnonzero(cond_mask[k][sl].any(1).numpy())
                if not len(rows):
                    continue
                m = cond_mask[k][sl][rows][:, None, None, :]
                Xk = torch.where(m, Xs[rows], X[sl][rows][:, None])
            p[k][sl.start + rows] = predict(models, Xk.reshape(-1, 4, L)).reshape(len(rows), N_SHUF)
        if c0 % (CHUNK * 50) == 0:
            print(f'  {row.tf_cells} {c0}/{N} peaks {time.time() - t0:.0f}s', flush=True)

    p['additive'] = p['p0_only'] + p['non_only'] - p['none']
    p['interaction'] = wt[:, None] - p['additive']
    res = peaks.assign(rank=pd.Series(wt).rank(ascending=False).values,
                       n_p0=np.bincount(hits.example_idx[hits.is_p0], minlength=N),
                       n_other=np.bincount(hits.example_idx[~hits.is_p0], minlength=N),
                       bp_p0=m_p0.sum(1).numpy(), bp_other=m_oth.sum(1).numpy(), wt=wt)
    for k, v in p.items():
        res[k] = v.mean(1)
    res['c_p0'] = res.p0_only - res.none
    res['c_non_p0'] = res.non_only - res.none
    for k, v in p.items():
        res[f'{k}_sd'] = v.std(1)
    if reuse:
        res['shuffled'], res['shuffled_sd'] = prev.shuffled.values, prev.shuffled_sd.values
    res = res[['chrom', 'start', 'end', 'rank', 'n_p0', 'n_other', 'bp_p0', 'bp_other', 'wt'] + CONDS +
              ['additive', 'interaction', 'c_p0', 'c_non_p0'] + [f'{k}_sd' for k in CONDS + ['additive', 'interaction']]]
    res.to_parquet(dest + '.tmp', index=False)
    os.replace(dest + '.tmp', dest)
    print(f'DONE {row.tf_cells} peaks={N} with_p0={(res.n_p0 > 0).sum()} with_other={(res.n_other > 0).sum()} '
          f'{time.time() - t0:.0f}s', flush=True)
    del models
    torch.cuda.empty_cache()


meta = bpnet_sets.select(meta, sys.argv[1:])
print('node', os.uname().nodename, '| GPU', torch.cuda.get_device_name(0), '| TFs', len(meta),
      '| MOTIFS', MOTIFS, '| P0_POS_ONLY', P0_POS_ONLY, flush=True)
failed = []
for row in meta.itertuples():
    try:
        run(row)
    except Exception as e:
        failed.append(row.tf_cells)
        print('FAILED', row.tf_cells, repr(e), flush=True)
print('FAILED', failed, flush=True)
print('ALLDONE', flush=True)
