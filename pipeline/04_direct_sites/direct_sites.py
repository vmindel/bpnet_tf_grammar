"""PaperAnalysis/direct_sites.ipynb as a script: per experiment, knock out each direct motif hit one at a time and all
together, in the native background (WT) and in a dinucleotide-shuffled background with the direct bases kept (DiSh).

Peaks, hits, donors exactly as motif_contrib.py with MOTIFS=bigbed: 2114 bp peaks without N; ENCODE bigBed hits,
counts head, pos_ patterns, category_type_simple == 'direct', overlapping the central 600 bp (NOT merged, exact
duplicates dropped); donor = N_SHUF dinucleotide shuffles of the window, seed SEED + chunk start, CHUNK peaks per chunk.
Prediction = predicted log counts, mean over the 5 folds, no control.

  WT          native                         KO_i WT     site i bases from the shuffle, rest native
  DiSh        window shuffled, direct native KO_i DiSh   DiSh with site i also from the shuffle (keep direct & ~site_i)
  KOALL_WT    all direct bases from shuffle  KOALL_DiSh  fully shuffled window
KO_i removes every base of site i, also bases shared with an overlapping site. Sites touching the window edge only
(bp_masked = 0) have KO_i = reference.

Outputs in {BASE}/{dir}/:
  direct_sites_sites_{tag}.parquet  one row per (site, condition WT|DiSh): peak_name peak_chrom peak_start peak_end
      site_idx site_name site_start site_end site_rel_start site_rel_end strand bp_masked overlaps condition
      prediction prediction_sd prediction_ref   (ref = WT for WT rows, DiSh for DiSh rows; mean / sd over shuffles)
  direct_sites_peaks_{tag}.parquet  one row per peak: peak_name peak_chrom peak_start peak_end n_direct
      WT DiSh KOALL_WT KOALL_DiSh and *_sd
Checks against motif_epistasis_direct_{tag}.parquet when present (same donors): n_direct == n_p0,
KOALL_WT ~ non_only, KOALL_DiSh ~ shuffled.
MERGE=1: sites of a peak sharing >= 1 base merged into their union (run_merged) -> direct_merged_sites_{sites,peaks}_{tag}
  .parquet, same columns; sites: overlaps -> n_merged, site_name/strand joined by ','; peaks: n_direct = merged count,
  n_direct_raw = old count. Needs the direct_sites outputs; only merged sites are predicted, the rest copied.
Existing outputs are skipped. Usage: [MERGE=1] BPNET_SET=paper|all direct_sites.py [DIR ...]  (no args: LSF array slice)."""
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
L, FLANK, N_SHUF, SEED, CHUNK, BATCH = 2114, 300, 20, 0, 64, 1024
MID = L // 2
OFF = MID - FLANK
WIN_END = MID + FLANK
pm.gsetroot(paths.MISHA_HG38)


def load_peaks(d):
    [bed] = glob.glob(f'{d}/training_and_test_regions__*/peaks.all_input_regions.*.bed.gz')
    peaks = pm.gintervals_force_range(pm.gintervals_normalize(pm.gintervals_import_bed(bed), L))
    seqs = pd.Series(pm.gseq_extract(peaks)).str.upper()
    ok = (~seqs.str.contains('N') & (seqs.str.len() == L)).values
    return peaks[ok].reset_index(drop=True)[['chrom', 'start', 'end']], seqs[ok].reset_index(drop=True)


def read_direct_hits(d):
    [bb] = glob.glob(f'{d}/sequence_motifs_instances__*.bigBed')
    with bbi.open(bb) as f:
        h = pd.concat([f.fetch_intervals(c, 0, -1) for c in f.chromsizes], ignore_index=True)
    if 'category_type_simple' not in h:
        return None
    h = h[h.head_type.str.contains('counts') & h.name_tfmodisco.str.startswith('pos_')
          & (h.category_type_simple == 'direct')]
    h = h.rename(columns={'name_tfmodisco': 'motif_name', 'strand': 'hit_strand'})
    return h[['chrom', 'start', 'end', 'motif_name', 'hit_strand']].astype({'start': int, 'end': int}).reset_index(drop=True)


SITE_COLS = ['peak_idx', 'site_idx', 'site_start', 'site_end', 'site_name', 'strand', 'a', 'b',
             'site_rel_start', 'site_rel_end', 'bp_masked', 'overlaps']


def cross(hits, peaks):
    hits = hits[hits.chrom.isin(set(peaks.chrom))]
    win = peaks[['chrom', 'start', 'end']].assign(peak_idx=np.arange(len(peaks)))
    win = pm.gintervals_normalize(win, 2 * FLANK)[['chrom', 'start', 'end', 'peak_idx']]
    ov = None if not len(hits) else pm.gintervals_neighbors(win, hits, maxneighbors=10_000, mindist=0, maxdist=0)
    if ov is None:
        return pd.DataFrame({c: pd.Series(dtype=int) for c in SITE_COLS})
    ov.columns = ['chrom', 'win_start', 'win_end', 'peak_idx', 'hit_chrom', 'hit_start', 'hit_end'] + list(ov.columns[7:])
    s = pd.DataFrame({'peak_idx': ov.peak_idx.astype(int),
                      'site_start': ov.hit_start.astype(int), 'site_end': ov.hit_end.astype(int),
                      'site_name': ov.motif_name.values, 'strand': ov.hit_strand.values})
    s = s.drop_duplicates().sort_values(['peak_idx', 'site_start', 'site_end']).reset_index(drop=True)
    s.insert(1, 'site_idx', s.groupby('peak_idx').cumcount())
    pk = peaks.start.values[s.peak_idx]
    s['a'] = np.clip(s.site_start - pk, OFF, WIN_END)
    s['b'] = np.clip(s.site_end - pk, OFF, WIN_END)
    s['site_rel_start'] = s.site_start - (pk + MID)
    s['site_rel_end'] = s.site_end - (pk + MID)
    s['bp_masked'] = s.b - s.a
    prev_end = s.groupby('peak_idx').site_end.cummax().groupby(s.peak_idx).shift()
    ov_prev = (s.site_start < prev_end).to_numpy()
    ov_next = np.r_[ov_prev[1:] & (s.peak_idx.values[1:] == s.peak_idx.values[:-1]), False]
    s['overlaps'] = ov_prev | ov_next
    return s


@torch.no_grad()
def predict(models, X):
    """Mean over folds of predicted log counts, X (N, 4, L) int8 on CPU -> (N,) numpy."""
    out = []
    for i in range(0, len(X), BATCH):
        x = X[i:i + BATCH].cuda().float()
        out.append(torch.stack([m(x).flatten() for m in models]).mean(0).cpu())
    return torch.cat(out).numpy() if out else np.zeros(0)


def run(row):
    d, tag = f'{BASE}/{row.dir}', row.tag
    dest_s, dest_p = f'{d}/direct_sites_sites_{tag}.parquet', f'{d}/direct_sites_peaks_{tag}.parquet'
    if os.path.exists(dest_s) and os.path.exists(dest_p):
        print('skip (exists)', dest_p, flush=True)
        return
    if not glob.glob(f'{d}/sequence_motifs_instances__*.bigBed'):
        print('skip (no bigBed)', row.dir, flush=True)
        return
    hits = read_direct_hits(d)
    if hits is None:
        print('skip (old-format bigBed, no category_type_simple)', row.dir, flush=True)
        return
    t0 = time.time()
    folds = sorted(glob.glob(f'{d}/models_*/fold_*/*.h5'))
    assert len(folds) == 5, folds
    models = [CountWrapper(ControlWrapper(BasePairNet.from_bpnet(f))).cuda().eval() for f in folds]
    peaks, seqs = load_peaks(d)
    N = len(peaks)
    X = torch.stack([one_hot_encode(s) for s in seqs]).to(torch.int8)
    sites = cross(hits, peaks)
    n_direct = np.bincount(sites.peak_idx, minlength=N)
    print(f'  {row.tf_cells}: {N} peaks, {len(sites)} direct sites, {(n_direct > 0).sum()} peaks with sites', flush=True)

    wt = predict(models, X)
    pk_mean = {k: np.repeat(wt[:, None], N_SHUF, 1).astype(np.float32) for k in ['DiSh', 'KOALL_WT', 'KOALL_DiSh']}
    ko = {k: np.zeros((len(sites), N_SHUF), np.float32) for k in ['WT', 'DiSh']}
    a_all, b_all, pidx_all = sites.a.to_numpy(), sites.b.to_numpy(), sites.peak_idx.to_numpy()
    for c0 in range(0, N, CHUNK):
        c1 = min(c0 + CHUNK, N)
        n = c1 - c0
        Xc = X[c0:c1]
        Xs = dinucleotide_shuffle(Xc, start=OFF, end=WIN_END, n=N_SHUF, random_state=SEED + c0).to(torch.int8)
        jj = np.flatnonzero((pidx_all >= c0) & (pidx_all < c1))   # global site rows of this chunk
        loc = pidx_all[jj] - c0
        m_i = torch.zeros(len(jj), L, dtype=torch.bool)
        m_all = torch.zeros(n, L, dtype=torch.bool)
        for r, (i, a, b) in enumerate(zip(loc, a_all[jj], b_all[jj])):
            m_i[r, a:b] = True
            m_all[i, a:b] = True
        has = np.flatnonzero(m_all.any(1).numpy())                # peaks with something to replace
        ma = m_all[has][:, None, None]
        koall_wt = torch.where(ma, Xs[has], Xc[has][:, None])
        dish = torch.where(ma, Xc[has][:, None], Xs[has])
        keep = (m_all[loc] & ~m_i)[:, None, None]
        ko_wt = torch.where(m_i[:, None, None], Xs[loc], Xc[loc][:, None])
        ko_dish = torch.where(keep, Xc[loc][:, None], Xs[loc])
        parts = [Xs, koall_wt, dish, ko_wt, ko_dish]
        p = predict(models, torch.cat([q.reshape(-1, 4, L) for q in parts]))
        p = np.split(p, np.cumsum([q.shape[0] * N_SHUF for q in parts])[:-1])
        p = [q.reshape(-1, N_SHUF) for q in p]
        pk_mean['KOALL_DiSh'][c0:c1] = p[0]
        pk_mean['DiSh'][c0:c1] = p[0]                             # no direct bases kept -> DiSh = shuffled
        pk_mean['KOALL_WT'][c0 + has] = p[1]
        pk_mean['DiSh'][c0 + has] = p[2]
        ko['WT'][jj], ko['DiSh'][jj] = p[3], p[4]
        if c0 % (CHUNK * 25) == 0:
            print(f'  {row.tf_cells} {c0}/{N} peaks {time.time() - t0:.0f}s', flush=True)

    # ---------- peaks table ----------
    pk = peaks.rename(columns={'chrom': 'peak_chrom', 'start': 'peak_start', 'end': 'peak_end'})
    pk.insert(0, 'peak_name', tag + '_' + pk.index.astype(str))
    pk['n_direct'] = n_direct
    pk['WT'] = wt
    for k in ['DiSh', 'KOALL_WT', 'KOALL_DiSh']:
        pk[k] = pk_mean[k].mean(1)
    for k in ['DiSh', 'KOALL_WT', 'KOALL_DiSh']:
        pk[k + '_sd'] = pk_mean[k].std(1)

    # ---------- sites table ----------
    ref = {'WT': wt, 'DiSh': pk['DiSh'].to_numpy()}
    out = []
    for cond in ['WT', 'DiSh']:
        t = sites.assign(condition=cond, prediction=ko[cond].mean(1), prediction_sd=ko[cond].std(1),
                         prediction_ref=ref[cond][sites.peak_idx])
        out.append(t)
    st = pd.concat(out, ignore_index=True)
    st = (pk[['peak_name', 'peak_chrom', 'peak_start', 'peak_end']].iloc[st.peak_idx].reset_index(drop=True)
          .join(st[['site_idx', 'site_name', 'site_start', 'site_end', 'site_rel_start', 'site_rel_end', 'strand',
                    'bp_masked', 'overlaps', 'condition', 'prediction', 'prediction_sd', 'prediction_ref']]))

    # ---------- checks vs the epistasis run (same donors) ----------
    prev = f'{d}/motif_epistasis_direct_{tag}.parquet'
    if os.path.exists(prev):
        prev = pd.read_parquet(prev)
        assert (prev[['chrom', 'start', 'end']].values == peaks[['chrom', 'start', 'end']].values).all(), 'peaks differ'
        print(f'  check n_direct == n_p0: {(n_direct == prev.n_p0.values).mean():.4f}', flush=True)
        for mine, theirs in [('WT', 'wt'), ('KOALL_WT', 'non_only'), ('KOALL_DiSh', 'shuffled')]:
            dd = np.abs(pk[mine].values - prev[theirs].values)
            print(f'  check {mine} vs {theirs}: max |diff| {dd.max():.2e}, median {np.median(dd):.2e}', flush=True)
    z = st[st.bp_masked == 0]                 # KO_i == reference sequence: differences are only GPU batch noise
    if len(z):
        dz = np.abs(z.prediction - z.prediction_ref).max()
        print(f'  check bp_masked=0 sites ({len(z) // 2}) vs ref: max |diff| {dz:.2e}', flush=True)
        assert dz < 1e-2, 'bp_masked = 0 sites far from ref: masking bug'

    for df, dest in [(st, dest_s), (pk, dest_p)]:
        df.to_parquet(dest + '.tmp', index=False)
        os.replace(dest + '.tmp', dest)
    print(f'DONE {row.tf_cells} peaks={N} sites={len(sites)} {time.time() - t0:.0f}s', flush=True)


# ===================== MERGE=1: overlapping direct sites merged into one site =====================
def merge_sites(s):
    """Merge sites of the same peak that share >= 1 base (strict overlap, same rule as `overlaps`; touching sites stay
    separate) into their union. site_name / strand = unique values joined by ','; raw_rows = rows of s merged."""
    s = s.reset_index(drop=True)
    prev_end = s.groupby('peak_idx').site_end.cummax().groupby(s.peak_idx).shift()
    gid = (prev_end.isna() | (s.site_start >= prev_end)).cumsum()
    uj = lambda x: ','.join(dict.fromkeys(x))
    m = s.assign(raw_row=s.index).groupby(gid).agg(
        peak_idx=('peak_idx', 'first'), site_start=('site_start', 'min'), site_end=('site_end', 'max'),
        site_name=('site_name', uj), strand=('strand', uj), n_merged=('raw_row', 'size'),
        raw_rows=('raw_row', list)).reset_index(drop=True)
    m.insert(1, 'site_idx', m.groupby('peak_idx').cumcount())
    return m


def run_merged(row):
    """Same conditions as run(), with merged sites. Single (unmerged) sites and all peak-level values are identical
    computations to the direct_sites run (same donors), so they are copied from it; only merged sites are predicted."""
    d, tag = f'{BASE}/{row.dir}', row.tag
    dest_s, dest_p = f'{d}/direct_merged_sites_sites_{tag}.parquet', f'{d}/direct_merged_sites_peaks_{tag}.parquet'
    old_s, old_p = f'{d}/direct_sites_sites_{tag}.parquet', f'{d}/direct_sites_peaks_{tag}.parquet'
    if os.path.exists(dest_s) and os.path.exists(dest_p):
        print('skip (exists)', dest_p, flush=True)
        return
    if not (os.path.exists(old_s) and os.path.exists(old_p)):
        print('skip (no direct_sites outputs)', row.dir, flush=True)
        return
    t0 = time.time()
    peaks, seqs = load_peaks(d)
    N = len(peaks)
    pko, sto = pd.read_parquet(old_p), pd.read_parquet(old_s)
    assert (pko[['peak_chrom', 'peak_start', 'peak_end']].values == peaks[['chrom', 'start', 'end']].values).all()
    raw = cross(read_direct_hits(d), peaks)
    old = {c: sto[sto.condition == c].reset_index(drop=True) for c in ['WT', 'DiSh']}
    for c in old:                                              # old site rows are in raw order
        assert len(old[c]) == len(raw) and (old[c][['site_start', 'site_end']].values ==
                                            raw[['site_start', 'site_end']].values).all(), 'raw sites differ'
    if not len(raw):
        pk = pko.assign(n_direct_raw=0)
        st = sto.drop(columns='overlaps').assign(n_merged=pd.Series(dtype=int))
        for df, dest in [(st, dest_s), (pk, dest_p)]:
            df.to_parquet(dest + '.tmp', index=False)
            os.replace(dest + '.tmp', dest)
        print(f'DONE {row.tf_cells} peaks={N} no direct sites', flush=True)
        return
    sites = merge_sites(raw)
    pk = peaks.start.values[sites.peak_idx]
    sites['a'] = np.clip(sites.site_start - pk, OFF, WIN_END)
    sites['b'] = np.clip(sites.site_end - pk, OFF, WIN_END)
    sites['site_rel_start'] = sites.site_start - (pk + MID)
    sites['site_rel_end'] = sites.site_end - (pk + MID)
    sites['bp_masked'] = sites.b - sites.a
    multi = np.flatnonzero(sites.n_merged.values > 1)
    print(f'  {row.tf_cells}: {N} peaks, {len(raw)} raw sites -> {len(sites)} merged sites, '
          f'{len(multi)} of them from >1 raw site', flush=True)

    # copy singles, predict merged ones
    first = np.array([r[0] for r in sites.raw_rows])
    ko = {c: np.full((len(sites), 2), np.nan, np.float32) for c in ['WT', 'DiSh']}   # mean, sd
    for c in ko:
        ko[c][:] = old[c][['prediction', 'prediction_sd']].values[first]
    if len(multi):
        folds = sorted(glob.glob(f'{d}/models_*/fold_*/*.h5'))
        assert len(folds) == 5, folds
        models = [CountWrapper(ControlWrapper(BasePairNet.from_bpnet(f))).cuda().eval() for f in folds]
        X = torch.stack([one_hot_encode(s) for s in seqs]).to(torch.int8)
        pidx_m, pidx_raw = sites.peak_idx.values, raw.peak_idx.values
        for c0 in sorted(set((pidx_m[multi] // CHUNK) * CHUNK)):   # only chunks with merged sites; seed = SEED + c0
            c1 = min(c0 + CHUNK, N)
            Xc = X[c0:c1]
            Xs = dinucleotide_shuffle(Xc, start=OFF, end=WIN_END, n=N_SHUF, random_state=SEED + c0).to(torch.int8)
            m_all = torch.zeros(c1 - c0, L, dtype=torch.bool)
            for i, a, b in zip(pidx_raw, raw.a.values, raw.b.values):
                if c0 <= i < c1:
                    m_all[i - c0, a:b] = True
            jj = multi[(pidx_m[multi] >= c0) & (pidx_m[multi] < c1)]
            loc = pidx_m[jj] - c0
            m_i = torch.zeros(len(jj), L, dtype=torch.bool)
            for r, (a, b) in enumerate(zip(sites.a.values[jj], sites.b.values[jj])):
                m_i[r, a:b] = True
            keep = (m_all[loc] & ~m_i)[:, None, None]
            ko_wt = torch.where(m_i[:, None, None], Xs[loc], Xc[loc][:, None])
            ko_dish = torch.where(keep, Xc[loc][:, None], Xs[loc])
            p = predict(models, torch.cat([ko_wt.reshape(-1, 4, L), ko_dish.reshape(-1, 4, L)])).reshape(2, -1, N_SHUF)
            for c, q in zip(['WT', 'DiSh'], p):
                ko[c][jj, 0], ko[c][jj, 1] = q.mean(1), q.std(1)

    # tables
    pk = pko.copy()
    pk['n_direct_raw'] = pk.n_direct
    pk['n_direct'] = np.bincount(sites.peak_idx, minlength=N)
    ref = {'WT': pk.WT.to_numpy(), 'DiSh': pk.DiSh.to_numpy()}
    st = pd.concat([sites.assign(condition=c, prediction=ko[c][:, 0], prediction_sd=ko[c][:, 1],
                                 prediction_ref=ref[c][sites.peak_idx]) for c in ['WT', 'DiSh']], ignore_index=True)
    st = (pk[['peak_name', 'peak_chrom', 'peak_start', 'peak_end']].iloc[st.peak_idx].reset_index(drop=True)
          .join(st[['site_idx', 'site_name', 'site_start', 'site_end', 'site_rel_start', 'site_rel_end', 'strand',
                    'bp_masked', 'n_merged', 'condition', 'prediction', 'prediction_sd', 'prediction_ref']]))

    # checks: a merged site covering all direct sites of its peak has KO_i == KOALL (same sequences)
    one = st[st.peak_name.isin(pk.peak_name[pk.n_direct == 1]) & (st.n_merged > 1)].merge(
        pk[['peak_name', 'KOALL_WT', 'KOALL_DiSh']], on='peak_name')
    if len(one):
        dd = max(np.abs(np.where(one.condition == 'WT', one.KOALL_WT, one.KOALL_DiSh) - one.prediction).max(), 0)
        print(f'  check merged sites = whole peak ({len(one) // 2}) vs KOALL: max |diff| {dd:.2e}', flush=True)
        assert dd < 1e-2, 'merged KO != KOALL: masking bug'
    assert not st.prediction.isna().any()

    for df, dest in [(st, dest_s), (pk, dest_p)]:
        df.to_parquet(dest + '.tmp', index=False)
        os.replace(dest + '.tmp', dest)
    print(f'DONE {row.tf_cells} peaks={N} merged sites={len(sites)} predicted={len(multi)} {time.time() - t0:.0f}s',
          flush=True)


if __name__ == '__main__':
    fn = run_merged if os.environ.get('MERGE') == '1' else run
    for row in bpnet_sets.select(meta, sys.argv[1:]).itertuples():
        fn(row)
