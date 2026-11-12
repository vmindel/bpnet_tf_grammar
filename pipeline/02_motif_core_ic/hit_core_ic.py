"""Information content of the 20-bp TF-MoDISco core of every Fi-NeMo hit, for one ENCODE BPNet model.

Core: the central 20 bp of the untrimmed TF-MoDISco matrix, [(W-20)/2, (W+20)/2), and the same offset from a hit's
start_untrimmed, on either strand (the flanks are symmetric). Most models: W = 50 = 15 + 20 + 15 (modisco --trim_size 20,
--initial_flank_to_add 5, --final_flank_to_add 10) -> [15:35], exactly modisco's trimmed window. ~58 models, mostly
ZNFs: W = 110 (--trim_size 30, flanks 20 + 20) -> [45:65], the centre 20 of modisco's 30-bp window, so all cores are
20 bp and IC sums compare across models; a longer ZNF motif can extend past it. Column motif_width says which.

Per hit:
  core_bits        sum_i p_i(s_i) * IC_i over the core (one-hot of the motif-oriented site x ICM)
  core_frac        core_bits / consensus core_bits      (1 = consensus site)
  core_frac_total  core_bits / total core IC            (ceiling < 1 for degenerate motifs)
Per motif: core_ic (bits in the core), total_ic (bits in all 50 bp).

Per motif (summary): n_peaks (all peaks Fi-NeMo scanned, with or without hits), n_peaks_with_motif (>= 1 hit,
however many), frac_peaks_with_motif = n_peaks_with_motif / n_peaks, hits_per_motif_peak, median core_frac, core_ic,
total_ic, plus TF_CellLine / tf / cell_line / experiment / bpnet_set so all models can be concatenated.

Hits come from the Fi-NeMo .tsv (full list, with peak_id); peak_id is the row of
training_and_test_regions__*/peaks.all_input_regions.*.bed.gz, whose Fi-NeMo region is summit +- 500 bp.

Usage (models from PAPER/bpnet_sets.py, env BPNET_SET=paper|all):
  hit_core_ic.py [DIR ...]        dirs as in the metadata ({cell}/{TF} or {cell}/{TF}_{ENCSR}); none -> LSF array slice
  env HEAD=counts|profile, LAMBDA=0p7
Writes MODEL_DIR/hit_core_ic_{head}_{lambda}.parquet (per hit) and hit_core_ic_summary_{head}_{lambda}.csv (per motif).
"""
import glob, os, sys
import numpy as np
import pandas as pd
import moodspy as mp
import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[2] / 'modules'))
import paths

GENOME = paths.HG38_FA
CORE_W = 20
IDX = np.full(256, 4, dtype=np.int8)                 # anything but ACGT -> index 4 (dropped)
for i, c in enumerate('ACGT'):
    IDX[ord(c)] = IDX[ord(c.lower())] = i


def one(pattern):
    [f] = glob.glob(pattern)
    return f


def motif_key(name):
    """Hits write the biosample with spaces, .meme with hyphens; prefix-less .meme ids are pos_patterns."""
    name = name.replace(' ', '-')
    return name if '.' in name else 'pos_patterns.' + name


def one_hot(seqs):
    """(n, 4, L) float from equal-length strings; N / non-ACGT columns are all zero."""
    a = np.frombuffer(''.join(seqs).encode(), dtype=np.uint8).reshape(len(seqs), -1)
    x = np.zeros((len(seqs), 5, a.shape[1]))
    np.put_along_axis(x, IDX[a][:, None, :].astype(np.int64), 1, axis=1)
    return x[:, :4]


def run(d, head='counts', lam='0p7', label=None):
    """label: dict of columns added to the summary (TF_CellLine, tf, cell_line, ...)."""
    # motifs: untrimmed PFM -> ICM (4, 50); core = columns 15..34
    pfm = one(f'{d}/sequence_motifs__*/{head}/seq_motifs.{head}.fold_mean.*.meme/*.PFM.meme')
    motifs = mp.read_motifs(path=pfm)
    [W] = {m.length for m in motifs}                 # one modisco window per model: 50 (most) or 110 (ZNFs)
    FLANK = (W - CORE_W) // 2
    CORE = slice(FLANK, FLANK + CORE_W)
    icm, mtab = {}, []
    for m in motifs:
        key = motif_key(m.id)                        # MEME 'MOTIF id' -> m.id (m.name is the empty altname)
        icm[key] = mp.icm(m)
        ic = mp.motif_ic(m)
        mtab.append({'motif_name': key, 'motif_width': W, 'core_ic': ic[CORE].sum(), 'total_ic': ic.sum(),
                     'core_consensus': ''.join('ACGT'[j] for j in icm[key][:, CORE].argmax(0))})
    mtab = pd.DataFrame(mtab)

    # peaks in Fi-NeMo's order (pandas, NOT pm.gintervals_import_bed: that sorts and breaks peak_id)
    pk = one(f'{d}/training_and_test_regions__*/peaks.all_input_regions.*.bed.gz')
    peaks = pd.read_csv(pk, sep='\t', header=None, usecols=[0, 1, 9], names=['chrom', 'peak_start', 'summit'])
    peaks['peak_id'] = np.arange(len(peaks))
    peaks['summit_pos'] = peaks.peak_start + peaks.summit
    peaks['region_start'] = peaks.summit_pos - 500
    peaks['region_end'] = peaks.summit_pos + 500

    # hits: full Fi-NeMo list (one row per hit per peak window), positive patterns only
    tsv = one(f'{d}/sequence_motifs_instances__*/{head}/seq_motifs_instances.{head}.lambda_{lam}/*.tsv')
    hits = pd.read_csv(tsv, sep='\t')
    hits['motif_name'] = hits.motif_name.map(motif_key)
    hits = hits[hits.motif_name.str.startswith('pos_patterns.')].merge(peaks, on='peak_id', how='left')
    assert (hits.chr == hits.chrom).all(), 'peak_id does not match the peak file'
    assert ((hits.start_untrimmed >= hits.region_start) & (hits.end_untrimmed <= hits.region_end)).all()
    assert (hits.end_untrimmed - hits.start_untrimmed == W).all()
    assert hits.motif_name.isin(icm.keys()).all(), set(hits.motif_name) - set(icm)

    # core sequence in motif orientation
    g = mp.FastaFile(GENOME)
    hits['core_start'] = hits.start_untrimmed + FLANK
    hits['core_end'] = hits.core_start + CORE_W
    seqs = [g.fetch(c, s, e).upper() for c, s, e in zip(hits.chr, hits.core_start, hits.core_end)]
    hits['core_seq'] = [mp.revcomp(s) if st == '-' else s for s, st in zip(seqs, hits.strand)]

    # score
    hits['core_bits'] = np.nan
    for name, idx in hits.groupby('motif_name').groups.items():
        M = icm[name][:, CORE]                                          # (4, 20)
        X = one_hot(hits.loc[idx, 'core_seq'].tolist())                 # (n, 4, 20)
        hits.loc[idx, 'core_bits'] = (X * M).sum(axis=(1, 2))
    hits = hits.merge(mtab, on='motif_name', how='left')
    cons = {k: v[:, CORE].max(0).sum() for k, v in icm.items()}
    hits['core_frac'] = hits.core_bits / hits.motif_name.map(cons)
    hits['core_frac_total'] = hits.core_bits / hits.core_ic

    # sanity: the consensus core scores exactly 1
    for k in icm:
        c = mtab.set_index('motif_name').core_consensus[k]
        assert np.isclose((one_hot([c]) * icm[k][:, CORE]).sum() / cons[k], 1)

    cols = ['chr', 'start', 'end', 'start_untrimmed', 'end_untrimmed', 'core_start', 'core_end', 'strand',
            'motif_name', 'hit_coefficient', 'hit_similarity', 'hit_importance', 'peak_id', 'summit_pos',
            'core_seq', 'core_bits', 'core_frac', 'core_frac_total', 'core_ic', 'total_ic', 'core_consensus',
            'motif_width']
    hits[cols].to_parquet(f'{d}/hit_core_ic_{head}_{lam}.parquet', index=False)

    # per motif: fraction of peaks with >= 1 hit (presence, not multiplicity)
    per_peak = hits.groupby(['motif_name', 'peak_id']).size()          # hits of the motif in each peak that has it
    g = per_peak.groupby('motif_name')
    summ = (mtab.merge(pd.DataFrame({'n_peaks_with_motif': g.size(), 'n_hits': g.sum(),
                                     'hits_per_motif_peak': g.mean()}).reset_index(), on='motif_name', how='left')
                .merge(hits.groupby('motif_name').core_frac.median().rename('median_core_frac').reset_index(),
                       on='motif_name', how='left'))
    summ[['n_peaks_with_motif', 'n_hits']] = summ[['n_peaks_with_motif', 'n_hits']].fillna(0).astype(int)
    summ['n_peaks'] = len(peaks)
    summ['frac_peaks_with_motif'] = summ.n_peaks_with_motif / summ.n_peaks
    summ['short_name'] = summ.motif_name.str.extract(r'(pattern_\d+)$', expand=False)
    summ['head'], summ['lambda'] = head, lam
    for k, v in reversed(list((label or {}).items())):
        summ.insert(0, k, v)
    summ.to_csv(f'{d}/hit_core_ic_summary_{head}_{lam}.csv', index=False)
    return hits[cols], mtab, summ


if __name__ == '__main__':
    import bpnet_sets
    head, lam = os.environ.get('HEAD', 'counts'), os.environ.get('LAMBDA', '0p7')
    base, meta = bpnet_sets.load()
    failed = []
    for r in bpnet_sets.select(meta, sys.argv[1:]).itertuples():
        label = {'TF_CellLine': f'{r.tf_name}_{r.cell_line}', 'tf': r.tf_name, 'cell_line': r.cell_line,
                 'experiment': r.experiment_accession, 'bpnet_set': bpnet_sets.SET, 'dir': r.dir}
        try:
            hits, mtab, summ = run(f'{base}/{r.dir}', head, lam, label)
            print(r.dir, 'hits', len(hits), 'motifs', len(summ), flush=True)
            print(summ[['short_name', 'core_ic', 'total_ic', 'n_peaks_with_motif', 'n_peaks', 'frac_peaks_with_motif',
                        'hits_per_motif_peak', 'median_core_frac']].round(3).to_string(index=False), flush=True)
        except Exception as e:
            print(r.dir, 'FAILED', repr(e), flush=True)
            failed.append(r.dir)
    print('FAILED', failed, flush=True)
    print('DONE', flush=True)
