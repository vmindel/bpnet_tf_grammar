"""Helpers for the example-peak panels of Figs 2 and 3: one experiment's models, motif hits and pipeline outputs,
DeepLIFT/SHAP attributions of one peak, and the logo / bar plots drawn from them.

Models are used as in the pipeline: CountWrapper(ControlWrapper(BasePairNet)) from bpnet-lite, i.e. predicted log
counts with an all-zero control track, averaged over the 5 folds. Needs a GPU (device='cuda').
"""
import glob
import os
import bbi
import numpy as np
import pandas as pd
import pymisha as pm
import torch
from matplotlib.colors import to_hex
from bpnetlite.bpnet import BasePairNet, ControlWrapper, CountWrapper
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.utils import one_hot_encode
from plotnine import (aes, element_text, facet_wrap, geom_col, geom_hline, geom_rect, ggplot, labs,
                      scale_fill_brewer, scale_fill_manual, scale_x_continuous, theme)
from plotnine_extensions import geom_logo, logo_data, theme_cowplot

import paths

L = 2114                          # BPNet input length


def fold_models(exp_folder):
    """The 5 fold .h5 files of an experiment, sorted by fold."""
    folds = sorted(glob.glob(os.path.join(exp_folder, 'models_*/fold_*/*.h5')))
    assert len(folds) == 5, folds
    return folds


def read_bigbed(exp_folder, counts_only=True, pos_only=True):
    """ENCODE motif instances (sequence_motifs_instances__*.bigBed) of an experiment. counts_only / pos_only
    reproduce the filter of motif_epistasis_direct (head_type ~ 'counts', pos_ patterns)."""
    [bb] = glob.glob(os.path.join(exp_folder, 'sequence_motifs_instances__*.bigBed'))
    with bbi.open(bb) as f:
        h = pd.concat([f.fetch_intervals(c, 0, -1) for c in f.chromsizes], ignore_index=True)
    if counts_only:
        h = h[h.head_type.str.contains('counts')]
    if pos_only:
        h = h[h.name_tfmodisco.str.startswith('pos_')]
    if 'category_type_simple' in h:                       # missing in the 47 old-format bigBeds
        h['is_direct'] = h.category_type_simple == 'direct'   # exact match: 'in' would also hit 'indirect'
    return h.astype({'start': int, 'end': int}).reset_index(drop=True)


def _one(folder, pattern):
    [f] = glob.glob(os.path.join(folder, pattern))
    return f


def load_experiment(dir_, bpnet_set):
    """Pipeline outputs, fold models and motif hits of one experiment.
    Pipeline outputs are read from data/examples/{tag}/ when committed there, else from the experiment folder."""
    folder = paths.experiment_dir(dir_, bpnet_set)
    tag = dir_.replace('/', '_')
    derived = paths.EXAMPLES / tag if (paths.EXAMPLES / tag).is_dir() else folder
    hits = read_bigbed(folder)
    hits['strand'] = np.where(hits.strand == '+', 1, -1)
    return dict(folder=folder, tag=tag,
                epi=pd.read_parquet(_one(derived, f'motif_epistasis_direct_{tag}.parquet')),
                direct_peaks=pd.read_parquet(_one(derived, f'direct_merged_sites_peaks_{tag}.parquet')),
                direct_sites=pd.read_parquet(_one(derived, f'direct_merged_sites_sites_{tag}.parquet')),
                cfolds=fold_models(folder), hits=hits.rename(columns={'strand': 'hit_strand'}))


def query_hits(interval, hits):
    """Rows of `hits` (chrom/start/end) lying fully inside one interval."""
    if isinstance(interval, pd.DataFrame):
        assert len(interval) == 1, f'expected one interval, got {len(interval)}'
        interval = interval.iloc[0]
    chrom, start, end = interval.chrom, interval.start, interval.end
    return hits.query('chrom == @chrom & start >= @start & end <= @end')


def peak_interval(row):
    """misha interval (1-row DataFrame) of a table row with chrom, start, end."""
    return pm.gintervals(row.chrom, int(row.start), int(row.end))


def run_attributions(cfolds, X):
    """DeepLIFT/SHAP of predicted log counts (20 dinucleotide shuffles, seed 0), mean over the folds. X (N, 4, L)."""
    atts = []
    for fold in cfolds:
        w = CountWrapper(ControlWrapper(BasePairNet.from_bpnet(fold)))
        atts.append(deep_lift_shap(w, X, device='cuda', n_shuffles=20, random_state=0, batch_size=64))
    return torch.stack(atts).mean(0)                      # (N, 4, L)


def attribution_logo(interval, hits, cfolds, width=400, step=50):
    """Logo of the fold-mean DeepLIFT/SHAP scores over the central `width` bp of the peak, with the ENCODE motif
    hits of that window as coloured boxes. x = distance from the peak centre."""
    seq = pm.gseq_extract(interval)[0].upper()
    att = run_attributions(cfolds, one_hot_encode(seq)[None])
    scores = att.squeeze().T.numpy()                      # (L, 4)
    n = scores.shape[0]
    w_start, w_end = (n - width) // 2, (n - width) // 2 + width
    df = logo_data(scores[w_start:w_end], alphabet='ACGT', method='none', sequence=seq[w_start:w_end])
    df['position'] = df['position'] - df['position'].min() + w_start
    pos_top = df.loc[df.height > 0].groupby('position').height.sum().max()
    neg_bot = np.nanmin([0, df.loc[df.height < 0].groupby('position').height.sum().min()])

    off = int(interval.iloc[0]['start'])
    peak_hits = query_hits(interval, hits)
    sl = peak_hits.assign(s=peak_hits['start'] - off, e=peak_hits['end'] - off)
    sl = sl.loc[(sl.e > w_start) & (sl.s < w_end)].copy()
    sl['xmin'] = sl.s.clip(lower=w_start) - 0.5
    sl['xmax'] = sl.e.clip(upper=w_end) - 0.5
    sl['ymin'], sl['ymax'] = neg_bot, pos_top
    sl['motif'] = sl['name'].str.replace(r'_(pos|neg)_\w+$', '', regex=True)

    c = n // 2                                            # peak centre
    breaks = np.arange(-(width // 2), width // 2 + 1, step) + c
    return (ggplot(df, aes('position', 'height', label='letter'))
            + geom_rect(sl, aes(xmin='xmin', xmax='xmax', ymin='ymin', ymax='ymax', fill='motif'),
                        inherit_aes=False, alpha=0.25)
            + geom_logo()
            + geom_hline(yintercept=0, size=0.3)
            + scale_fill_brewer(type='qual', palette='Set2')
            + scale_x_continuous(breaks=breaks, labels=[f'{b - c:+d}' if b != c else '0' for b in breaks],
                                 limits=(w_start - 0.5, w_end + 0.5), expand=(0, 0))
            + theme_cowplot()
            + labs(x='distance from peak centre (bp)', y='contribution', fill=None)
            + theme(figure_size=(20, 3)))


# ------------------------------------------------------------------------------- Fig. 2: direct-site knockouts
def rename_peak_cols(df):
    return df.rename(columns={'peak_chrom': 'chrom', 'peak_start': 'start', 'peak_end': 'end'})


def peak_knockouts(interval, sites, peaks):
    """Direct-site deletions of one peak in both contexts (WT, DiSh), predicted log counts.
    levels: one row per site KO + KOALL observed + KOALL additive (= ref - sum of single losses).
    losses: single losses (ref - KO_i, stacked) vs the joint loss (ref - KOALL). synergy = sum single - joint."""
    st = query_hits(interval, rename_peak_cols(sites)).query('bp_masked > 0')
    pk = query_hits(interval, rename_peak_cols(peaks)).iloc[0]
    levels, losses, synergy = [], [], {}
    for c in ['WT', 'DiSh']:
        s = st[st.condition == c]
        ref, koall = pk[c], pk[f'KOALL_{c}']
        loss = (ref - s.prediction).to_numpy()
        site = [f'site {i}' for i in s.site_idx]
        synergy[c] = loss.sum() - (ref - koall)
        levels.append(pd.DataFrame({'context': c,
                                    'what': site + ['all deleted (observed)', 'all deleted (additive)'],
                                    'prediction': list(s.prediction) + [koall, ref - loss.sum()], 'ref': ref}))
        losses.append(pd.DataFrame({'context': c,
                                    'bar': ['sum of single deletions'] * len(site) + ['all deleted together'],
                                    'part': site + ['all deleted'], 'loss': list(loss) + [ref - koall]}))
    levels, losses = pd.concat(levels, ignore_index=True), pd.concat(losses, ignore_index=True)
    lab = {c: f'{c}  (synergy {v:+.2f})' for c, v in synergy.items()}
    for df in (levels, losses):
        df['context'] = pd.Categorical(df.context.map(lab), categories=[lab['WT'], lab['DiSh']])
    levels['what'] = pd.Categorical(levels.what, categories=levels.what.unique())
    losses['bar'] = pd.Categorical(losses.bar, categories=['sum of single deletions', 'all deleted together'])
    losses['part'] = pd.Categorical(losses.part, categories=losses.part.unique())
    return levels, losses, synergy, pk


def loss_bars(losses, cmap):
    """Stacked single-site losses vs the joint loss, faceted by context. Left bar taller -> cooperative
    (synergy > 0); right bar taller -> redundant (synergy < 0)."""
    site_parts = [x for x in losses.part.cat.categories if x != 'all deleted']
    cols = {q: to_hex(cmap(v)) for q, v in zip(site_parts, np.linspace(0.25, 1, len(site_parts)))}
    cols['all deleted'] = 'grey'
    return (ggplot(losses, aes('bar', 'loss', fill='part'))
            + geom_col(width=0.6, color='black', size=0.2)
            + geom_hline(yintercept=0)
            + scale_fill_manual(values=cols)
            + facet_wrap('~context')
            + labs(x='', y='loss of predicted log counts (ref − KO)', fill='')
            + theme_cowplot()
            + theme(axis_text_x=element_text(rotation=20, ha='right'), figure_size=(12, 8)))


# ------------------------------------------------------------------- Fig. 3: direct vs other motifs epistasis
def gain_bars(r, cmap):
    """One peak of motif_epistasis_direct: the gains over `none` of the direct motifs alone and the other motifs
    alone, stacked, next to the gain of the intact peak. WT bar taller -> cooperative (interaction > 0)."""
    gains = pd.DataFrame({
        'bar': ['direct alone + other alone', 'direct alone + other alone', 'all native (WT)'],
        'part': ['direct (p0_only − none)', 'other (non_only − none)', 'WT − none'],
        'gain': [r.p0_only - r.none, r.non_only - r.none, r.wt - r.none]})
    gains['bar'] = pd.Categorical(gains.bar, categories=['direct alone + other alone', 'all native (WT)'])
    gains['part'] = pd.Categorical(gains.part, categories=gains.part.unique())
    gain_parts = [x for x in gains.part.cat.categories if x != 'WT − none']
    cols = {q: to_hex(cmap(v)) for q, v in zip(gain_parts, np.linspace(0.25, 1, len(gain_parts)))}
    cols['WT − none'] = 'grey'
    return (ggplot(gains, aes('bar', 'gain', fill='part'))
            + geom_col(width=0.6, color='black', size=0.2)
            + geom_hline(yintercept=0)
            + scale_fill_manual(values=cols)
            + labs(x='', y='gain over none (log counts)', fill='', title=f'interaction = {r.wt - r.additive:+.2f}')
            + theme_cowplot()
            + theme(figure_size=(12, 8), axis_text_x=element_text(rotation=20, ha='right')))
