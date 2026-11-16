"""Per-experiment threshold curves behind Figs 2 and 3, and the plots drawn from them.

Each curve is the fraction of an experiment's peaks whose value passes threshold t, for every t in a grid. Built by
pipeline/05_curves/build_curves.py into data/*.parquet: one row per experiment with columns set, cell, exp, tf and
one column per threshold (named str(t)).

  epi_curve        motif_epistasis_direct_*: interaction = wt - (p0_only + non_only - none) >= t, over peaks with both
                   direct (n_p0 > 0) and other (n_other > 0) motif hits; elsewhere interaction is 0 by construction.
  effect_curves    the same peaks: direct effect p0_only - none >= t and other-motif effect non_only - none >= t.
  self_epi_curve   direct_merged_sites_*: synergy among the direct sites of a peak, in its native context (WT),
                   synergy = sum_i (ref - KO_i) - (ref - KOALL), over peaks with >= 2 direct sites (in both WT and
                   DiSh). Sites with no base inside the knockout window (bp_masked = 0) are left out.
Linear-scale versions (Fig. S3), predictions converted from log counts to counts (expm1) and the departure from
additivity expressed as a fraction of the intact peak's counts:
  epi_curve_linear     (wt - none - (p0_only - none) - (non_only - none)) / wt
  self_epi_curve(..., linear=True)   (KOALL + sum_i (ref - KO_i) - ref) / ref
  side='neg' counts values <= -t (redundancy).
"""
import os
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt

TH = np.round(np.arange(0, 2.0001, 0.02), 2)        # epistasis / synergy thresholds (log counts)
TH_EFF = np.round(np.arange(0, 5.0001, 0.05), 2)    # single-motif-set effect thresholds (log counts)
HIGHLIGHT = ['KLF16', 'SP1', 'FOXA1', 'FOXA2']


def _frac_ge(x, th):
    """Fraction of x >= t for every t in th."""
    x = np.sort(x)
    return 1 - np.searchsorted(x, th, side='left') / len(x)


def epi_curve(p, th=TH, side='pos'):
    """Fraction of peaks with interaction >= t (side='pos') or <= -t (side='neg'); None if no peak qualifies."""
    d = (pl.read_parquet(p, columns=['interaction', 'n_p0', 'n_other'])
         .filter((pl.col('n_p0') > 0) & (pl.col('n_other') > 0)))
    x = d['interaction'].to_numpy()
    if side == 'neg':
        x = -x
    if len(x) == 0:
        return None
    return _frac_ge(x, th)


COUNT_COLS = ['none', 'non_only', 'p0_only', 'wt']


def epi_curve_linear(p, th=TH, side='pos'):
    """Linear-scale interaction relative to wt counts, >= t (side='pos') or <= -t ('neg'); peaks with both motif sets."""
    x = (pl.scan_parquet(p)
         .filter((pl.col('n_p0') > 0) & (pl.col('n_other') > 0))
         .with_columns(np.expm1(pl.col(COUNT_COLS)))
         .select(((pl.col('wt') - pl.col('none') - (pl.col('non_only') - pl.col('none'))
                   - (pl.col('p0_only') - pl.col('none'))) / pl.col('wt')).alias('interaction'))
         .filter(pl.col('interaction').is_finite())
         .collect()['interaction'].to_numpy())
    if len(x) == 0:
        return None
    return _frac_ge(-x if side == 'neg' else x, th)


def effect_curves(p, th=TH_EFF):
    """{'direct': ..., 'others': ...}: fraction of peaks with p0_only - none >= t / non_only - none >= t."""
    d = (pl.read_parquet(p, columns=['p0_only', 'non_only', 'none', 'n_p0', 'n_other'])
         .filter((pl.col('n_p0') > 0) & (pl.col('n_other') > 0)))
    if d.is_empty():
        return None
    return dict(direct=_frac_ge((d['p0_only'] - d['none']).to_numpy(), th),
                others=_frac_ge((d['non_only'] - d['none']).to_numpy(), th))


def self_synergy(peaks, sites, linear=False):
    """Per peak with >= 2 direct sites in both contexts: sum of single losses, the joint loss and
    synergy_{WT,DiSh} = (KOALL + sum single losses) - ref. Index = peak_name. None if no site was knocked out.
    linear=True: predictions in counts (expm1 of log counts) and synergy divided by ref."""
    if linear:
        sites = sites.assign(**{c: np.expm1(sites[c]) for c in ['prediction_ref', 'prediction']})
        peaks = peaks.assign(**{c: np.expm1(peaks[c]) for c in ['WT', 'DiSh', 'KOALL_WT', 'KOALL_DiSh']})
    st = sites[sites.bp_masked > 0].copy()
    if len(st) == 0:
        return None
    st['d_i'] = st.prediction_ref - st.prediction                      # loss when only site i is removed
    st['ov'] = st['overlaps'] if 'overlaps' in st else st['n_merged'] > 1
    per = (st.groupby(['peak_name', 'condition'])
             .agg(sum_single=('d_i', 'sum'), k=('d_i', 'size'), any_overlap=('ov', 'any'))
             .unstack('condition'))
    per.columns = [f'{a}_{b}' for a, b in per.columns]                 # sum_single_WT, k_DiSh, ...
    w = peaks.set_index('peak_name').join(per, how='inner')
    w = w[(w.k_WT >= 2) & (w.k_DiSh >= 2)]
    for c in ['WT', 'DiSh']:
        ref, koall = w[c], w[f'KOALL_{c}']
        w[f'ref_{c}'] = ref
        w[f'ref_add_{c}'] = koall + w[f'sum_single_{c}']               # additive expectation for the intact peak
        w[f'synergy_{c}'] = w[f'ref_add_{c}'] - ref                    # = sum single - (ref - KOALL); > 0 cooperative
        if linear:
            w[f'synergy_{c}'] /= ref
    return w


def self_epi_curve(peaks_file, sites_file, th=TH, context='WT', linear=False, side='pos'):
    w = self_synergy(pd.read_parquet(peaks_file), pd.read_parquet(sites_file), linear=linear)
    if w is None or len(w) == 0:
        return None
    x = w[f'synergy_{context}'].to_numpy()
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return None
    return _frac_ge(-x if side == 'neg' else x, th)


def label(path, root):
    """set, cell, exp, tf of an experiment file under root ({set}/{cell}/{exp}/file)."""
    st, cell, exp = os.path.relpath(path, root).split(os.sep)[:3]
    return dict(set=st, cell=cell, exp=exp, tf=exp.split('_')[0])


def per_tf(curves, th=TH):
    """Mean and SEM over each TF's experiments (one curve per TF, so TFs with many experiments don't dominate)."""
    cols = th.astype(str).tolist()
    g = curves.groupby('tf')[cols]
    return g.mean(), g.sem()


# ----------------------------------------------------------------------------------------------- plots
def plot_epi_curves(per_tf, highlight=HIGHLIGHT, ax=None, logy=True,
                    ylabel='fraction of peaks with epistasis ≥ t', sem=None, ymin=1e-4,
                    xlabel='epistasis threshold t (log counts)', show_mean=False):
    """Grey line per TF, IQR band and median across TFs, highlighted TFs (± SEM over their experiments)."""
    ax = ax or plt.gca()
    th = per_tf.columns.astype(float)
    # On a log axis zeros push path vertices off the canvas: summaries use raw values, masking is at plot time only.
    mask = (lambda y: y.where(y > 0)) if logy else (lambda y: y)
    band = (lambda y: np.clip(y, ymin, None)) if logy else (lambda y: y)

    for _, r in per_tf.iterrows():
        ax.plot(th, mask(r).values, color='0.7', lw=0.4, alpha=0.3, rasterized=True)
    q25, q75 = per_tf.quantile(0.25), per_tf.quantile(0.75)
    ax.fill_between(th, band(q25.values), band(q75.values), color='tab:purple', alpha=0.25, lw=0,
                    label='IQR across TFs')
    if show_mean:
        ax.plot(th, mask(per_tf.mean()).values, color='darkslateblue', lw=2, label=f'mean over {len(per_tf)} TFs')
    ax.plot(th, mask(per_tf.median()).values, color='darkslateblue', lw=1.5, ls='--', label='median')
    for j, tf in enumerate(t for t in highlight if t in per_tf.index):
        c, m = plt.cm.Set2(j), per_tf.loc[tf]
        ax.plot(th, mask(m).values, color=c, lw=1.5, label=tf)
        if sem is not None and tf in sem.index:
            s = sem.loc[tf].values                                     # NaN when the TF has a single experiment
            ax.fill_between(th, band(m.values - s), band(m.values + s), color=c, alpha=0.3, lw=0)
    if logy:
        ax.set_yscale('log', nonpositive='mask')
        ax.set_ylim(ymin, 1)
    ax.set(xlabel=xlabel, ylabel=ylabel, xlim=(th.min(), th.max()))
    ax.spines[['top', 'right']].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_rank(curves, col='0.7', top_n=30, highlight=HIGHLIGHT, quantity='ε'):
    """Per-TF median (over experiments) of the fraction of peaks passing threshold `col`: ranked step curve of the
    TFs with median > 0, and a bar plot of the top_n. quantity: name of the thresholded value in the y label."""
    hl_col = {tf: plt.cm.Set2(i) for i, tf in enumerate(highlight)}
    s = curves[['tf', col]].dropna().groupby('tf')[col].median().sort_values(ascending=False)
    nz = s[s > 0]
    rank = pd.Series(np.arange(len(nz)), index=nz.index)
    ymin = 10 ** np.floor(np.log10(nz.min()))
    zero_hl = [t for t in highlight if t in s.index and s[t] == 0]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3), gridspec_kw={'width_ratios': [3, 1]},
                                 constrained_layout=True)
    a1.step(np.arange(len(nz)), nz.values, where='mid', color='tab:blue', lw=1)
    a1.set_yscale('log', nonpositive='mask')
    a1.set(xlim=(-0.5, len(nz) - 0.5), ylim=(ymin, 1),
           xlabel=f'TF rank ({len(nz)} of {len(s)} TFs with median > 0)')
    a1.set_ylabel(f'median fraction of peaks with {quantity} ≥ {col} ln\n(×{np.exp(float(col)):.2f} over expected)')
    for j, tf in enumerate(t for t in highlight if t in nz.index):
        a1.annotate(tf, (rank[tf], nz[tf]), xytext=(6, 10 + 8 * j), textcoords='offset points',
                    fontsize=7, color=hl_col[tf], arrowprops=dict(arrowstyle='-', lw=0.5, color=hl_col[tf]))
    if zero_hl:
        a1.text(0.99, 0.02, f"median = 0: {', '.join(zero_hl)}", transform=a1.transAxes,
                ha='right', va='bottom', fontsize=7)
    top = s.head(top_n)[::-1]
    a2.barh(top.index, top.values, color='tab:blue', height=0.8)
    a2.tick_params(axis='y', labelsize=6)
    a2.margins(y=0.01)
    a2.set_xlabel('median fraction')
    a2.set_title(f'top {top_n}', fontsize=8)
    for a in (a1, a2):
        a.spines[['top', 'right']].set_visible(False)
    return fig, s
