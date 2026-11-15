"""Helpers for Fig. S1 (model performance under sequence perturbations): per-experiment test-set ROC / PR curves,
TF-MoDISco report logos, prediction scatters, and TF family labels.

Per experiment, three pipeline outputs (pipeline/01_auroc) hold one prediction per test region, each from its
held-out fold model, with the region's own control (ctl) and an all-zero control (noctl):
  testset_predictions_motifko_{tag}.parquet  test peaks: wild type, Fi-NeMo motif knockouts (p0ko, p0rand, othko),
                                             identity-mixed (mix) and dinucleotide-shuffled (shuf) sequences
  testset_predictions_shuf_{tag}.parquet     peaks and nonpeaks, real and dinucleotide-shuffled sequences
  testset_predictions_mix_{tag}.parquet      peaks and nonpeaks scored on another region's sequence (mix_source)
"""
import base64
import glob
import html
import io
import re
from itertools import repeat
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

import paths

CTLS = ['ctl', 'noctl']
# condition -> (peak column, negative column); '{}' is filled with ctl / noctl
ALL_CONDS = {
    'WT':      ('{}',        '{}'),          # real peaks vs real negatives
    'p0ko':    ('p0ko_{}',   '{}'),          # pattern_0 sites shuffled in the peaks
    'p0rand':  ('p0rand_{}', '{}'),          # pattern_0 sites replaced by random bases
    'othko':   ('othko_{}',  '{}'),          # other positive patterns shuffled, pattern_0 kept
    'shufpk':  ('shuf_{}',   '{}'),          # dinuc-shuffled peaks vs real negatives      (= auroc_shufpk_*)
    'shufall': ('shuf_{}',   'shuf_{}'),     # dinuc-shuffled peaks vs shuffled negatives  (= auroc_shufall_*)
    'mix':     ('mix_{}',    'mix_{}'),      # identity mix of peaks and negatives          (= auroc_mix)
}
STYLE = {'WT': dict(c='k', lw=2, zorder=5), 'p0ko': dict(c='C0'), 'p0rand': dict(c='C1'), 'othko': dict(c='C2'),
         'shufpk': dict(c='C3', ls='--'), 'shufall': dict(c='C3', ls=':'), 'mix': dict(c='C4', ls='--')}


def _folder(dir_, bpnet_set):
    """Pipeline outputs from data/examples/{tag}/ when committed there, else the experiment folder."""
    tag = dir_.replace('/', '_')
    ex = paths.EXAMPLES / tag
    return (ex if ex.is_dir() else paths.experiment_dir(dir_, bpnet_set)), tag


def load(dir_, bpnet_set):
    """Knockout peaks (motifko file) and test-set negatives of one experiment. Negatives: real and dinucleotide-
    shuffled from the shuf file, identity-mixed from the mix file (pred_mix_noctl = pred_noctl[mix_source],
    mix_source = row within the same fold, as for the peaks)."""
    d, tag = _folder(dir_, bpnet_set)
    ko = pd.read_parquet(d / f'testset_predictions_motifko_{tag}.parquet')
    sh = pd.read_parquet(d / f'testset_predictions_shuf_{tag}.parquet')
    mx = pd.read_parquet(d / f'testset_predictions_mix_{tag}.parquet')
    key = ['chrom', 'start', 'end', 'fold', 'label']
    assert len(sh) == len(mx) and (sh[key].values == mx[key].values).all(), 'shuf and mix rows differ'
    regions = []
    for k, f in sh.groupby('fold', sort=True):
        m = mx[mx.fold == k]
        regions.append(f.assign(pred_mix_ctl=m.pred_mix_ctl.values, pred_mix_noctl=f.pred_noctl.values[m.mix_source.values]))
    return ko, pd.concat(regions).query('label == 0')


def fold_data(ko, neg, k, cond, ctl, peaks='all'):
    """Fold k: labels and scores of the peaks (positives) vs the negatives, for one condition and control setting."""
    p, ng = ko[ko.fold == k], neg[neg.fold == k]
    if peaks == 'p0':
        p = p[p.n_p0 > 0]
    pc, nc = (f'pred_{c.format(ctl)}' for c in ALL_CONDS[cond])
    return np.r_[np.ones(len(p)), np.zeros(len(ng))], np.r_[p[pc].values, ng[nc].values]


def auc_table(ko, neg, conds, min_pos=3):
    """AUROC / AUPRC per fold, condition and control setting (all peaks, and peaks with pattern_0), and fold means."""
    rows = []
    for peaks in ['all', 'p0']:
        for k in range(5):
            for ctl in CTLS:
                for cond in conds:
                    y, s = fold_data(ko, neg, k, cond, ctl, peaks)
                    if y.sum() < min_pos:
                        continue
                    rows.append(dict(peaks=peaks, fold=k, cond=cond, ctl=ctl, n_pos=int(y.sum()), n_neg=int((y == 0).sum()),
                                     auroc=roc_auc_score(y, s), auprc=average_precision_score(y, s), baseline=y.mean()))
    per_fold = pd.DataFrame(rows)
    return per_fold, per_fold.groupby(['peaks', 'ctl', 'cond'], sort=False)[['auroc', 'auprc', 'baseline', 'n_pos']].mean()


def mean_curves(ko, neg, cond, ctl, peaks, fgrid, rgrid, min_pos=3):
    """Fold-mean ROC (TPR on an FPR grid) and PR (precision on a recall grid)."""
    tprs, precs = [], []
    for k in range(5):
        y, s = fold_data(ko, neg, k, cond, ctl, peaks)
        if y.sum() < min_pos:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        tprs.append(np.interp(fgrid, fpr, tpr))
        prec, rec, _ = precision_recall_curve(y, s)
        precs.append(np.interp(rgrid, rec[::-1], prec[::-1]))      # recall comes out decreasing -> flip
    return np.mean(tprs, 0), np.mean(precs, 0)


def plot_ko_curves(dir_, bpnet_set, conds, peaks='all', log_fpr=False):
    """Fold-mean ROC and PR curves of one experiment, with (left) and without (right) the control track."""
    ko, neg = load(dir_, bpnet_set)
    _, mean = auc_table(ko, neg, conds)
    fgrid = np.geomspace(1e-4, 1, 400) if log_fpr else np.linspace(0, 1, 201)
    rgrid = np.linspace(0, 1, 201)
    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    for col, ctl in enumerate(CTLS):
        a1, a2 = axes[0, col], axes[1, col]
        for cond in conds:
            tpr, prec = mean_curves(ko, neg, cond, ctl, peaks, fgrid, rgrid)
            m = mean.loc[(peaks, ctl, cond)]
            a1.plot(fgrid, tpr, label=f'{cond}  {m.auroc:.3f}', **STYLE[cond])
            a2.plot(rgrid, prec, label=f'{cond}  {m.auprc:.3f}', **STYLE[cond])
        a1.plot(fgrid, fgrid, c='grey', lw=.8)                          # random classifier
        a2.axhline(mean.loc[(peaks, ctl, 'WT'), 'baseline'], c='grey', lw=.8)
        if log_fpr:
            a1.set_xscale('log')
            a1.set_xlim(1e-4, 1)
        a1.set(title=f'ROC, {ctl}', xlabel='FPR', ylabel='TPR', ylim=(0, 1.02))
        a2.set(title=f'PR, {ctl}', xlabel='recall', ylabel='precision', ylim=(0, 1.02))
        a1.legend(title='AUROC', fontsize=8, loc='lower right')
        a2.legend(title='AUPRC', fontsize=8, loc='lower left')
    n = int(round(mean.loc[(peaks, 'ctl', 'WT'), 'n_pos'] * 5))
    fig.suptitle(f'{dir_} ({bpnet_set}): peaks={peaks}, ~{n} positives, fold-mean curves')
    plt.tight_layout()
    return fig, mean


def _clean(c):
    return html.unescape(' '.join(re.sub(r'<[^>]+>', ' ', c).split()))


def plot_modisco_report(dir_, bpnet_set, n=5, n_matches=3, figsize=None):
    """CWM (forward) logos of the first n patterns of the experiment's ENCODE TF-MoDISco HTML report (counts head),
    titled with pattern id, seqlet count and the top database matches (q-value). Returns (fig, table)."""
    [path] = glob.glob(f'{paths.experiment_dir(dir_, bpnet_set)}/sequence_motifs_report__*/counts/*.html')
    text = open(path).read()
    table = text[text.index('<table class="summary-table">'):]
    body = re.search(r'<tbody>(.*?)</tbody>', table, re.S).group(1)
    rows = re.findall(r'<tr>(.*?)</tr>', body, re.S)[:n]
    fig, axes = plt.subplots(len(rows), 1, figsize=figsize or (6, 1.8 * len(rows)), squeeze=False)
    recs = []
    for ax, row in zip(axes[:, 0], rows):
        pid = re.search(r'class="pattern-id">\s*(.*?)\s*</span>', row, re.S).group(1).split('_counts_')[-1]
        cells = re.findall(r'<td[^>]*>(.*?)</td>', row, re.S)
        nseq = _clean(cells[1])                                                 # number of seqlets
        matches = [(_clean(cells[k]), _clean(cells[k + 1]))                     # (match, q-value, logo) triplets
                   for k in range(6, len(cells) - 1, 3) if _clean(cells[k])][:n_matches]
        png = re.search(r'src="data:image/png;base64,([^"]+)"', row).group(1)   # first image = CWM forward
        ax.imshow(Image.open(io.BytesIO(base64.b64decode(png))))
        ax.set_title(f'{pid}  ({nseq} seqlets)\n' + ', '.join(f'{m} (q={q})' for m, q in matches), fontsize=8)
        ax.axis('off')
        recs.append({'pattern': pid, 'n_seqlets': nseq, **{f'match{i}': m for i, (m, _) in enumerate(matches)},
                     **{f'qval{i}': q for i, (_, q) in enumerate(matches)}})
    fig.tight_layout()
    return fig, pd.DataFrame(recs)


PANELS = (("pred_noctl", "Prediction 0s control"),
          ("pred_shuf_ctl", "Predictions on Dinuc. Shuffled"),
          ("pred_mix_ctl", "Predictions Mixed Identities"))


def plot_motifko_hexbin(dir_, bpnet_set, panels=PANELS, xcol='pred_ctl', gridsize=80, vmax=150, cmap='Reds_r',
                        sharey=False, figsize=(15, 5)):
    """Hexbins over the test peaks: prediction with the own control vs each perturbation."""
    d, tag = _folder(dir_, bpnet_set)
    df = pd.read_parquet(d / f'testset_predictions_motifko_{tag}.parquet', columns=[xcol, *(c for c, _ in panels)])
    fig, ax = plt.subplots(1, len(panels), figsize=figsize, constrained_layout=True, sharey=sharey, squeeze=False, dpi=300)
    ax = ax[0]
    for axi, (ycol, ylab) in zip(ax, panels):
        hb = axi.hexbin(df[xcol], df[ycol], gridsize=gridsize, bins='log', mincnt=1, cmap=cmap, vmin=1, vmax=vmax,
                        rasterized=True)
        axi.set_xlabel('Prediction on Peaks')
        axi.set_ylabel(ylab)
    fig.colorbar(hb, ax=ax, label='Density')
    return fig, ax


def tf_subfamilies(cluster_file=paths.METADATA / 'hocomoco_v14_cluster_list.tsv'):
    """TF -> HOCOMOCO v14 motif-cluster subfamily (first cluster listing the TF), from the cluster table."""
    mc = pd.read_csv(cluster_file, sep='\t')
    rows = []
    for r in mc.itertuples():
        sub = r.Primary_Subfamily.split(' {')[0]
        rows.extend(zip(r.Clustered_Motifs_Gene_Symbols.split('; '), repeat(sub)))
    return pd.DataFrame(rows, columns=['tf_name', 'subfam']).drop_duplicates('tf_name')
