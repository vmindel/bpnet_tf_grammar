"""Family knockouts in HepG2 (Fig. 4): do family members' predicted binding profiles differ because of the family's
own motif or because of the other motifs around it? Script version of the notebooks klf_hepg2.ipynb /
fox_hepg2.ipynb that produced the paper's tables (same steps, constants and seeds; their exploratory plots dropped).

FAMILY=klf  15 KLF/SP models, {ROOT}/klf_sp_hepg2/   'direct' = ENCODE direct, or any instance with a GC-box
            (SP1 / KLF16 HOCOMOCO v14 PWM hit, p < 1e-4, centre inside the instance)
FAMILY=fox  15 FOX models,    {ROOT}/fox_hepg2/      'direct' = ENCODE direct, or any instance with a FOX PWM hit
            (7 HOCOMOCO v14 FOX PWMs, same rule)
Steps:
 1. Windows: union of all members' conservative IDR peaks (misha canonic), 2114 bp around each centre; windows with
    N or off the chromosome dropped.
 2. Observed: log1p(plus + |minus|) observed-signal bigWig sum over the central 1 kb, per member -> {fam}_obs.
 3. Predicted: log counts, no control (CountWrapper(ControlWrapper(BasePairNet))), mean of 5 folds -> {fam}_pred.
 4. Motif instances: every member's bigBed instances (counts head, pos patterns), labelled fam_direct as above.
    Masks over the central 1 kb: direct bases, non-direct bases not covered by a direct instance, all motif bases.
 5. Knockouts: masked bases taken from N_SHUF = 5 dinucleotide shuffles of the central 1 kb (seed SEED + chunk start),
    predicted by every member's models, mean over shuffles.
 6. Pairwise Pearson r between members, over windows: observed, predicted (WT) and each knockout
    -> {fam}_pairwise_corr_famdirect.csv.
Outputs: data/{fam}_obs.parquet, data/{fam}_pred.parquet, data/{fam}_pairwise_corr_famdirect.csv (used by Fig. 4);
{ROOT}/family_ko/{fam}_windows.parquet, {fam}_motif_instances_famdirect.parquet, {fam}_mut_pred_famdirect[_sd].parquet.
Columns / experiment order: sorted folder names. GPU job, ~2-4 h (run.sh)."""
import glob
import os
import sys
from pathlib import Path
import bbi
import numpy as np
import pandas as pd
import pymisha as pm
import moodspy as mp
import torch
from bpnetlite.bpnet import BasePairNet, ControlWrapper, CountWrapper
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import one_hot_encode
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'modules'))
import paths
from peaks import read_bigbed

FAMILY = os.environ['FAMILY']
FAM_DIR = {'klf': 'klf_sp_hepg2', 'fox': 'fox_hepg2'}[FAMILY]
if FAMILY == 'klf':
    PWM_FILES = [paths.PWMS / 'SP1.H14CORE.2.P.B_meme_format.meme', paths.PWMS / 'KLF16.H14CORE.0.SM.B_meme_format.meme']
else:
    PWM_FILES = sorted(paths.PWMS.glob('FOX*.pwm'))                  # HOCOMOCO v14 FOX PWMs
OUT = paths.ROOT / 'family_ko'
OUT.mkdir(exist_ok=True)
pm.gsetroot(paths.MISHA_HG38)

L, OUT_W = 2114, 1000
OFF = (L - OUT_W) // 2              # 557: the 1 kb output window starts here inside the input
LO, HI = OFF, OFF + OUT_W           # region that is mutated (and shuffled for the donors)
PAD, P = 10, 1e-4                   # instance padding for the PWM scan; PWM-hit p-value
N_SHUF, CHUNK, SEED = 5, 512, 0     # donors per window; donor seed = SEED + chunk start
N_JOBS = int(os.environ.get('N_THREADS', 8))
CONDS = ['nondirect', 'direct', 'all']


def observed_counts(exp_folder, peaks):
    """log1p(plus + |minus|) summed over each interval: the BPNet count-head target."""
    tot = 0
    for strand in ('plus', 'minus'):
        [bw] = glob.glob(os.path.join(exp_folder, f'observed_signal_profile_{strand}__*.bigWig'))
        s = bbi.stackup(bw, peaks.chrom, peaks.start, peaks.end, bins=1, summary='sum', missing=0)[:, 0]
        tot = tot + np.abs(s)
    return np.log1p(tot)


def mask_matrix(win, ivs, lo=LO, hi=HI):
    """(N, L) bool: True where a base of `ivs` falls inside [lo, hi) of the input window."""
    M = np.zeros((len(win), L), bool)
    ov = pm.gintervals_neighbors(win, ivs, maxneighbors=10_000, mindist=0, maxdist=0)
    if ov is None:
        return M
    ws = ov.iloc[:, 1].to_numpy()                                # cols 0-3 = win, 4-6 = the interval
    a = np.clip(ov.iloc[:, 5].to_numpy() - ws, lo, hi)           # window-relative, clipped to [lo, hi)
    b = np.clip(ov.iloc[:, 6].to_numpy() - ws, lo, hi)
    for i, s, e in zip(ov['example_idx'].to_numpy(int), a, b):
        M[i, s:e] = True                                         # empty slice when the hit is outside [lo, hi)
    return M


@torch.no_grad()
def predict_mean(fold_models, X, batch=1024):
    """Mean over folds of predicted log counts; X (n, 4, L) int8 on CPU -> (n,) numpy."""
    out = []
    for i in range(0, len(X), batch):
        x = X[i:i + batch].cuda().float()
        out.append(torch.stack([m(x).flatten() for m in fold_models]).mean(0).cpu())
    return torch.cat(out).numpy()


def read_hocomoco_pwm(path):
    """HOCOMOCO .pwm: '>name' line, then one row per position with A C G T log-odds (natural log, uniform bg)."""
    with open(path) as f:
        name = f.readline()[1:].strip()
        mat = np.loadtxt(f)                                      # (length, 4)
    return mp.Motif.from_log_odds(mat.T, name=name, log_base=np.e, background=[0.25] * 4)


def pairwise(df, label):
    """Correlation between experiments, one row per unordered pair."""
    c = df.corr()
    iu = np.triu_indices(len(c), k=1)
    return pd.DataFrame({'level_0': c.index[iu[0]], 'level_1': c.columns[iu[1]], label: c.values[iu]})


# 1. windows -------------------------------------------------------------------------------------------------
exp_folders = sorted(d.rstrip('/') for d in glob.glob(f'{paths.ROOT}/{FAM_DIR}/*/'))   # fixes the column order
names = [os.path.basename(d) for d in exp_folders]
print(FAMILY, len(names), 'models:', names, flush=True)
peaks_loaded = pd.concat([pm.gintervals_import_bed(f) for d in exp_folders
                          for f in glob.glob(f'{d}/*chipseq_conservative_idr_thresholded_peaks*')])
peaks_canonic = pm.gintervals_force_range(pm.gintervals_canonic(peaks_loaded))
win = pm.gintervals_force_range(pm.gintervals_normalize(peaks_canonic, L))[['chrom', 'start', 'end']]
win = win.reset_index(drop=True)
seqs = pd.Series(pm.gseq_extract(win)).str.upper()
ok = (~seqs.str.contains('N') & (seqs.str.len() == L)).to_numpy()   # drops N-containing and chromosome-edge windows
win, seqs = win[ok].reset_index(drop=True), seqs[ok].reset_index(drop=True)
win['example_idx'] = np.arange(len(win))
N = len(win)
win1kb = win.assign(start=win.start + OFF, end=win.start + OFF + OUT_W)   # exact centre of each input window
X = torch.stack([one_hot_encode(s) for s in seqs]).to(torch.int8)           # (N, 4, L)
print(N, 'windows kept of', len(peaks_canonic), flush=True)

# 2. observed -------------------------------------------------------------------------------------------------
obs = pd.DataFrame({n: observed_counts(d, win1kb) for n, d in zip(names, exp_folders)})
obs.to_parquet(paths.DATA / f'{FAMILY}_obs.parquet')

# 4. motif instances and masks --------------------------------------------------------------------------------
motif_instances = pd.concat([read_bigbed(d).assign(exp=n) for n, d in zip(names, exp_folders)], ignore_index=True)
motifs = (sum((mp.read_motifs(str(f)) for f in PWM_FILES), []) if FAMILY == 'klf'
          else [read_hocomoco_pwm(f) for f in PWM_FILES])
padded = motif_instances[['chrom', 'start', 'end']].assign(start=lambda d: d.start - PAD, end=lambda d: d.end + PAD)
motif_seqs = list(pm.gseq_extract(padded))                       # row i <-> 'seq{i}'
pos = mp.match_motifs(motifs, motif_seqs, out='positions', bg='even', p_cutoff=P, n_jobs=N_JOBS)
pos['row'] = pos.seq.astype(str).str[3:].astype(int)
ilen = (motif_instances.end - motif_instances.start).to_numpy()
centre = (pos.start + pos.end).to_numpy() / 2                   # hit centre, padded-sequence coords
inside = (centre >= PAD) & (centre < PAD + ilen[pos.row.to_numpy()])
motif_instances['pwm_hit'] = False
motif_instances.loc[np.unique(pos.row[inside]), 'pwm_hit'] = True
best = pos[inside].sort_values('score').drop_duplicates('row', keep='last')   # best PWM per instance
motif_instances['pwm_motif'] = None
motif_instances.loc[best.row.to_numpy(), 'pwm_motif'] = best.motif.astype(str).to_numpy()
motif_instances['fam_direct'] = motif_instances.is_direct | motif_instances.pwm_hit
print('relabelled to direct:', int((motif_instances.pwm_hit & ~motif_instances.is_direct).sum()), flush=True)
motif_instances[['chrom', 'start', 'end', 'exp', 'name', 'name_tfmodisco', 'category_type_simple', 'is_direct',
                 'pwm_hit', 'pwm_motif', 'fam_direct']].to_parquet(OUT / f'{FAMILY}_motif_instances_famdirect.parquet')

iv = lambda d: pm.gintervals_canonic(d[['chrom', 'start', 'end']].reset_index(drop=True))
fd = motif_instances.fam_direct
direct = iv(motif_instances[fd])
nondir = pm.gintervals_diff(iv(motif_instances[~fd]), direct)   # non-direct bases not covered by any direct instance
allmot = iv(motif_instances)                                   # = direct ∪ nondir
masks = {'nondirect': mask_matrix(win, nondir), 'direct': mask_matrix(win, direct), 'all': mask_matrix(win, allmot)}
assert not (masks['nondirect'] & masks['direct']).any()
assert (masks['all'] == (masks['nondirect'] | masks['direct'])).all()
bp = pd.DataFrame({k: m.sum(1) for k, m in masks.items()})
win1kb[['chrom', 'start', 'end']].join(bp.add_prefix('bp_')).to_parquet(OUT / f'{FAMILY}_windows.parquet')

# 3. predicted (WT) -------------------------------------------------------------------------------------------
models = {n: [CountWrapper(ControlWrapper(BasePairNet.from_bpnet(f))).cuda().eval()   # no control
              for f in sorted(glob.glob(f'{d}/models__*/fold_*/*.h5'))]
          for n, d in zip(names, exp_folders)}
assert all(len(v) == 5 for v in models.values())
print(torch.cuda.get_device_name(0), flush=True)
wt = pd.DataFrame({n: predict_mean(ms, X) for n, ms in models.items()})
wt.to_parquet(paths.DATA / f'{FAMILY}_pred.parquet')

# 5. knockouts ------------------------------------------------------------------------------------------------
ko = {c: {n: np.zeros((N, N_SHUF), np.float32) for n in names} for c in CONDS}
for c0 in range(0, N, CHUNK):
    sl = slice(c0, min(c0 + CHUNK, N))
    Xs = dinucleotide_shuffle(X[sl], start=LO, end=HI, n=N_SHUF, random_state=SEED + c0).to(torch.int8)
    for c in CONDS:
        m = torch.from_numpy(masks[c][sl])[:, None, None, :]           # (n, 1, 1, L)
        Xk = torch.where(m, Xs, X[sl][:, None]).reshape(-1, 4, L)      # masked bases from the donor, rest WT
        for n, ms in models.items():
            ko[c][n][sl] = predict_mean(ms, Xk).reshape(-1, N_SHUF)
    if c0 % (CHUNK * 10) == 0:
        print(c0, '/', N, flush=True)
big_df = pd.DataFrame({f'{n}_{c}': ko[c][n].mean(1) for c in CONDS for n in names})   # {experiment}_{condition}
big_sd = pd.DataFrame({f'{n}_{c}': ko[c][n].std(1) for c in CONDS for n in names})
big_df.to_parquet(OUT / f'{FAMILY}_mut_pred_famdirect.parquet')
big_sd.to_parquet(OUT / f'{FAMILY}_mut_pred_famdirect_sd.parquet')

# 6. pairwise correlations ------------------------------------------------------------------------------------
merged_corr = pairwise(obs, 'observed').merge(pairwise(wt, 'predicted'), on=['level_0', 'level_1'])
for cond in CONDS:
    cdf = big_df.filter(regex=f'_{cond}$')                      # anchored: 'direct' alone also matches '_nondirect'
    cdf.columns = cdf.columns.str.removesuffix(f'_{cond}')
    merged_corr = merged_corr.merge(pairwise(cdf[names], cond), on=['level_0', 'level_1'])
merged_corr.to_csv(paths.DATA / f'{FAMILY}_pairwise_corr_famdirect.csv', index=False)
print(merged_corr.drop(columns=['level_0', 'level_1']).corr().round(3).to_string())
print('DONE', flush=True)
