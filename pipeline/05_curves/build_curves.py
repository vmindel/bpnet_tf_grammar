"""Per-experiment threshold curves (definitions in modules/curves.py), from the step-03 and step-04 outputs of all
2,073 experiments that have an ENCODE bigBed. CPU job, ~2k parquet reads per table:
  data/epi_curves_pos.parquet          interaction >= t                    (Fig. 3, curves + rank plot)
  data/effect_curves_direct.parquet    p0_only - none >= t                 (Fig. 3, effect curves)
  data/effect_curves_others.parquet    non_only - none >= t
  data/self_epistasis_merged_direct_calc_wt.parquet   direct-site synergy >= t, native context (Fig. 2)
Experiments with no qualifying peak get no row. Existing tables are rebuilt only with FORCE=1.
Usage: bsub -q short -o $LOGS/curves.log $PY build_curves.py [epi effect self]"""
import glob
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'modules'))
import paths
from curves import TH, TH_EFF, epi_curve, effect_curves, self_epi_curve, label

FORCE = os.environ.get('FORCE') == '1'
WHICH = sys.argv[1:] or ['epi', 'effect', 'self']
ROOT = str(paths.ROOT)


def experiments(pattern):
    """Matching files of both model sets, sorted by path (ALL_BPNET before PAPER)."""
    return sorted(glob.glob(f'{paths.ALL_BPNET}/*/*/{pattern}') + glob.glob(f'{paths.PAPER}/*/*/{pattern}'))


def save(rows, name):
    df = pd.DataFrame(rows)
    df.to_parquet(paths.DATA / name)
    print(name, df.shape, 'TFs', df.tf.nunique(), flush=True)


def todo(*names):
    return FORCE or not all((paths.DATA / n).exists() for n in names)


paths.DATA.mkdir(exist_ok=True)
epi_files = experiments('motif_epistasis_direct_*.parquet')
print('motif_epistasis_direct files', len(epi_files), flush=True)

if 'epi' in WHICH and todo('epi_curves_pos.parquet'):
    rows = []
    for p in epi_files:
        c = epi_curve(p, side='pos')
        if c is not None:
            rows.append({**label(p, ROOT), **dict(zip(TH.astype(str), c))})
    save(rows, 'epi_curves_pos.parquet')

if 'effect' in WHICH and todo('effect_curves_direct.parquet', 'effect_curves_others.parquet'):
    rows = {'direct': [], 'others': []}
    for p in epi_files:
        c = effect_curves(p)
        if c is not None:
            for k in rows:
                rows[k].append({**label(p, ROOT), **dict(zip(TH_EFF.astype(str), c[k]))})
    for k, r in rows.items():
        save(r, f'effect_curves_{k}.parquet')

if 'self' in WHICH and todo('self_epistasis_merged_direct_calc_wt.parquet'):
    pk = experiments('direct_merged_sites_peaks_*.parquet')
    st = experiments('direct_merged_sites_sites_*.parquet')
    assert [os.path.dirname(x) for x in pk] == [os.path.dirname(x) for x in st]
    print('direct_merged_sites experiments', len(pk), flush=True)
    rows = []
    for p, s in zip(pk, st):
        c = self_epi_curve(p, s)
        if c is not None:
            rows.append({**label(p, ROOT), **dict(zip(TH.astype(str), c))})
    save(rows, 'self_epistasis_merged_direct_calc_wt.parquet')
print('DONE', flush=True)
