"""All locations in one place. Override with environment variables; the defaults are the paths on the
WEXAC cluster where the analysis was run.

  BPNET_DATA   data root that holds the ENCODE downloads and every per-experiment output
               (PAPER/{cell}/{TF}/, ALL_BPNET/{cell}/{TF}_{ENCSR}/, klf_sp/, klf_sp_hepg2/, fox_hepg2/)
  MISHA_HG38   misha genome database (hg38) used by pymisha for sequence extraction and interval ops
  HG38_FA      hg38 FASTA (with .fai), used by hit_core_ic.py

Everything committed to the repository (metadata tables, PWMs, cached tables behind the figures, figures)
is addressed relative to the repository root, REPO.
"""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

ROOT = Path(os.environ.get('BPNET_DATA', '/home/labs/barkailab/vovam/encodedata'))
PAPER = ROOT / 'PAPER'              # 440 experiments of the user's selection, {cell}/{TF}/
ALL_BPNET = ROOT / 'ALL_BPNET'      # 1,842 further ENCODE BPNet experiments, {cell}/{TF}_{ENCSR}/

MISHA_HG38 = os.environ.get('MISHA_HG38', '/home/labs/barkailab/vovam/MISHADB/hg38/')
HG38_FA = os.environ.get('HG38_FA', str(ROOT / 'hg38.fa'))

METADATA = REPO / 'metadata'
META_PAPER = METADATA / 'big_subset_meta.csv'            # written by pipeline/00_download/build_paper_table.py
META_PAPER_ORIG = METADATA / 'big_subset_meta.orig.csv'  # the input: the 440-experiment selection
META_ALL = METADATA / 'all_bpnet_meta.csv'                # written by build_all_table.py + build_motif_manifest.py
PWMS = METADATA / 'pwms'

DATA = REPO / 'data'                # small cached tables the figure notebooks read
EXAMPLES = DATA / 'examples'        # per-experiment pipeline outputs of the example experiments
FIGURES = REPO / 'figures'
LOGS = REPO / 'logs'                # LSF logs of the pipeline jobs (git-ignored)


def experiment_dir(dir_, bpnet_set):
    """Data folder of one experiment; dir_ as in the metadata ('{cell}/{TF}' or '{cell}/{TF}_{ENCSR}')."""
    return (PAPER if bpnet_set == 'paper' else ALL_BPNET) / dir_
