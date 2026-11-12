"""Build the two family folders of Fig. 4 under the data root, one folder per BPNet model of the family in HepG2:
  klf_sp_hepg2/{TF}_HepG2_{BPNet annotation}/   15 KLF/SP models
  fox_hepg2/{TF}_HepG2_{BPNet annotation}/      15 FOX models
Each holds the model's peaks, bigBed motif instances and models (symlinks into PAPER/ or ALL_BPNET/, step 00) and
its observed-signal bigWigs (klf_sp: symlinks into klf_sp/, from download_klf_sp.py; FOX: real files written by
download_fox_hepg2.py). The links are listed in metadata/family_links.tsv (relative targets, as in the paper run).
Folder names carry the BPNet *annotation* accession, not the ChIP-seq experiment (see the accession tables).
Usage: make_family_dirs.py   (I/O only)"""
import os
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'modules'))
import paths

links = pd.read_csv(paths.METADATA / 'family_links.tsv', sep='\t')
made = missing = 0
for r in links.itertuples():
    d = paths.ROOT / r.family_dir / r.experiment
    d.mkdir(parents=True, exist_ok=True)
    dest = d / r.file
    if not (d / r.target).exists():
        print('target missing (run step 00 / the downloads first):', d / r.target)
        missing += 1
    if not dest.is_symlink():
        os.symlink(r.target, dest)
        made += 1
print(f'{len(links)} links, {made} made, {missing} with a missing target')
