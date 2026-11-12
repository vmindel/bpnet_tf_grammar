#!/bin/bash
# ENCODE metadata, download manifests and downloads (CPU only; network). Run from the repository root:
#   source pipeline/env.sh && bash pipeline/00_download/run.sh
# Every script is resumable (md5-checked) and skips what is already there.
set -euo pipefail
cd "$REPO/pipeline/00_download"
# PAPER: 440 experiments of metadata/big_subset_meta.orig.csv -> metadata/big_subset_meta.csv + $BPNET_DATA/PAPER/manifest*.json
bsub -q short -K -J paper_table -o $LOGS/paper_table.log $PY build_paper_table.py
bsub -q long  -J paper_dl -o $LOGS/paper_download.log $PY download_paper.py manifest.json
# ALL_BPNET: every further ENCODE BPNet model -> metadata/all_bpnet_meta.csv, metadata/excluded.csv, $BPNET_DATA/ALL_BPNET/manifest.json
bsub -q short -K -J all_table -o $LOGS/all_table.log $PY build_all_table.py
bsub -q long  -J all_dl -o $LOGS/all_download.log $PY download_all.py manifest.json
# TF-MoDISco reports + Fi-NeMo instances (tar + bigBed) for ALL_BPNET (PAPER gets them from build_paper_table.py)
bsub -q short -K -w "ended(all_table)" -J all_motif_manifest -o $LOGS/all_motif_manifest.log $PY build_motif_manifest.py
bsub -q long  -w "ended(all_motif_manifest)" -J all_motif_dl -o $LOGS/all_motif_download.log $PY download_all.py manifest_motifs.json
# untrimmed / trimmed TF-MoDISco motif matrices (.meme) for both sets
bsub -q long -n 4 -w "ended(paper_dl) && ended(all_dl)" -J motif_matrices -o $LOGS/motif_matrices.log $PY download_motif_matrices.py
