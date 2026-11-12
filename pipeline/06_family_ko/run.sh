#!/bin/bash
# Fig. 4 inputs: KLF/SP and FOX family members in HepG2 (after step 00).
#   source pipeline/env.sh && bash pipeline/06_family_ko/run.sh
set -euo pipefail
cd "$REPO/pipeline/06_family_ko"
# observed-signal bigWigs: KLF/SP via the klf_sp accession table (-> $BPNET_DATA/klf_sp/), FOX directly
bsub -q short -K -J klf_table -o $LOGS/klf_sp_table.log $PY build_klf_sp_table.py
bsub -q long  -J klf_dl  -o $LOGS/klf_sp_download.log $PY download_klf_sp.py
bsub -q short -J fox_dl  -o $LOGS/fox_download.log    $PY download_fox_hepg2.py
# family folders = symlinks into PAPER/, ALL_BPNET/, klf_sp/ (metadata/family_links.tsv)
bsub -q short -K -w "ended(klf_dl) && ended(fox_dl)" -J fam_dirs -o $LOGS/family_dirs.log $PY make_family_dirs.py
G="-q long-gpu -gpu num=1:j_exclusive=no:gmem=16G -R rusage[mem=64000] -M 64000 -R affinity[thread*8]"
for F in klf fox; do
  bsub $G -env "all,FAMILY=$F,N_THREADS=8" -J fam_ko_$F -o $LOGS/family_ko_$F.log $PY family_ko.py
done
