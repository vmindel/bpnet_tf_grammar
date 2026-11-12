#!/bin/bash
# Single direct-site knockouts, in the native peak (WT) and in a dinucleotide-shuffled peak that keeps only the
# direct sites (DiSh). Two passes, both needed for Fig. 2:
#  1. direct_sites_{sites,peaks}_{tag}.parquet         every direct hit knocked out alone, and all together
#  2. MERGE=1 -> direct_merged_sites_{sites,peaks}_{tag}.parquet
#     direct sites sharing >= 1 base merged into one site; single sites are copied from pass 1.
# Needs step 03 (motif_epistasis_direct_*) for its consistency checks.
#   source pipeline/env.sh && bash pipeline/04_direct_sites/run.sh
# Task files: bsub -q short -K $PY make_tasks.py 150000 _direct   (reads ../03_epistasis/tasks/peak_counts.csv)
# The paper MERGE=1 run also passed -R "select[hname!=lgn09]" to bsub (that node was avoided at the time).
set -euo pipefail
cd "$REPO/pipeline/04_direct_sites"
for S in paper all; do
  N=$(wc -l < tasks/tasks_${S}_direct.txt)
  SET=$S TASKS=tasks/tasks_${S}_direct.txt bsub -J "ds_${S}[1-$N]" -oo "$LOGS/ds_${S}_%I.log" < run_array.sh
done
for S in paper all; do
  N=$(wc -l < tasks/tasks_${S}_direct.txt)
  MERGE=1 SET=$S TASKS=tasks/tasks_${S}_direct.txt bsub -J "dm_${S}[1-$N]" -w "ended(ds_${S})" -oo "$LOGS/dm_${S}_%I.log" < run_array.sh
done
# checks -> validate.csv, validate_merged.csv
bsub -q short -w "ended(ds_paper) && ended(ds_all)" -o $LOGS/validate_ds.log $PY validate.py
bsub -q short -w "ended(dm_paper) && ended(dm_all)" -o $LOGS/validate_dm.log $PY validate_merged.py
