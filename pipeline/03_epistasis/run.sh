#!/bin/bash
# Shuffle-donor motif epistasis on every peak of every model (no control track). Two passes:
#  1. MOTIFS=finemo -> motif_epistasis_{tag}.parquet         (Fi-NeMo lambda 0.7 hits, p0 = pattern_0)
#     Its wt / shuffled columns are reused by pass 2, so pass 1 must finish first.
#  2. MOTIFS=bigbed -> motif_epistasis_direct_{tag}.parquet  (ENCODE bigBed hits, p0 = 'direct' motifs; Figs 2, 3)
#   source pipeline/env.sh && bash pipeline/03_epistasis/run.sh
# Task files (tasks/) are the ones used for the paper; to rebuild them:
#   bsub -q short -K $PY count_peaks.py && bsub -q short -K $PY make_tasks.py 250000 && bsub -q short -K $PY make_tasks.py 250000 _direct
# Jobs skip finished experiments, so a preempted task is simply resubmitted (tasks/*_rerun.txt were such resubmissions).
set -euo pipefail
cd "$REPO/pipeline/03_epistasis"
for S in paper all; do
  N=$(wc -l < tasks/tasks_${S}.txt)
  MOTIFS=finemo SET=$S TASKS=tasks/tasks_${S}.txt bsub -J "mc_${S}[1-$N]" -oo "$LOGS/mc_${S}_%I.log" < run_array.sh
done
for S in paper all; do
  N=$(wc -l < tasks/tasks_${S}_direct.txt)
  MOTIFS=bigbed SET=$S TASKS=tasks/tasks_${S}_direct.txt bsub -J "mcd_${S}[1-$N]" -w "ended(mc_${S})" -oo "$LOGS/mcd_${S}_%I.log" < run_array.sh
done
# checks: rows vs peaks, NaNs, the additive identity, wt/shuffled equal to the finemo pass -> validate_*.csv
# bsub -q short -K -o $LOGS/validate_epi.log $PY validate.py ; OUT=motif_epistasis_direct bsub -q short -K -o $LOGS/validate_epi_direct.log $PY validate.py
