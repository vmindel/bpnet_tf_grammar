#!/bin/bash
# Information content of the 20-bp TF-MoDISco core of every Fi-NeMo hit (lambda 0.7, counts head), per model,
# then one table over all 2,282 models (Fig. 1D-G).
#   source pipeline/env.sh && bash pipeline/02_motif_core_ic/run.sh
set -euo pipefail
cd "$REPO/pipeline/02_motif_core_ic"
bsub -q short -R "rusage[mem=8000]" -env "all,BPNET_SET=paper,N_TASKS=8" -J "coreic_paper[1-8]" -o $LOGS/coreic_paper_%I.log $PY hit_core_ic.py
bsub -q short -R "rusage[mem=8000]" -env "all,BPNET_SET=all,N_TASKS=24"  -J "coreic_all[1-24]"  -o $LOGS/coreic_all_%I.log   $PY hit_core_ic.py
# -> data/ic_motifs_data.parquet
bsub -q short -w "ended(coreic_paper) && ended(coreic_all)" -J ic_table -o $LOGS/ic_table.log $PY build_ic_table.py
