#!/bin/bash
# Test-set counts AUROC with ENCODE's test regions: no control (testset_auc.py) and with the model's own control
# (testset_auc_mix.py, auroc_ctl), then the per-experiment summary and the table behind Fig. 1A.
#   source pipeline/env.sh && bash pipeline/01_auroc/run.sh
set -euo pipefail
cd "$REPO/pipeline/01_auroc"
G="$GPU -R rusage[mem=24000]"
bsub $G -env "all,BPNET_SET=paper,N_TASKS=8"  -J "auc_paper[1-8]"     -o $LOGS/auc_paper_%I.log    $PY testset_auc.py
bsub $G -env "all,BPNET_SET=all,N_TASKS=16"   -J "auc_all[1-16]"      -o $LOGS/auc_all_%I.log      $PY testset_auc.py
bsub $G -env "all,BPNET_SET=paper,N_TASKS=8"  -J "aucmix_paper[1-8]"  -w "ended(auc_paper)" -o $LOGS/aucmix_paper_%I.log $PY testset_auc_mix.py
bsub $G -env "all,BPNET_SET=all,N_TASKS=16"   -J "aucmix_all[1-16]"   -w "ended(auc_all)"   -o $LOGS/aucmix_all_%I.log   $PY testset_auc_mix.py
bsub -q short -env "all,BPNET_SET=paper" -J collect_paper -w "ended(aucmix_paper)" -o $LOGS/collect_paper.log $PY collect_testset_auc.py
bsub -q short -env "all,BPNET_SET=all"   -J collect_all   -w "ended(aucmix_all)"   -o $LOGS/collect_all.log   $PY collect_testset_auc.py
# -> data/auroc_summary.parquet (Fig. 1A)
bsub -q short -w "ended(collect_paper) && ended(collect_all)" -J auroc_table -o $LOGS/auroc_table.log $PY build_auroc_table.py
