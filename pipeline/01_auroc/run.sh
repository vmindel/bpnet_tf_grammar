#!/bin/bash
# Test-set counts AUROC/AUPRC on ENCODE's test folds (peaks vs GC-matched nonpeaks) under sequence perturbations,
# with the model's own control track (ctl) and with an all-zero control (noctl), and the tables behind Figs 1A and S1.
#   source pipeline/env.sh && bash pipeline/01_auroc/run.sh
# Order: real sequence (no control) -> own control + identity-mixed sequences -> dinucleotide-shuffled sequences
#        -> per-experiment summary -> Fi-NeMo motif knockouts on the test peaks -> AUROC for every perturbation
#        -> one table over all 2,282 models.
set -euo pipefail
cd "$REPO/pipeline/01_auroc"
G="$GPU -R rusage[mem=24000]"
G64="$GPU -R rusage[mem=64000]"
for S in paper all; do
  if [ $S = paper ]; then NA=8; NS=12; NK=8; NC=5; else NA=16; NS=32; NK=24; NC=20; fi
  bsub $G   -env "all,BPNET_SET=$S,N_TASKS=$NA" -J "auc_$S[1-$NA]"    -o $LOGS/auc_${S}_%I.log    $PY testset_auc.py
  bsub $G   -env "all,BPNET_SET=$S,N_TASKS=$NA" -J "aucmix_$S[1-$NA]" -w "ended(auc_$S)"    -o $LOGS/aucmix_${S}_%I.log $PY testset_auc_mix.py
  bsub $G64 -env "all,BPNET_SET=$S,N_TASKS=$NS" -J "shuf_$S[1-$NS]"   -w "ended(aucmix_$S)" -o $LOGS/shuf_${S}_%I.log   $PY testset_auc_shuf.py
  bsub -q short -env "all,BPNET_SET=$S" -J "collect_$S" -w "ended(shuf_$S)" -o $LOGS/collect_$S.log $PY collect_testset_auc.py
  bsub $G64 -env "all,BPNET_SET=$S,N_TASKS=$NK" -J "motifko_$S[1-$NK]" -w "ended(aucmix_$S)" -o $LOGS/motifko_${S}_%I.log $PY testset_motif_ko.py
  bsub -q short -env "all,BPNET_SET=$S" -J "append_$S" -w "ended(motifko_$S) && ended(shuf_$S)" -o $LOGS/append_shuf_$S.log $PY append_shuf_to_motifko.py
  bsub -q short -env "all,BPNET_SET=$S,N_TASKS=$NC" -J "aucko_$S[1-$NC]" -w "ended(append_$S)" -o $LOGS/aucko_${S}_%I.log $PY testset_auc_ko.py
done
# -> data/auroc_summary.parquet (Fig. 1A) and data/bpnet_big_table.parquet (Fig. S1)
W="ended(collect_paper) && ended(collect_all)"
bsub -q short -w "$W" -J auroc_table -o $LOGS/auroc_table.log $PY build_auroc_table.py
bsub -q short -w "$W && ended(aucko_paper) && ended(aucko_all)" -J big_table -o $LOGS/big_table.log $PY build_big_table.py
