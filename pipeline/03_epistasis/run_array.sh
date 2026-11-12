#!/bin/bash
# One LSF array task = one line of the task file (a list of experiment dirs). Submitted by run.sh with
# SET=paper|all, MOTIFS=finemo|bigbed, TASKS=tasks/<file> in the environment (bsub copies it).
#BSUB -q short-gpu
#BSUB -gpu "num=1:j_exclusive=no:gmem=16G"
#BSUB -R "rusage[mem=32000]"
#BSUB -M 32000
#BSUB -R "affinity[thread*4]"
set -euo pipefail
export BPNET_SET=$SET
cd "$REPO/pipeline/03_epistasis"
nvidia-smi --query-gpu=name,memory.total --format=csv
DIRS=$(sed -n "${LSB_JOBINDEX}p" "$TASKS")
$PY motif_contrib.py $DIRS
