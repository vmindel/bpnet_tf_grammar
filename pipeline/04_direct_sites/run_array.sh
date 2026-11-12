#!/bin/bash
# One LSF array task = one line of the task file. Submitted by run.sh with SET=paper|all, TASKS=tasks/<file>
# and optionally MERGE=1 in the environment (bsub copies it).
#BSUB -q short-gpu
#BSUB -gpu "num=1:j_exclusive=no:gmem=16G"
#BSUB -R "rusage[mem=32000]"
#BSUB -M 32000
#BSUB -R "affinity[thread*4]"
set -euo pipefail
export BPNET_SET=$SET
cd "$REPO/pipeline/04_direct_sites"
nvidia-smi --query-gpu=name,memory.total --format=csv
DIRS=$(sed -n "${LSB_JOBINDEX}p" "$TASKS")
$PY direct_sites.py $DIRS
