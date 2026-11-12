#!/bin/bash
# Threshold-curve tables behind Figs 2 and 3 (after steps 03 and 04). CPU.
#   source pipeline/env.sh && bash pipeline/05_curves/run.sh        (FORCE=1 to rebuild existing tables)
set -euo pipefail
cd "$REPO/pipeline/05_curves"
bsub -q short -R "rusage[mem=16000]" -J curves -o $LOGS/curves.log $PY build_curves.py
