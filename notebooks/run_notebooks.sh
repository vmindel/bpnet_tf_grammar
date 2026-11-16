#!/bin/bash
# Execute the figure notebooks top to bottom on compute nodes (Fig2/3 on a GPU node), outputs saved in place.
#   source pipeline/env.sh && bash notebooks/run_notebooks.sh [Fig1 Fig2 ...]     (OUT=<dir> to write copies there instead)
set -euo pipefail
cd "$REPO/notebooks"
export PATH="$(dirname "$PY"):$PATH"     # the generic python3 kernelspec launches a bare `python`
WHERE=${OUT:+--output-dir $OUT}; WHERE=${WHERE:---inplace}
for nb in ${@:-Fig1 Fig2 Fig3 Fig4 FigS1 FigS3}; do
  case $nb in Fig2|Fig3) Q="$GPU -R rusage[mem=32000]";; *) Q="-q short -R rusage[mem=16000]";; esac
  bsub $Q -J nb_$nb -o $LOGS/nb_$nb.log $PY -m nbconvert --to notebook --execute $WHERE \
       --ExecutePreprocessor.kernel_name=python3 --ExecutePreprocessor.timeout=-1 $nb.ipynb
done
