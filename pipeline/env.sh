# Source this before submitting anything:   source pipeline/env.sh
# bsub copies the submitting shell's environment into each job, so every job sees these variables.
export REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PY=${PY:-/home/labs/barkailab/vovam/miniconda/envs/seqtfun031/bin/python}   # env from environment.yml
export BPNET_DATA=${BPNET_DATA:-/home/labs/barkailab/vovam/encodedata}             # data root (see modules/paths.py)
export MISHA_HG38=${MISHA_HG38:-/home/labs/barkailab/vovam/MISHADB/hg38/}
export HG38_FA=${HG38_FA:-$BPNET_DATA/hg38.fa}
export PYTHONNOUSERSITE=1 TORCHDYNAMO_DISABLE=1
export LOGS=$REPO/logs
export GPU='-q short-gpu -gpu num=1:j_exclusive=no:gmem=16G'
mkdir -p "$LOGS"
