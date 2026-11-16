#!/bin/bash
# Fig. S2 inputs: ChIP-seq signal over ENCODE cCREs and peak-set signatures of the 440 PAPER experiments.
# Work folder (downloads, per-experiment results): $BPNET_DATA/cre_signatures; tables for the figure: data/cre_signatures/.
#   source pipeline/env.sh && bash pipeline/07_cre_signatures/run.sh
# Needs R with the Bioconductor packages listed in the README ($RBIN) and snakemake (+ cluster-generic executor).
set -euo pipefail
HERE="$REPO/pipeline/07_cre_signatures"
WD="$BPNET_DATA/cre_signatures"; OUT="$REPO/data/cre_signatures"; META="$REPO/metadata/big_subset_meta.orig.csv"
mkdir -p "$WD/external" "$WD/logs" "$OUT"
cd "$WD"
export PATH="$RBIN:$PATH"

# 1. external files: ENCODE SCREEN cCREs (GRCh38-cCREs.bed) and UCSC hg38 CpG islands
#    put GRCh38-cCREs.bed into external/ (ENCODE SCREEN registry of cCREs, 2,348,854 regions)
[ -s external/cpgIslandExt.txt.gz ] || wget -q -O external/cpgIslandExt.txt.gz \
    https://hgdownload.soe.ucsc.edu/goldenPath/hg38/database/cpgIslandExt.txt.gz

# 2. peaks (ENCODE peak file of each experiment -> peaks_all/{cell}/{gene}.bed) and the HOCOMOCO v14 PWMs (pwm/)
$PY "$HERE/submit_downloads.py" --meta "$META" --peaks_dir peaks_all --pwms_dir pwm --log_dir logs/downloads --python $PY

# 3. representative motif per subfamily, trimmed (after the PWM download job has finished)
bsub -q short -K -w "ended(dl_pwms)" -o logs/motifs.log Rscript "$HERE/scripts/build_representative_motifs.R" \
    "$META" pwm "$OUT/representative_motifs_pwms_big_families_trimmed.rds"

SNK="snakemake --executor cluster-generic --jobs 50 --latency-wait 60 --rerun-incomplete --keep-going \
  --cluster-generic-submit-cmd 'bsub -J {rule} -q short -n 1 -R \"span[hosts=1] rusage[mem={resources.mem_mb}]\" -o logs/{rule}_%J.out -e logs/{rule}_%J.err'"
# 4. signal over every cCRE (max fold change) and the correlations between experiments
bsub -q short -W 24:00 -o logs/snakemake_signal.log \
  "$SNK -s $HERE/Snakefile_signal --resources bigwig_slots=10 --config meta=$META repo_data=$OUT"
# 5. 7-mer frequencies and motif enrichment in the top 1,000 peaks
bsub -q short -W 24:00 -o logs/snakemake_motifs.log "$SNK -s $HERE/Snakefile_motifs --config repo_data=$OUT"
# 6. genomic annotation fractions of every peak set
bsub -q short -R "rusage[mem=16000]" -o logs/cre_fractions.log Rscript "$HERE/scripts/cre_fractions.R" \
    "$META" external/GRCh38-cCREs.bed external/cpgIslandExt.txt.gz "$OUT/cre_fractions_all_peaks.parquet"
