# The motif set for the enrichment scans (Fig. S2): the representative HOCOMOCO v14 motif of every subfamily in the
# metadata (representative_mot_id), each trimmed of low-information edges (universalmotif::trim_motifs, min.ic = 0.3).
# Usage: Rscript build_representative_motifs.R <metadata.csv> <pwm_dir> <out.rds>
suppressPackageStartupMessages({
  library(universalmotif)
  library(stringr)
})
args <- commandArgs(trailingOnly = TRUE)
meta_path <- args[1]; pwm_dir <- args[2]; out_path <- args[3]

meta.data <- read.csv(meta_path)
unique_needed_pwm <- unique(meta.data$representative_mot_id)

pwm_files <- list.files(pwm_dir, pattern = "\\.pwm$", full.names = TRUE)
representative <- lapply(pwm_files, function(f) {
  mot <- trim_motifs(read_matrix(f, positions = "rows"), min.ic = .3)
  mot@type <- "PWM"
  mot
})
names(representative) <- str_remove(basename(pwm_files), "\\.pwm$")
representative <- representative[unique_needed_pwm]
saveRDS(representative, out_path)
message(length(representative), " motifs -> ", out_path)
