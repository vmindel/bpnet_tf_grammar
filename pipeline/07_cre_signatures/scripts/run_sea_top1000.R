library(nanoparquet)
library(rtracklayer)
library(GenomicRanges)
library(stringr)
library(universalmotif)
library(TFBSTools)
library(motifmatchr)
library(BSgenome.Hsapiens.UCSC.hg38)

peak_path <- snakemake@input[["peaks"]]
pwm_path  <- snakemake@input[["pwm"]]
out_path  <- snakemake@output[[1]]

# Parse cell line and TF from path (peaks_all/{cell_line}/{tf}.bed)
parts    <- str_split(peak_path, "/")[[1]]
cellline <- parts[length(parts) - 1]
tf       <- str_remove(parts[length(parts)], "\\.bed$")

# Load and convert PWMs
needed_set <- readRDS(pwm_path)
pwm_list   <- convert_motifs(needed_set, class = "TFBSTools-PWMatrix")
pwm_list   <- do.call(PWMatrixList, pwm_list)

# Load peaks and keep top 1000 by score column
peaks_gr <- import(peak_path, "narrowPeak")
peaks_gr <- keepStandardChromosomes(peaks_gr, pruning.mode = "coarse")
if (length(peaks_gr) > 1000) {
  peaks_gr <- peaks_gr[order(peaks_gr$score, decreasing = TRUE)[1:1000]]
}
peaks_gr <- GRanges(
  seqnames = seqnames(peaks_gr),
  ranges   = IRanges(
    start = start(peaks_gr) + peaks_gr$peak - 99,
    end   = start(peaks_gr) + peaks_gr$peak + 100
  )
)
peaks_gr <- keepStandardChromosomes(peaks_gr, pruning.mode = "coarse")

# Foreground sequences
seqs <- getSeq(BSgenome.Hsapiens.UCSC.hg38, peaks_gr)

# Background: dinucleotide-shuffled sequences
set.seed(42)
shuf_seqs <- universalmotif::shuffle_sequences(seqs, k = 2)

# Motif matches
fg_hits <- motifMatches(matchMotifs(pwm_list, seqs,      genome = "hg38", out = "matches", p.cutoff = 5e-04))
bg_hits <- motifMatches(matchMotifs(pwm_list, shuf_seqs, genome = "hg38", out = "matches", p.cutoff = 5e-04))

n_fg <- nrow(fg_hits)
n_bg <- nrow(bg_hits)

# Fisher's test per motif
sea_results <- data.frame(
  cellline = cellline,
  tf       = tf,
  motif    = colnames(fg_hits),
  fg_count = colSums(fg_hits),
  bg_count = colSums(bg_hits),
  fraction = colSums(fg_hits) / n_fg,
  stringsAsFactors = FALSE
)

sea_results$pval <- mapply(function(a, c) {
  fisher.test(matrix(c(a, n_fg - a, c, n_bg - c), nrow = 2),
              alternative = "greater")$p.value
}, sea_results$fg_count, sea_results$bg_count)

sea_results$padj       <- p.adjust(sea_results$pval, "BH")
sea_results$enrichment <- (sea_results$fg_count / n_fg) / (sea_results$bg_count / n_bg)
sea_results            <- sea_results[order(sea_results$pval), ]

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_parquet(sea_results, out_path)
