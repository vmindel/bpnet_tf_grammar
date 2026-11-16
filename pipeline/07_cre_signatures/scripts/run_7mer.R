library(nanoparquet)
library(rtracklayer)
library(GenomicRanges)
library(IRanges)
library(stringr)
library(dplyr)
library(BSgenome.Hsapiens.UCSC.hg38)

peak_path <- snakemake@input[["peaks"]]
out_path  <- snakemake@output[[1]]

# Parse cell line and TF from path (peaks_all/{cell_line}/{tf}.bed)
parts    <- str_split(peak_path, "/")[[1]]
cellline <- parts[length(parts) - 1]
tf       <- str_remove(parts[length(parts)], "\\.bed$")

# Load peak file and center on summit (200 bp window)
loaded_peak <- import(peak_path, format = "narrowPeak")
loaded_peak <- keepStandardChromosomes(loaded_peak, pruning.mode = "coarse")
loaded_peak <- GRanges(
  seqnames = seqnames(loaded_peak),
  ranges   = IRanges(
    start = start(loaded_peak) + loaded_peak$peak - 99,
    end   = start(loaded_peak) + loaded_peak$peak + 100
  )
)

# Slide 7-mer windows across each peak
tf_7mer     <- unlist(slidingWindows(loaded_peak, width = 7, step = 1))
tf7mer_seqs <- getSeq(BSgenome.Hsapiens.UCSC.hg38, tf_7mer)

seqdf <- as.data.frame(tf7mer_seqs) %>%
  group_by(x) %>%
  summarise(c.count = n()) %>%
  mutate(frac = c.count / sum(c.count)) %>%
  arrange(desc(c.count))

seqdf$cellline <- cellline
seqdf$tf       <- tf

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_parquet(as.data.frame(seqdf), out_path)
