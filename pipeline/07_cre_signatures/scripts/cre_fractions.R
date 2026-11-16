# Genomic annotation of every peak set (Fig. S2): each peak (summit +- 100 bp) annotated with ChIPseeker
# (UCSC knownGene, promoter = TSS -2000/+200 bp); promoters overlapping a CpG island are 'CGI Promoter'.
# Output: one row per experiment and annotation class with the fraction of its peaks, plus |distance to TSS| summaries.
# Usage (from the step's work folder): Rscript cre_fractions.R <metadata.csv> <cCREs.bed> <cpgIslandExt.txt.gz> <out.parquet>
suppressPackageStartupMessages({
  library(GenomicRanges)
  library(rtracklayer)
  library(ChIPseeker)
  library(TxDb.Hsapiens.UCSC.hg38.knownGene)
  library(org.Hs.eg.db)
  library(BSgenome.Hsapiens.UCSC.hg38)
  library(dplyr)
  library(stringr)
  library(nanoparquet)
})
args <- commandArgs(trailingOnly = TRUE)
meta_path <- args[1]; cre_path <- args[2]; cpg_path <- args[3]; out_path <- args[4]
txdb <- TxDb.Hsapiens.UCSC.hg38.knownGene

cres <- makeGRangesFromDataFrame(read.table(cre_path, sep = "\t"), seqnames.field = "V1", start.field = "V2",
                                 end.field = "V3", keep.extra.columns = TRUE)
cpg.track <- makeGRangesFromDataFrame(read.table(cpg_path), seqnames.field = "V2", start.field = "V3", end.field = "V4")
cpg.track <- keepStandardChromosomes(cpg.track, pruning.mode = "coarse")

peak_dirs <- list.dirs("peaks_all")[-1]                                  # one folder per cell line
df <- data.frame(peaks.self = unlist(lapply(peak_dirs, list.files, pattern = "*.bed", full.names = TRUE)))
df$cell_line <- str_split(df$peaks.self, "/", simplify = TRUE)[, 2]
df$tf_name <- str_replace(str_split(df$peaks.self, "/", simplify = TRUE)[, 3], ".bed", "")
comp.df <- merge(df, read.csv(meta_path))                                # by cell_line and tf_name

process_annotations <- function(vec, cgi.vec) {
  p <- grepl("Promoter", vec)
  p.cgi <- p & cgi.vec
  i <- grepl("Intron", vec)
  i.first <- grepl("intron 1", vec)
  e <- grepl("Exon", vec)
  d <- grepl("Distal Intergenic", vec)
  vec[p] <- "Other Promoter"
  vec[p.cgi] <- "CGI Promoter"
  vec[i] <- "Other Intronic"
  vec[e] <- "Exonic"
  vec[d] <- "Distal"
  vec[i.first] <- "1st Intron"
  vec
}

proc_peak <- function(peak.path, cres, cpgs, txdb) {
  pf <- import(peak.path, format = "narrowPeak")
  pf <- keepStandardChromosomes(pf, pruning.mode = "coarse")
  summit.pos <- start(pf) + pf$peak
  start(pf) <- summit.pos - 100
  end(pf) <- summit.pos + 100
  annots <- as.GRanges(annotatePeak(pf, tssRegion = c(-2000, 200), TxDb = txdb))
  result <- as.data.frame(annots)
  result$peak_index <- seq_len(nrow(result))
  result$cpg <- overlapsAny(annots, cpgs)
  # cCRE class of the peak: the cCRE it overlaps most (>= 100 bp), else 'none' (not used in the figure)
  overlaps <- findOverlaps(annots, cres, minoverlap = 100)
  peaks_with_overlaps <- data.frame(
    peak_index = queryHits(overlaps), cCRE_type = mcols(cres)$V6[subjectHits(overlaps)],
    overlap_width = width(pintersect(annots[queryHits(overlaps)], cres[subjectHits(overlaps)])))
  peak_summary <- peaks_with_overlaps %>% group_by(peak_index) %>%
    summarize(cCRE_types = cCRE_type[which.max(overlap_width)])
  result <- merge(result, peak_summary, by = "peak_index", all.x = TRUE)
  result$cCRE_types[is.na(result$cCRE_types)] <- "none"
  result$genomic_annots <- process_annotations(annots$annotation, result$cpg)
  result$distanceToTSS <- annots$distanceToTSS
  result
}

long_ccre <- lapply(seq_len(nrow(comp.df)), function(i) {
  tryCatch({
    res <- proc_peak(comp.df$peaks.self[i], cres, cpg.track, txdb)
    res %>% dplyr::count(genomic_annots) %>%
      mutate(fraction = n / sum(n), cellline = comp.df$cell_line[i], tf = comp.df$tf_name[i],
             subfamily = comp.df$subfamily[i],
             quantile.low.distTSS = quantile(abs(res$distanceToTSS), .25),
             quantile.high.distTSS = quantile(abs(res$distanceToTSS), .75),
             mean.distTSS = mean(abs(res$distanceToTSS)),
             median.distTSS = median(abs(res$distanceToTSS)),
             quantile.10p.distTSS = quantile(abs(res$distanceToTSS), .1),
             quantile.90p.distTSS = quantile(abs(res$distanceToTSS), .9))
  }, error = function(e) {
    message("Failed: ", comp.df$peaks.self[i], " - ", e$message)
    NULL
  })
}) %>% bind_rows()
write_parquet(long_ccre, out_path, compression = "snappy")
