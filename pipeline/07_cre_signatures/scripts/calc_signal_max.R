suppressPackageStartupMessages({
    library(rtracklayer)
    library(GenomicRanges)
    library(IRanges)
    library(GenomeInfoDb)
})

load_bigwig_sumoverregion <- function(bw_path, peakgrange) {
    bw_sig <- import(bw_path, which = peakgrange, as = "RleList")

    # Peaks on chromosomes absent from the bigwig get 0
    result      <- numeric(length(peakgrange))
    common_seqs <- intersect(seqlevels(peakgrange), names(bw_sig))

    if (length(common_seqs) == 0) return(result)

    peak_mask <- as.logical(seqnames(peakgrange) %in% common_seqs)
    sub_peaks <- keepSeqlevels(peakgrange[peak_mask], common_seqs, pruning.mode = "coarse")
    
    views <- RleViewsList(
        rleList    = bw_sig[seqlevels(sub_peaks)],
        rangesList = split(ranges(sub_peaks), seqnames(sub_peaks))
    )
    peak_sums <- viewMaxs(views)          # maximum fold change over the cCRE
    peak_sums_ordered <- unsplit(as.list(peak_sums),
                                 as.factor(seqnames(sub_peaks)))

    result[peak_mask] <- unlist(peak_sums_ordered)
    return(result)
}

bw_path    <- snakemake@input[["bw"]]
peaks_path <- snakemake@params[["peak_set"]]
out_path   <- snakemake@output[[1]]

# read.table avoids rtracklayer's strict BED type checks, which fail when
# non-numeric accession IDs appear in the score column (e.g. cCREs BED)
peaks_df   <- read.table(peaks_path, sep = "\t", header = FALSE)
peakgrange <- makeGRangesFromDataFrame(peaks_df,
                                       seqnames.field = "V1",
                                       start.field    = "V2",
                                       end.field      = "V3",
                                       ignore.strand  = TRUE)

signal <- load_bigwig_sumoverregion(bw_path, peakgrange)
saveRDS(signal, out_path)

message("Done: ", snakemake@wildcards$cell_line, "/",
        snakemake@wildcards$gene_symbol,
        "  (", length(signal), " peaks)")
