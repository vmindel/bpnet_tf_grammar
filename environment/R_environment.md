# R environment (pipeline/07_cre_signatures, notebooks/FigS2_cre_correlation.R)

The R parts of the analysis need R 4.4 with these Bioconductor / CRAN packages. The versions are those of the
lab's `genomics_r` conda environment, used to run the repository's checks; the figure tables were first made in an
interactive R session with the same packages.

| package | version | used by |
|---|---|---|
| R | 4.4.3 | |
| GenomicRanges | 1.58.0 | all |
| IRanges | 2.40.1 | all |
| GenomeInfoDb | 1.42.3 | all |
| rtracklayer | 1.66.0 | bigWig / BED import |
| Biostrings | 2.74.1 | sequences |
| BSgenome.Hsapiens.UCSC.hg38 | 1.4.5 | sequences |
| TFBSTools | 1.44.0 | motif enrichment |
| motifmatchr | 1.30.0 | motif enrichment |
| universalmotif | 1.24.0 | motifs, dinucleotide shuffles, logos |
| ChIPseeker | 1.42.0 | genomic annotation |
| TxDb.Hsapiens.UCSC.hg38.knownGene | 3.20.0 | genomic annotation |
| org.Hs.eg.db | 3.20.0 | genomic annotation |
| nanoparquet | 0.5.1 | parquet I/O |
| dplyr | 1.2.1 | |
| tidyr | 1.3.2 | |
| stringr | 1.6.0 | |
| ggplot2 | 4.0.3 | figure |
| cowplot | 1.2.0 | figure |
| patchwork | 1.3.2 | figure |
| paletteer | 1.7.0 | figure |

The Snakemake workflows (`Snakefile_signal`, `Snakefile_motifs`) were run with Snakemake and the
`cluster-generic` executor plugin (1.0.3) submitting to LSF.
