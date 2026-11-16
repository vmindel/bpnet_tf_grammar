# Pearson correlation between experiments of the maximum fold change over every cCRE (untransformed).
# Columns = experiments ({cell_line}_{gene_symbol}), in metadata order; the row order is the same.
suppressPackageStartupMessages(library(nanoparquet))
rds_files <- unlist(snakemake@input)
mat <- sapply(rds_files, readRDS)                 # cCREs x experiments
colnames(mat) <- snakemake@params[["names"]]
corrs <- as.data.frame(cor(mat))
write_parquet(corrs, snakemake@output[[1]])
message("cCRE signal correlations: ", nrow(mat), " cCREs x ", ncol(mat), " experiments")
