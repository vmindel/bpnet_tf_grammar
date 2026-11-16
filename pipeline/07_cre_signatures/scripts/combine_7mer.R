library(dplyr)
library(nanoparquet)

parquet_paths <- unlist(snakemake@input)
out_path      <- snakemake@output[[1]]

all_7mer <- bind_rows(lapply(parquet_paths, read_parquet))

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_parquet(all_7mer, out_path)
