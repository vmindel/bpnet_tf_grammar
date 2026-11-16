library(dplyr)
library(nanoparquet)

parquet_paths <- unlist(snakemake@input)
out_path      <- snakemake@output[[1]]

all_sea <- bind_rows(lapply(parquet_paths, read_parquet))

dir.create(dirname(out_path), recursive = TRUE, showWarnings = FALSE)
write_parquet(all_sea, out_path)
