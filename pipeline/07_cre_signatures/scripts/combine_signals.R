rds_files <- unlist(snakemake@input)
exp_names  <- snakemake@params[["names"]]
out_path   <- snakemake@output[[1]]

signals <- lapply(rds_files, readRDS)
names(signals) <- exp_names

# Each element is a numeric vector (one value per peak).
# Bind into a matrix: rows = peaks, columns = experiments.
mat <- do.call(cbind, signals)

saveRDS(mat, out_path)
message("Signal matrix: ", nrow(mat), " peaks × ", ncol(mat), " experiments → ", out_path)
