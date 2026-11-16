# Figure S2: what makes TF binding similar between experiments - ChIP-seq signal over ENCODE cCREs, 7-mer content,
# motif enrichment and genomic annotation, by HOCOMOCO v14 subfamily (HepG2 by default).
# Inputs (pipeline/07_cre_signatures): data/cre_signatures/*, metadata/big_subset_meta.orig.csv, and the ENCODE cCREs
# ($BPNET_DATA/cre_signatures/external/GRCh38-cCREs.bed, env CCRES to override) for the 7-mer background.
# Run from the repository root:  Rscript notebooks/FigS2_cre_correlation.R [--cell-line HepG2] [--output-dir figures]
suppressPackageStartupMessages({
  library(GenomicRanges)
  library(rtracklayer)
  library(BSgenome.Hsapiens.UCSC.hg38)
  library(stringr)
  library(dplyr)
  library(tidyr)
  library(nanoparquet)
  library(ggplot2)
  library(cowplot)
  library(patchwork)
  library(universalmotif)
  library(paletteer)
})

# options: --cell-line HepG2 (comma-separated, or 'all'), --output-dir figures, --prefix cre_corr, --width 8, --height 20
opt <- list("cell-line" = "HepG2", "output-dir" = "figures", prefix = "cre_corr", width = "8", height = "20")
args <- commandArgs(trailingOnly = TRUE)
for (k in seq(1, length(args), by = 2)[seq_len(length(args) %/% 2)]) {
  key <- sub("^--", "", args[k]); stopifnot(key %in% names(opt)); opt[[key]] <- args[k + 1]
}
cell_lines  <- trimws(strsplit(opt[["cell-line"]], ",")[[1]])
output_dir  <- opt[["output-dir"]]
prefix      <- opt[["prefix"]]
plot_width  <- as.numeric(opt[["width"]])
plot_height <- as.numeric(opt[["height"]])

D <- "data/cre_signatures"
CCRES <- Sys.getenv("CCRES", file.path(Sys.getenv("BPNET_DATA", "/home/labs/barkailab/vovam/encodedata"),
                                       "cre_signatures/external/GRCh38-cCREs.bed"))
stopifnot(dir.exists(D), file.exists(CCRES))
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)
genome <- BSgenome.Hsapiens.UCSC.hg38

# ── Load data ─────────────────────────────────────────────────────────────────
message("Loading ChIP-seq correlations...")
corrs     <- read_parquet(file.path(D, "big_cre_matrix_correlations.parquet"))
rownames(corrs) <- colnames(corrs)
meta.data <- read.csv("metadata/big_subset_meta.orig.csv")

message("Loading 7-mer results and computing CRE background...")
k7.mers       <- read_parquet(file.path(D, "7mer_results.parquet"))
k7.mers$cell.tf <- paste0(k7.mers$cellline, "_", k7.mers$tf)

screen <- read.csv(CCRES, sep = "\t", header = FALSE)
cres   <- makeGRangesFromDataFrame(screen,
                                   seqnames.field    = "V1",
                                   start.field       = "V2",
                                   end.field         = "V3",
                                   keep.extra.columns = TRUE)
remove(screen)

count_7mers <- function(granges, genome, sample_size = 5000, n_iter = 50, seed = 42) {
  set.seed(seed)
  all_counts <- lapply(seq_len(n_iter), function(i) {
    idx  <- sample(length(granges), min(sample_size, length(granges)))
    seqs <- getSeq(genome, granges[idx])
    oligonucleotideFrequency(seqs, width = 7, step = 1, as.prob = FALSE) |> colSums()
  })
  mean_counts <- Reduce("+", all_counts) / n_iter
  mean_counts / sum(mean_counts)
}

cre_freq   <- count_7mers(cres, genome)
remove(cres)
gc()

seqdf_cres      <- as.data.frame(cre_freq)
seqdf_cres$x    <- rownames(seqdf_cres)

k7.mers <- k7.mers %>%
  left_join(seqdf_cres, by = "x") %>%
  mutate(log2enr = case_when(
    cre_freq == 0 ~ NA_real_,
    frac == 0     ~ -Inf,
    TRUE          ~ log2(frac / cre_freq)
  ))

k7.mers <- k7.mers %>%
  pivot_wider(names_from  = cell.tf,
              values_from = log2enr,
              id_cols     = x,
              values_fill = NA_real_)

corrs.7mers <- as.data.frame(cor(k7.mers[2:ncol(k7.mers)], use = "pairwise.complete.obs"))
corrs.7mers$cells_tf <- colnames(corrs.7mers)
remove(k7.mers)
gc()

# ── Build long_df ─────────────────────────────────────────────────────────────
message("Building combined long data frame...")
corrs$cells_tf     <- colnames(corrs)
meta.data$cells_tf <- paste0(meta.data$cell_line, "_", meta.data$tf_name)

cell_pattern <- "^(A549|GM12878|H1|HEK293|HepG2|K562)_"

long_corrs <- corrs %>%
  pivot_longer(cols      = matches(cell_pattern),
               names_to  = "cells_tf2",
               values_to = "correlation")

long_7mers <- corrs.7mers %>%
  pivot_longer(cols      = matches(cell_pattern),
               names_to  = "cells_tf2",
               values_to = "correlation_7mer")

tf_meta <- meta.data %>%
  distinct(cells_tf, subfamily, tf_name) %>%
  rename(cells_tf2     = cells_tf,
         subfamily.tf2 = subfamily,
         tf2           = tf_name)

long_df <- long_corrs %>%
  left_join(long_7mers, by = c("cells_tf", "cells_tf2")) %>%
  left_join(meta.data,  by = "cells_tf") %>%
  left_join(tf_meta,    by = "cells_tf2") %>%
  mutate(
    tf1          = tf_name,
    cellline.tf1 = cell_line,
    cellline.tf2 = str_extract(cells_tf2, "^[^_]+"),
    subfamily.tf1 = subfamily
  ) %>%
  select(tf1, tf2, cellline.tf1, cellline.tf2,
         subfamily.tf1, subfamily.tf2,
         correlation, correlation_7mer)

long_df$intra.family <- long_df$subfamily.tf1 == long_df$subfamily.tf2
long_df$is.self      <- long_df$tf1 == long_df$tf2 & long_df$cellline.tf1 != long_df$cellline.tf2

# ── Filter to selected cell lines ─────────────────────────────────────────────
if (!"all" %in% tolower(cell_lines)) {
  df_cell <- long_df %>%
    filter(cellline.tf1 %in% cell_lines, cellline.tf2 %in% cell_lines)
} else {
  df_cell <- long_df
}

# Remove exact self-pairs (same TF, same cell line)
df_cell <- df_cell %>% filter(!(tf1 == tf2 & cellline.tf1 == cellline.tf2))
df_cell <- df_cell[order(df_cell$intra.family), ]

# ── Filter to valid families (>2 intra-family pairs) ─────────────────────────
valid_families <- df_cell %>%
  group_by(subfamily.tf1, intra.family) %>%
  summarise(n_self = sum(intra.family), .groups = "drop") %>%
  filter(n_self > 2) %>%
  pull(subfamily.tf1) %>%
  unique()

df_filtered <- df_cell %>%
  filter(subfamily.tf1 %in% valid_families,
         subfamily.tf2 %in% valid_families)

# ── Summarise: df_subfamily ───────────────────────────────────────────────────
df_subfamily <- df_filtered %>%
  group_by(tf1, intra.family) %>%
  summarise(
    med.cre  = mean(correlation,          na.rm = TRUE),
    lo.cre   = quantile(correlation,   0.1, na.rm = TRUE),
    hi.cre   = quantile(correlation,   0.9, na.rm = TRUE),
    med.7mer = mean(correlation_7mer,     na.rm = TRUE),
    lo.7mer  = quantile(correlation_7mer, 0.1, na.rm = TRUE),
    hi.7mer  = quantile(correlation_7mer, 0.9, na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  mutate(intra.family = factor(intra.family, levels = c(TRUE, FALSE)))

df_subfamily <- merge(df_subfamily,
                      long_df[, c("tf1", "subfamily.tf1")],
                      by = "tf1")
df_subfamily <- df_subfamily[!duplicated(df_subfamily), ]

# ── Ordering based on ChIP-seq (applied to both plots) ───────────────────────
tf_order <- df_subfamily %>%
  filter(intra.family == TRUE) %>%
  arrange(med.cre) %>%
  pull(tf1)

subfamily_order <- df_subfamily %>%
  filter(intra.family == TRUE) %>%
  group_by(subfamily.tf1) %>%
  summarise(mean_intra_med = mean(med.cre), .groups = "drop") %>%
  arrange(desc(mean_intra_med)) %>%
  pull(subfamily.tf1)

df_subfamily <- df_subfamily %>%
  mutate(
    tf1           = factor(tf1,           levels = tf_order),
    subfamily.tf1 = factor(subfamily.tf1, levels = subfamily_order)
  )

# ── Summarise: df_crosscell ───────────────────────────────────────────────────
# Cross cell-line: same TF across different cell lines (uses full long_df)
df_crosscell <- long_df %>%
  filter(tf1 %in% tf_order) %>%
  group_by(tf1, is.self) %>%
  summarise(
    med.cre  = mean(correlation,       na.rm = TRUE),
    lo.cre   = quantile(correlation,   0.1, na.rm = TRUE),
    hi.cre   = quantile(correlation,   0.9, na.rm = TRUE),
    med.7mer = mean(correlation_7mer,  na.rm = TRUE),
    lo.7mer  = quantile(correlation_7mer, 0.1, na.rm = TRUE),
    hi.7mer  = quantile(correlation_7mer, 0.9, na.rm = TRUE),
    .groups  = "drop"
  ) %>%
  mutate(
    intra.family = factor(is.self, levels = c(TRUE, FALSE)),
    tf1          = factor(tf1, levels = levels(df_subfamily$tf1))
  ) %>%
  left_join(distinct(df_subfamily[, c("tf1", "subfamily.tf1")]), by = "tf1")

df_crosscell$intra.family <- df_crosscell$is.self

# ── Build df_combined ─────────────────────────────────────────────────────────
df_subplot1 <- df_subfamily %>%
  mutate(intra.family = as.logical(intra.family),
         comparison   = "Intra / Inter family")

df_subplot2 <- df_crosscell %>%
  mutate(intra.family = as.logical(intra.family),
         comparison   = "Cross cell-line")

df_combined <- bind_rows(df_subplot1, df_subplot2) %>%
  mutate(comparison = factor(comparison,
                             levels = c("Intra / Inter family", "Cross cell-line")))

# ── Plot function ─────────────────────────────────────────────────────────────
make_corr_plot <- function(df, med_col, lo_col, hi_col, title) {
  ggplot(df, aes(y = tf1, color = intra.family)) +
    geom_linerange(aes(xmin = .data[[lo_col]], xmax = .data[[hi_col]]),
                   linewidth = 2, alpha = 0.6) +
    geom_point(aes(x = .data[[med_col]]), size = 2) +
    scale_color_manual(
      values = c("FALSE"      = "#756F6F",
                 "TRUE"       = "#3AA154",
                 "cross.cell" = "#3A6CA1"),
      labels = c("FALSE"      = "Inter-family",
                 "TRUE"       = "Intra-family",
                 "cross.cell" = "Cross cell-line")
    ) +
    facet_grid(subfamily.tf1 ~ comparison,
               scales = "free_y",
               space  = "free_y",
               switch = "y") +
    theme_half_open() +
    background_grid(major = "y") +
    labs(x = "Correlation (IQR)", y = NULL, color = NULL, title = title) +
    theme(
      legend.position   = "top",
      axis.text.y       = element_text(size = 8),
      strip.placement   = "outside",
      strip.text.y.left = element_text(angle = 0, hjust = 1, size = 7),
      strip.background  = element_rect(fill = "grey92", color = NA),
      panel.spacing     = unit(0.15, "lines")
    )
}

# ── Save plots ────────────────────────────────────────────────────────────────
message("Saving ChIP-seq plot...")
plot_chip <- make_corr_plot(df_combined,
                             med_col = "med.cre",
                             lo_col  = "lo.cre",
                             hi_col  = "hi.cre",
                             title   = paste("ChIP-seq correlation -",
                                             paste(cell_lines, collapse = ", ")))
ggsave(file.path(output_dir, paste0(prefix, "_chip.pdf")),
       plot = plot_chip, width = plot_width, height = plot_height, units = "in")

message("Saving 7-mer plot...")
plot_7mer <- make_corr_plot(df_combined,
                             med_col = "med.7mer",
                             lo_col  = "lo.7mer",
                             hi_col  = "hi.7mer",
                             title   = paste("7-mer correlation -",
                                             paste(cell_lines, collapse = ", ")))
ggsave(file.path(output_dir, paste0(prefix, "_7mer.pdf")),
       plot = plot_7mer, width = plot_width, height = plot_height, units = "in")

message("Saving motif enrichment plot...")
needed.motifs  <- unique(meta.data[meta.data$subfamily %in% valid_families,
                                   "representative_mot_id"])
subfamily_order <- unique(df_subplot1$subfamily.tf1)

motif_order <- meta.data %>%
  filter(representative_mot_id %in% needed.motifs) %>%
  distinct(representative_mot_id, subfamily) %>%
  mutate(subfamily = factor(subfamily, levels = levels(subfamily_order))) %>%
  arrange(subfamily) %>%
  pull(representative_mot_id)

sea.res <- read_parquet(file.path(D, "sea_top1000_results.parquet")) %>%
  filter(motif    %in% needed.motifs,
         cellline %in% cell_lines,
         tf       %in% levels(df_subplot1$tf1)) %>%
  left_join(df_subplot1 %>% distinct(tf1, subfamily.tf1),
            by = c("tf" = "tf1")) %>%
  mutate(
    motif         = factor(motif,         levels = motif_order),
    tf            = factor(tf,            levels = levels(df_subplot1$tf1)),
    subfamily.tf1 = factor(subfamily.tf1, levels = levels(subfamily_order))
  ) %>%
  arrange(subfamily.tf1, tf)

plot_sea <- ggplot(sea.res, aes(x = motif, y = tf, fill = fraction)) +
  geom_tile() +
  facet_grid(subfamily.tf1 ~ .,
             scales = "free_y",
             space  = "free_y",
             switch = "y") +
  scale_fill_gradient(low = "white", high = "mediumpurple3",
                      limits = c(0, 0.3), oob = scales::squish) +
  theme_half_open() +
  labs(x = "Motif Enrichment", y = NULL, fill = "Fraction",
       title = paste("Motif enrichment -", paste(cell_lines, collapse = ", "))) +
  theme(
    legend.position   = "top",
    axis.text.y       = element_text(size = 8),
    axis.text.x       = element_text(size = 8, angle = 90, hjust = 1),
    strip.placement   = "outside",
    strip.text.y.left = element_text(angle = 0, hjust = 1, size = 7),
    strip.background  = element_rect(fill = "grey92", color = NA),
    panel.spacing     = unit(0.15, "lines")
  )

ggsave(file.path(output_dir, paste0(prefix, "_sea.pdf")),
       plot = plot_sea, width = plot_width, height = plot_height, units = "in")

cre_loaded = read_parquet(file.path(D, "cre_fractions_all_peaks.parquet"))
cre_loaded = cre_loaded%>%
  filter(cellline %in% cell_lines,
         tf %in% unique(df_subplot1$tf1)) %>%
  left_join(df_subplot1 %>% distinct(tf1, subfamily.tf1),
            by = c("tf" = "tf1", "subfamily" = "subfamily.tf1"))%>%
  mutate(
    tf = factor(tf, levels = levels(df_subplot1$tf1)),
    subfamily.tf1 = factor(subfamily, levels = levels(subfamily_order))
  ) %>%
  arrange(genomic_annots) %>%
  arrange(subfamily.tf1, tf)

cres_fracs = cre_loaded%>%
  select(genomic_annots, fraction, tf, subfamily.tf1)


n.order = c( "CGI Promoter", "Other Promoter", '1st Intron',
             'Other Intronic','Exonic', 'Distal', "3' UTR", "5' UTR", "Downstream (<=300bp)")

plot_cre = cres_fracs%>%
  mutate(genomic_annots = factor(genomic_annots, 
                                 levels = n.order)) %>%
  ggplot(aes(x = fraction, y = tf, fill=genomic_annots)) +
  geom_bar(position = position_stack(reverse = TRUE), stat="identity") +
  facet_grid(subfamily.tf1 ~ .,scales = "free_y",
             space  = "free_y",
             switch = "y")+
  scale_fill_paletteer_d("MetBrewer::Cross", direction=-1)+
  theme_half_open() +
  labs(x = "Annotations", y = NULL, color = NULL) +
  theme(
    legend.position   = "top",
    axis.text.y       = element_text(size = 8),
    axis.text.x       = element_text(size = 8, angle = 90),
    strip.placement   = "outside",
    strip.text.y.left = element_text(angle = 0, hjust = 1, size = 7),
    strip.background  = element_rect(fill = "grey92", color = NA),
    panel.spacing     = unit(0.15, "lines")
  )




rep.motifs = readRDS(file.path(D, "representative_motifs_pwms_big_families_trimmed.rds"))
ic_df <- data.frame(
  representative_mot_id = names(rep.motifs),
  mean_ic = sapply(rep.motifs, function(m) m["icscore"] )
)

proc_motifs <- meta.data %>%
  filter(subfamily %in% subfamily_order) %>%
  distinct(subfamily, representative_mot_id) %>%
  left_join(ic_df, by = "representative_mot_id") %>%
  group_by(subfamily) %>%
  slice_max(mean_ic, n = 1) %>%
  mutate(subfamily.tf1 = factor(subfamily, levels = levels(subfamily_order))) %>%
  arrange(subfamily.tf1) 

best_motifs = proc_motifs%>%
  pull(representative_mot_id)



plot_motifs = view_motifs(trim_motifs(rep.motifs[best_motifs]),
                          tryRC = FALSE,min.overlap = .999)+
  facet_wrap(~ motif.id, ncol = 1, strip.position = "left", scales = "free_x") +
  theme_half_open() +
  theme(
    legend.position   = "none",
    axis.text.y       = element_text(size = 8),
    axis.title.y      = element_text(angle = 0, vjust = 0.5, size=8),
    axis.text.x       = element_blank(),
    strip.text.y.left = element_text(angle = 0, hjust = 1, size = 7),
    strip.text        = element_text(hjust = 0, size = 7),
    strip.background  = element_rect(fill = "grey92", color = NA),
    panel.spacing     = unit(0.15, "lines"),
    
  )




sea.res.filtered <- sea.res %>%
  inner_join(proc_motifs %>% dplyr::select(subfamily.tf1, representative_mot_id),
             by = c("subfamily.tf1" = "subfamily.tf1",
                    "motif"         = "representative_mot_id"))%>%
  mutate(log.padj =  -log10(padj))%>%
  mutate(log.padj.clamped = pmin(log.padj, 20))

self.enrich.plot = ggplot(data = sea.res.filtered, aes(x = '-log10padj', y = tf, color = log.padj.clamped, size = log.padj.clamped)) +
  geom_point() +
  facet_grid(subfamily.tf1 ~ .,scales = "free_y",
             space  = "free_y",
             switch = "y") +
  scale_color_gradient(low = "white", high = "#122451FF", limits = c(0, 20), oob = scales::squish) +
  scale_size_continuous(limits = c(0,20), range = c(.5, 4.5)) +
  theme_half_open() +
  labs(x = "Mot Enrichment", y = NULL, color = NULL) +
  theme(
    legend.position   = "top",
    axis.text.y       = element_text(size = 8),
    axis.text.x       = element_text(size = 8, angle = 90),
    strip.placement   = "outside",
    strip.text.y.left = element_text(angle = 0, hjust = 1, size = 7),
    strip.background  = element_rect(fill = "grey92", color = NA),
    panel.spacing     = unit(0.15, "lines")
  )

message("Saving combined plot...")
plot_combined <- plot_motifs + self.enrich.plot + plot_chip + plot_cre +
  plot_layout(widths = c(.4, .4, 1, 0.5))

ggsave(file.path(output_dir, paste0(prefix, "_combined.pdf")),
       plot = plot_combined,
       width = plot_width * 4.7 / 1,   # scale total width by sum of ratios
       height = plot_height, units = "in")

message("Done. Output: ",
        file.path(output_dir, paste0(prefix, "_chip.pdf")), ", ",
        file.path(output_dir, paste0(prefix, "_7mer.pdf")), ", ",
        file.path(output_dir, paste0(prefix, "_sea.pdf")), ", and ",
        file.path(output_dir, paste0(prefix, "_combined.pdf")))
