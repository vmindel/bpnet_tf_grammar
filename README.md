# BPNet grammar — code for the paper's figures

We use ENCODE's published BPNet models (5 folds each, TF ChIP-seq, 2,282 models in
total) with their TF-MoDISco motifs and Fi-NeMo / ENCODE motif instances, and ask:

- how well the models separate peaks from background with and without their input-control track, and how much
  information the motifs they learned carry (Fig. 1);
- whether a TF's own ("direct") motif sites act additively or synergistically within a peak (Fig. 2);
- whether the direct motifs and the other motifs of a peak act additively or synergistically (Fig. 3);
- whether members of a TF family bind differently because of their own motif or because of the other motifs
  around it — KLF/SP vs FOX in HepG2 (Fig. 4).

All model predictions are **predicted log counts with an all-zero control track**
(`CountWrapper(ControlWrapper(BasePairNet))`, bpnet-lite), averaged over the 5 folds, unless stated otherwise.

## Repository layout

```
notebooks/Fig1-4.ipynb    one notebook per figure; each writes its panels to figures/
modules/                  code shared by the notebooks and the pipeline
  paths.py                every location (data root, genome, repo folders), overridable by env variables
  bpnet_sets.py           the two model sets (PAPER / ALL_BPNET) and LSF array slicing
  curves.py               threshold-curve definitions (Figs 2, 3) and their plots
  peaks.py                example-peak helpers: models, motif hits, DeepLIFT/SHAP logos, knockout bar plots
  plot_setup.py           editable PDF text, figures/ folder
pipeline/                 the scripts that made every table the figures read, in run order (00 -> 06)
  env.sh                  source before submitting jobs (python, data root, genome, logs)
  NN_step/run.sh          the LSF (bsub) commands of that step, as run for the paper
metadata/                 ENCODE metadata tables, the PAPER selection, family accession tables, PWMs
data/                     small cached tables behind the figures (~40 MB), committed so the figures can be redrawn
  examples/               pipeline outputs of the example experiment (FOXA1 HepG2 ENCSR000BMO) used in Figs 2, 3
environment.yml           conda/pip environment; environment/ has the exact package lists
figures/                  output (not tracked)
```

## Redrawing the figures

```bash
conda env create -f environment.yml && conda activate bpnet_grammar
python -m ipykernel install --user --name bpnet_grammar
export BPNET_DATA=/path/to/data_root      # ENCODE downloads (pipeline step 00); see modules/paths.py
export MISHA_HG38=/path/to/misha/hg38     # misha genome database (pymisha)
jupyter lab notebooks/
```

| Notebook | Reads from `data/` | Also needs from `$BPNET_DATA` | GPU |
|---|---|---|---|
| Fig1 | `auroc_summary.parquet`, `ic_motifs_data.parquet` | BHLHE40 K562 motifs, peaks, Fi-NeMo instances | no |
| Fig2 | `examples/HepG2_FOXA1_ENCSR000BMO/`, `self_epistasis_merged_direct_calc_wt.parquet` | FOXA1 ENCSR000BMO models + bigBed | logo cells |
| Fig3 | `examples/HepG2_FOXA1_ENCSR000BMO/`, `epi_curves_pos.parquet`, `effect_curves_{direct,others}.parquet` | FOXA1 ENCSR000BMO models + bigBed | logo cells |
| Fig4 | `{klf,fox}_obs.parquet`, `{klf,fox}_pred.parquet`, `{klf,fox}_pairwise_corr_famdirect.csv` | – | no |

The panels that only plot cached tables (all of Fig. 4, the curves of Figs 2–3, Fig. 1A and D–G) run without the
data root. The example-peak logos recompute DeepLIFT/SHAP attributions and need the ENCODE model files and a GPU.

## Rebuilding everything: the pipeline

Written for the WEXAC LSF cluster (`bsub`); every step's `run.sh` lists the exact commands. All scripts skip outputs
that already exist, so an interrupted array is resumed by resubmitting it. Per-experiment outputs are written next to
the ENCODE files in `$BPNET_DATA/{PAPER,ALL_BPNET}/{cell}/{experiment}/` (the ENCODE downloads alone are several hundred GB; not in this repo).

```bash
source pipeline/env.sh
bash pipeline/00_download/run.sh     # then 01 ... 06, each after the previous one finished
```

| Step | What | Output | Figure | Compute |
|---|---|---|---|---|
| `00_download` | ENCODE metadata of the 440-experiment PAPER selection (`metadata/big_subset_meta.orig.csv`) and of the 1,842 other ENCODE BPNet models; downloads models, test regions, peaks, controls, TF-MoDISco motifs, Fi-NeMo instances, bigBed motif annotations | `metadata/*.csv`, `$BPNET_DATA/{PAPER,ALL_BPNET}/` | all | CPU, network |
| `01_auroc` | Test-set counts AUROC, peaks vs GC-matched nonpeaks, ENCODE's test folds: with the model's own control (reproduces ENCODE's value) and with no control | `data/auroc_summary.parquet` | 1A | GPU arrays |
| `02_motif_core_ic` | Information content of the central 20 bp of every TF-MoDISco motif, and the fraction of peaks with a Fi-NeMo hit (λ 0.7, counts head) | `data/ic_motifs_data.parquet` | 1D–G | CPU arrays |
| `03_epistasis` | Shuffle-donor knockouts on every peak: direct motifs and/or other motifs replaced by dinucleotide-shuffle bases (20 shuffles). First with Fi-NeMo hits, then with ENCODE's bigBed annotation (direct = `category_type_simple == 'direct'`), reusing the first pass's `wt` | `motif_epistasis[_direct]_{tag}.parquet` | 2, 3 | GPU arrays |
| `04_direct_sites` | Every direct site knocked out alone and all together, in the native peak and in a shuffled peak that keeps only the direct sites; then again with overlapping direct sites merged | `direct[_merged]_sites_{sites,peaks}_{tag}.parquet` | 2 | GPU arrays |
| `05_curves` | Per experiment, fraction of peaks passing each threshold (interaction, single-set effects, direct-site synergy) | `data/epi_curves_pos.parquet`, `data/effect_curves_*.parquet`, `data/self_epistasis_merged_direct_calc_wt.parquet` | 2, 3 | CPU |
| `06_family_ko` | KLF/SP and FOX models in HepG2: observed and predicted counts over the union of their peaks, knockouts of the family's own motif instances / the others / all, pairwise correlations between members | `data/{klf,fox}_*` | 4 | GPU, ~2–4 h per family |

Model sets: **PAPER** = 440 experiments with a BPNet model in six cell lines (A549, GM12878, H1, HEK293, HepG2, K562);
**ALL_BPNET** = 1,842 further ENCODE BPNet experiments (`modules/bpnet_sets.py`, env `BPNET_SET=paper|all`).
Steps 03 (bigBed pass), 04 and 05 cover the 2,073 experiments whose ENCODE bigBed has the direct/indirect annotation
(162 archived models have no bigBed and 47 have an old-format one). The epistasis and effect curves use the 1,480 of
them with at least one peak containing both direct and other motifs; the direct-site synergy curves use the 1,506
with at least one peak with two or more direct sites.

### Definitions

- **direct motif**: an ENCODE motif instance with `category_type_simple == 'direct'` (counts head, positive patterns).
  "p0" in column names means direct (it meant TF-MoDISco pattern_0 in the Fi-NeMo pass).
- **interaction** (Fig. 3) = `wt − (p0_only + non_only − none)`: `none` = direct and other motif bases from the shuffle,
  `p0_only` = only the direct motifs kept, `non_only` = only the other motifs kept. > 0 synergy, < 0 redundancy.
  Additive in log counts = multiplicative in counts.
- **direct-site synergy** (Fig. 2) = `Σᵢ (ref − KOᵢ) − (ref − KOALL)`, over peaks with ≥ 2 direct sites; direct
  sites sharing ≥ 1 base are merged into one site first. > 0 cooperative, < 0 redundant.

## Provenance notes

- The pipeline scripts are the ones that produced the paper's data. In the repository they were changed only to read
  locations from `modules/paths.py` and write task lists / validation tables next to themselves. Two pieces that
  were notebook cells in the original analysis became scripts: `05_curves/build_curves.py` and
  `06_family_ko/family_ko.py` (from the `klf_hepg2` / `fox_hepg2` notebooks; same steps, constants and seeds). The
  table builders `01_auroc/build_auroc_table.py` and `02_motif_core_ic/build_ic_table.py` are new; their output
  equals the tables the figures were made from.
- The pipeline steps were run in the lab's `seqtfun` environment (`environment/seqtfun_original_runs.yml`;
  tangermeme 1.5.0), the figure notebooks in `seqtfun031` (`environment.yml`; tangermeme 1.6.0), which also runs
  the pipeline. `pipeline/env.sh` points at `seqtfun031`.
- The family folders of Fig. 4 (`klf_sp_hepg2/`, `fox_hepg2/`) are named `{TF}_HepG2_{BPNet annotation accession}`,
  the accession of ENCODE's BPNet model annotation, not of the ChIP-seq experiment (see
  `metadata/{klf_sp_hepg2,fox_hepg2}_bpnet_accessions.tsv`).
