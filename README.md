# GBM_analysis

Code accompanying the research project **“Quantitative Spatial Profiling of the Glioblastoma Immune Microenvironment Using Self-Supervised Computational Pathology.”**

The repository contains Python scripts for TCGA-GBM cohort construction, whole-slide image quality control, cellular profiling, spatial tumor–immune analysis, self-supervised DINO training, feature extraction, and model validation.

## Pipeline Overview

| Order | Script | Purpose |
| --- | --- | --- |
| 1 | `scripts/01_cohort/cohort.py` | Build the TCGA GBM cohort. |
| 2 | `scripts/01_cohort/cohort_refinement.py` | Refine cohort and split metadata. |
| 3 | `scripts/02_wsi_qc/wsi_qc.py` | QC slides and generate spatial sampling coordinates. |
| 4 | `scripts/03_cells/cell_sampling.py` | Check the HoVer-Net sampling design. |
| 5 | `scripts/03_cells/cell_profiling.py` | Profile cells across the cohort. |
| 6 | `scripts/04_spatial/spatial_topology.py` | Compute patient spatial topology features. |
| 7 | `scripts/05_ssl/dino_training.py` | Train the GBM-adapted DINO encoder. |
| 8a | `scripts/06_embeddings/vit_embeddings.py` | Extract adapted ViT embeddings. |
| 8b | `scripts/06_embeddings/generic_dino_shard.py` | Extract generic DINO embeddings sequentially or in eight parallel shards. |
| 9 | `scripts/07_analysis/validation_inference.py` | Run the corrected validation and inference analysis. |
| 10 | `scripts/07_analysis/generic_dino_finalizer.py` | Combine all generic DINO patient embeddings and compare with stage 9. |
| 11 | `scripts/08_attention/attention_validation.py` | Validate attention against cell outputs. |
| 12 | `scripts/09_models/foundation_models.py` | Run the UNI and CONCH comparison. |

## Running the Pipeline

Run the scripts in the order listed above. Stage 8a precedes 8b and 9; stage 10 requires stages 8b and 9; stage 11 requires cell profiling and DINO training; and stage 12 requires stages 9 and 10.

The pipeline is designed for Python 3.11 in Google Colab with GPU access and Google Drive mounted at `/content/drive/MyDrive/AACR_GBM_Project`. The scripts retain their original data paths and output structure.

Cell profiling and generic DINO extraction support parallel execution through configurable shard IDs. The scripts in `scripts/maintenance/` are reserved for targeted cell-output recovery and are not part of the standard pipeline.

For additional details, see the [ZIP comparison audit](docs/zip_audit.md).

## Citation

If you use or reference this code, please cite:

A. Wang, “GBM_analysis,” GitHub, 2026. [Online]. Available: https://github.com/aaronwang0809/VT-AGSCL.
