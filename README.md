# AACR GBM analysis scripts

Python code extracted from the selected AACR GBM notebooks. The scripts keep the original data paths and stage outputs, but omit notebook results, markdown, checkpoint displays, and exploratory commentary. They are organized by the order in which their outputs are needed.

## Layout and run order

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

Stage 8b checks the adapted embedding matrix from stage 8a, so run 8a first. Stage 9 also needs 8a; stage 10 needs stages 8b and 9. Stage 11 needs cell profiling and DINO training. Stage 12 needs stages 9 and 10. The two scripts in `scripts/maintenance/` are targeted cell-output repairs and are not part of a routine full run.

## Parallel work

The eight `Generic_DINO_FAST_SHARD_0_of_8` through `7_of_8` notebooks had the same code except for the shard number. They are represented by one script. By default, one run processes all train/test patients sequentially. For parallel execution, run it in eight separate Colab runtimes with `AACR_GBM_SHARD_ID` set to each integer from `0` through `7`; each runtime writes patient-specific files and its own failure list. After all eight finish, run `generic_dino_finalizer.py` once. It checks that all 320 train/test patient embeddings are present before creating the combined matrix and comparison tables.

The selected 03C cell-profiling notebook is also one script. By default it processes the whole cohort. For three Colab runtimes, set `AACR_GBM_CELL_SHARDS=3` in each and `AACR_GBM_CELL_SHARD_ID` to `0`, `1`, or `2`. The sample manifest is written through a temporary file before replacement. An incomplete shard writes its own audit; the shared patient feature table is published only when the full processable cohort is complete. After all runtimes finish, run the script once with default shard settings to refresh and verify the full-cohort outputs, especially if any patient required a retry.

`foundation_models.py` retains its source notebook's single-run setting. Its extraction and full-cohort analysis are in one script, so it should be run once rather than launched as parallel shards.

## Selected notebook versions

| Script | Source notebook |
| --- | --- |
| `cohort.py` | `AACR_GBM_Notebook_01_TCGA_Cohort.ipynb` |
| `cohort_refinement.py` | `AACR_GBM_Notebook_01A_Cohort_Refinement.ipynb` |
| `wsi_qc.py` | `AACR_GBM_Notebook_02_WSI_QC_and_Spatial_Sampling.ipynb` |
| `cell_sampling.py` | `AACR_GBM_Notebook_03B_HoVerNet_Sampling_Convergence.ipynb` |
| `cell_profiling.py` | `AACR_GBM_Notebook_03C_Parallel_Full_Cohort_Cell_Profiling.ipynb` |
| `spatial_topology.py` | `AACR_GBM_Notebook_04_Spatial_Tumor_Immune_Topology.ipynb` |
| `dino_training.py` | `AACR_GBM_Notebook_05_GBM_SSL_DINO_ViT.ipynb` |
| `vit_embeddings.py` | `AACR_GBM_Notebook_06_Frozen_ViT_Embedding_Extraction.ipynb` |
| `generic_dino_shard.py` | `AACR_GBM_Generic_DINO_FAST_SHARD_0_of_8.ipynb` through `7_of_8` |
| `validation_inference.py` | `AACR_GBM_Notebook_07B_Validation_Fixes_and_Inference.ipynb` |
| `generic_dino_finalizer.py` | `AACR_GBM_Generic_DINO_FINALIZER_SELF_CONTAINED.ipynb` |
| `attention_validation.py` | `AACR_GBM_Notebook_08_Quantitative_ViT_Attention_Validation_FIXED.ipynb` |
| `foundation_models.py` | `AACR_GBM_Notebook_09_UNI_CONCH_FIXED_OFFICIAL_LOADERS.ipynb` |
| `cell_recovery.py` | `AACR_GBM_Notebook_03C_Finalize_TCGA_02_0099_FIXED_v2.ipynb` |
| `cell_rebuild.py` | `AACR_GBM_Notebook_03C_REDO_TWO_PATIENTS_FIXED.ipynb` |

Numbered download copies such as `(1)` and `(2)` were excluded because they contain saved results. The corrected 07B analysis replaces 07, and the FIXED 08 attention notebook was chosen over the interrupted diagnostic run. `cell_rebuild.py` deliberately deletes and rebuilds outputs for its two named patients.

## Environment and verification

Run the scripts in Linux Colab with Drive mounted at `/content/drive/MyDrive/AACR_GBM_Project`, the project data and model weights, and a suitable GPU. The scripts retain top-level execution and may download slides, train models, and write project outputs. `wsi_qc.py` defaults to the full cohort; `AACR_GBM_MAX_PATIENTS=3` limits it to a pilot. Stage 12 requires Hugging Face access to UNI and CONCH.

All 15 scripts compile with Python 3.11. Static checks found no notebook magics or unresolved globals, and six data-independent helper checks passed. Filename handoffs and dependencies were cross-checked. The full pipeline has not been run in this Windows environment, which lacks the mounted Drive, project data, GPU, and model assets; end-to-end runtime behavior is unverified.

For source coverage and differences from older ZIP notebooks, see [ZIP comparison audit](docs/zip_audit.md).
