# ZIP comparison audit

Source archive: `Notebooks-20261012T024226Z-1-001.zip` (26 notebooks). This project has 13 routine scripts and 2 targeted repair scripts. The eight Generic DINO shard notebooks share one parameterized script.

## Coverage

| ZIP notebooks | Project coverage |
| --- | --- |
| 01, 01A, 02, 03B, 04, 05, 06, 07B, 08 FIXED, 09 FIXED | Direct conversions into the corresponding numbered stage scripts. |
| 03C Full Cohort Cell Profiling | The later output-free 03C Parallel Full Cohort revision from Downloads is used. It supports a full run or three nonoverlapping shards. |
| Generic DINO FAST SHARD 0–7 | One `generic_dino_shard.py` processes all patients by default or each shard separately; `generic_dino_finalizer.py` combines the outputs. |
| Generic DINO ALL 8 SHARDS RECOVERY FIXED | The consolidated extraction script skips valid completed patients and now resumes partial GDC downloads. Its failure log uses a different filename. |
| Generic DINO FAST FINALIZER FIXED | The later self-contained finalizer from Downloads produces the same main embedding matrix, result, prediction, comparison, bootstrap, and record artifacts. |
| 03C Finalize TCGA-02-0099 FIXED | The later FIXED_v2 revision from Downloads is in `scripts/maintenance/cell_recovery.py`. A later two-patient repair is also available there. |
| 03 Cell Nuclei Pipeline | Its three-patient pilot role is superseded by 03B sampling and 03C full-cohort profiling. Its old `notebook03_design_report.json` is not emitted. |
| 07 Topology Prediction and Disparities Audit | 07B recalculates these analyses with corrected endpoint and clinical handling. It writes corrected output names rather than the original 07 filenames. |
| 08 FAST DIAGNOSTIC RESUMABLE | The selected 08 FIXED script performs the attention analysis and reuses completed patient CSVs. It stops on an error and can be rerun; it does not emit the diagnostic notebook's separate failed-patient CSV. |

The scripts are functionally aimed at the final corrected workflow. They do not reproduce every intermediate filename or diagnostic artifact from all 26 notebooks.

## Checks performed

- Every function and class in each selected notebook revision appears in its converted script. Function-level changes consist of removed progress prints and docstrings, with the documented shard, download, and completion changes.
- All selected notebook artifact filename strings are present in the corresponding scripts. The cell-profiling script adds per-shard completion audits.
- The main WSI QC, cell sampling, DINO training, ViT extraction, attention, and foundation-model processing loops retain their computation and output writes; differences inspected in those loops were progress messages.
- All 15 scripts compile with Python 3.11. The README stage paths exist.
- Isolated synthetic checks confirmed Generic DINO completion detection for absent, valid, damaged-status, and damaged-embedding cases, plus a partial-download retry and final file replacement.

## Runtime limit

The full pipeline has not been run against the mounted Colab Drive, TCGA slides, GPU models, or Hugging Face gated assets on this Windows computer. These checks support code and handoff equivalence, but they cannot establish that every stage completes or produces numerically identical results in Colab. Run the stages in README order and inspect their generated outputs before treating the research results as verified.