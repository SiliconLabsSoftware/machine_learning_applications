# Verification Docs README (`verification/`)

## Purpose
Contains the complete evidence package required for rigorous reproducibility, split logic verification, and absolute proof of zero data leakage.

## Priority Proof Files (in `proofs/`)
Reviewers and auditors should evaluate the documents in the following order:

1. **`TRAIN_TEST_VAL_SPLIT_EXPLANATION.txt`**
   - The conceptual rationale and split methodology used across datasets. Explains the difference between feature-level splitting and strict file-level isolation.

2. **`DATA_LEAKAGE_ANALYSIS.txt`**
   - Formal evaluation of leakage risks (e.g., augmentation cross-contamination) and the verdict confirming absolute separation.

3. **`DOC3_LINE_BY_LINE_PROOF_NO_LEAKAGE.txt`**
   - Traceability audit [code-to-claim mapping]. Line-by-line proof connecting the Python pipeline code to the leakage analysis claims.

4. **`DOC1_MODEL1_COMPLETE_LABELED_DATA.txt`**
   - Exact mapping of Model 1 labeled datasets mapping every acoustic feature back to its source category.

5. **`DOC2_MODEL2_COMPLETE_LABELED_DATA.txt`**
   - Exact mapping of Model 2 labeled datasets mapping.

6. **`MODEL1_COMPLETE_FILENAMES.txt` / `MODEL2_COMPLETE_FILENAMES.txt`**
   - The exhaustive registry of every WAV file processed.

## Performance Validation (in `plots/`)
These plots visually prove the statistical viability and noise-rejection robustness of the final deployed model.

- `confusion_matrix_m1.png`: True positive separation of Cry vs. Background.
- `confusion_matrix_m2.png`: True positive separation of Sad vs. Laugh emotional states.
- `fp_audit.png`: False Positive mapping showing confidence density on negative samples.
- `accuracy_vs_noise_IoT.png` / `confusion_matrix_15db_IoT.png`: Proof of real-world functionality—benchmarking model accuracy degradation across specific dB Signal-to-Noise thresholds mimicking real home environments.
