# Dataset README (`dataset/`)

## Purpose
This folder stores all audio data used to build and verify the final baby-cry pipeline. The data is deliberately gitignored due to its ~2 GB size, but this document provides a complete guide for reproducibility.

## Folder-by-Folder Meaning

### Positive Classes
- `sad/`:
  - Distressed baby vocal samples (pain, hunger, discomfort).
  - Used as CRY (Model 1) and SAD (Model 2).
- `laugh/`:
  - Happy baby vocal samples (cooing, laughing).
  - Used as CRY (Model 1) and LAUGH (Model 2).
- `audio/`:
  - Subset of [CryCeleb2023](https://huggingface.co/datasets/cryceleb/CryCeleb2023) recordings.
  - Used for generic infant vocalisation breadth (`CRYCELEB_CAP` limit [maximum sample extraction bound]).

### Negative/Background Classes
- `esc50/audio/` + `esc50/esc50.csv`:
  - [ESC-50](https://github.com/karolpiczak/ESC-50) environmental sounds.
  - Used for NOT-CRY class. The CSV is the metadata manifest [file-index mapping].
- `noise/`:
  - [DEMAND](https://zenodo.org/record/1227121) noise scenes (e.g., DKITCHEN, DLIVING, NPARK).
  - Used for data augmentation (mixing noise into clean cries) to build real-world robustness.
- `background/`:
  - [Freesound](https://freesound.org) ambient background audio for NOT-CRY coverage.

## Label Scheme

**Model 1 (Stage 1: Cry vs. Background):**
- `0 = CRY` (Includes both `sad/` and `laugh/` audio)
- `1 = NOT-CRY` (Includes environment and silence)

**Model 2 (Stage 2: Emotion Classifier):**
- `0 = SAD` (Distressed vocalisation)
- `1 = LAUGH` (Happy vocalisation)

## Split Policy & Data Leakage Prevention

- Train / Val / Test = **80 / 10 / 10**
- **Model 1 split type:** Feature-level split (Feature [window-level tensor representation]).
- **Model 2 split type:** File-level split before extraction (File-level isolation [no shared source file across splits]).

> [!CAUTION]
> The biggest trap in published academic papers is data leakage through augmentation before splitting. This pipeline strictly partitions files *before* any SNR noise mixing or pitch-shifting is applied.

## Reproducibility
To reproduce the environment:
1. Re-create the folder structure matching the directories above.
2. Download the external sets (CryCeleb, ESC-50, DEMAND).
3. Populate `sad/` and `laugh/` with raw WAV files.
4. Run `python code/split_datasets.py` (if required by the pipeline iteration).
