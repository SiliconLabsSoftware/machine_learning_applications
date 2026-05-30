# Incremental Update (No Retrain From Scratch)

This folder is for **expanding dataset + continuing training from existing weights**.

## What you asked (implemented)
- Add new data without starting over.
- Keep old model and continue learning (Fine-tuning [continue optimizing an existing model checkpoint]).
- Support very different new data with safer strategy.

## Folder layout
- `new_data/sad/` : new distressed baby samples
- `new_data/laugh/` : new happy baby samples
- `new_data/not_cry/` : new background/environment samples
- `new_data/cryceleb_extra/` : extra cry files (if any)
- `scripts/merge_new_data.py` : dedup + merge helper
- `checkpoints/` : place/track model snapshots for incremental runs
- `outputs/` : logs/results from incremental sessions

## Step A — add your new files
Drop files into relevant `new_data/*` class folders.

## Step B — merge safely (no duplicates)
Use hash-based deduplication (SHA256 [content fingerprint]):

```bash
cd /Users/rishabhsahay/Desktop/hola
/usr/bin/python3 incremental_update/scripts/merge_new_data.py --dry-run
/usr/bin/python3 incremental_update/scripts/merge_new_data.py
```

Windows PowerShell equivalent:

```powershell
cd C:\Users\<YourUser>\Desktop\hola
python incremental_update\scripts\merge_new_data.py --dry-run
python incremental_update\scripts\merge_new_data.py
```

## Step C — continue training from old checkpoints (not scratch)
Yes, this is possible and recommended.

### Concept
- Load existing `.pt` weights (Checkpoint [saved model state]).
- Train for fewer epochs with lower LR (Learning Rate [step size]).
- Keep part of old data in batches (Rehearsal [mix old+new to avoid forgetting]).

### Why for very different new data
If new data is very different (Domain Shift [distribution mismatch]):
1. Start with small LR (`1e-4` or lower)
2. Freeze early layers first (Feature extractor freeze [retain base acoustic patterns])
3. Train classifier head first, then unfreeze gradually
4. Mix old+new data (e.g., 70:30 or 50:50) to reduce catastrophic forgetting
5. Validate with old test set + new validation set

## Practical note for your current codebase
Your current `code/baby_cry_v3_local.py` is pipeline-style and not modular for resume hooks yet.

So recommended path:
- Keep using current full pipeline for baseline.
- Add a separate `resume_finetune.py` next (can be done), which:
  - loads `outputs/models/model_m1_detector.pt`
  - loads merged dataset
  - runs short fine-tune schedule
  - saves `outputs/models/model_m1_detector_ft.pt`

If you want, I can implement this `resume_finetune.py` in the next step.

## Old code + very different data: is it possible?
Yes — with transfer/fine-tuning, not scratch retrain.
But if labels/classes changed heavily, you may need partial head replacement (Classifier head swap [new final layer for changed class semantics]).
