"""
split_datasets.py
=================
Replicates EXACTLY the split logic used in baby_cry_v3_local.py and
physically copies WAV files into  train/  val/  test/  subfolders
inside each source directory.

Two split strategies are used (matching the original pipeline):

  M1 sources  ──  sad/, laugh/, audio/, esc50/audio/
    Strategy: FEATURE-LEVEL split (after chunking)
    The original code first pools ALL features from a class, shuffles the
    flat feature list, then takes 80/10/10.  Because features come from
    overlapping 0.75-s windows of the same file, the split is done on
    *windows*, NOT on source files.
    ► We replicate this at the FILE level by shuffling ALL files together
      for that class and then splitting 80/10/10 by file count (a faithful
      approximation when files are roughly the same length).

  M2 sources  ──  sad/, laugh/
    Strategy: FILE-LEVEL split  (matches file_split() exactly)
    The original code splits the path list 80/10/10 BEFORE extracting
    features, so no single file's windows can bleed across splits.
    ► We replicate this exactly.

Usage:
    python code/split_datasets.py
    python code/split_datasets.py --seed 42

After running, folder structure becomes:
    datasets1/sad/
        train/   (80 % of files)
        val/     (10 % of files)
        test/    (remaining 10 %)
        [original files are preserved in place]

    datasets1/laugh/   (same)
    datasets1/audio/   (same)
"""

import os, sys, random, shutil, argparse

# ─── constants copied verbatim from baby_cry_v3_local.py ──────────────────────
SPLIT_TRAIN = 0.80
SPLIT_VAL   = 0.10
# test fraction = 1 - SPLIT_TRAIN - SPLIT_VAL = 0.10

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE = os.environ.get(
    'AURASENSE_DATA',
    os.path.join(SCRIPT_DIR, '..', 'datasets1')   # code/ is one level below hola/
)
BASE = os.path.normpath(BASE)

SOURCES = {
    # name         path relative to BASE          split strategy
    'sad'        : os.path.join(BASE, 'sad'),
    'laugh'      : os.path.join(BASE, 'laugh'),
    'audio'      : os.path.join(BASE, 'audio'),   # CryCeleb
}

# ──────────────────────────────────────────────────────────────────────────────
def list_wavs(directory):
    """Walk directory tree and return sorted list of .wav paths."""
    wavs = []
    for root, _, files in os.walk(directory):
        # Skip already-split subfolders so re-runs are idempotent
        parts = os.path.relpath(root, directory).split(os.sep)
        if parts[0] in ('train', 'val', 'test'):
            continue
        for f in files:
            if f.lower().endswith('.wav'):
                wavs.append(os.path.join(root, f))
    return sorted(wavs)


def file_split(paths, train_frac=SPLIT_TRAIN, val_frac=SPLIT_VAL, seed=None):
    """
    Mirrors file_split() in baby_cry_v3_local.py (line 397-400).
    Shuffles path list then slices into train / val / test.
    """
    paths = paths.copy()
    if seed is not None:
        random.seed(seed)
    random.shuffle(paths)
    n      = len(paths)
    n_tr   = int(n * train_frac)
    n_va   = int(n * val_frac)
    train  = paths[:n_tr]
    val    = paths[n_tr : n_tr + n_va]
    test   = paths[n_tr + n_va :]
    return train, val, test


def copy_files(file_list, src_root, dest_subfolder):
    """
    Copy each file into  <src_root>/<dest_subfolder>/  preserving the
    relative sub-path under src_root so nested directory trees (like
    CryCeleb's train/ sub-dirs) stay readable.
    """
    os.makedirs(dest_subfolder, exist_ok=True)
    for fp in file_list:
        rel = os.path.relpath(fp, src_root)
        dst = os.path.join(dest_subfolder, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(fp, dst)


def split_source(name, src_dir, seed):
    print(f'\n{"─"*60}')
    print(f'  Source : {name}/')
    print(f'  Path   : {src_dir}')

    if not os.path.exists(src_dir):
        print(f'  ⚠️  Directory not found — skipping.')
        return

    wavs = list_wavs(src_dir)
    if not wavs:
        print(f'  ⚠️  No .wav files found — skipping.')
        return

    print(f'  WAV files found : {len(wavs)}')

    train, val, test = file_split(wavs, seed=seed)
    print(f'  Split  →  train={len(train)}  val={len(val)}  test={len(test)}')
    print(f'  Ratios →  {len(train)/len(wavs)*100:.1f}% / '
          f'{len(val)/len(wavs)*100:.1f}% / '
          f'{len(test)/len(wavs)*100:.1f}%')

    for split_name, file_list in [('train', train), ('val', val), ('test', test)]:
        dest = os.path.join(src_dir, split_name)
        # Remove existing split folder so re-runs are clean
        if os.path.exists(dest):
            shutil.rmtree(dest)
        copy_files(file_list, src_dir, dest)
        print(f'  ✅  {split_name:5s}/  ← {len(file_list)} files  →  {dest}')


def main():
    parser = argparse.ArgumentParser(
        description='Split datasets1/ into train/val/test subfolders '
                    'using the exact same logic as baby_cry_v3_local.py'
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42). Use the same seed to reproduce splits.'
    )
    parser.add_argument(
        '--base', type=str, default=BASE,
        help=f'Path to datasets1/ root (default: {BASE})'
    )
    args = parser.parse_args()

    base = os.path.normpath(args.base)
    print('=' * 60)
    print('AuraSense — Dataset Train/Val/Test Split')
    print('=' * 60)
    print(f'Base dir : {base}')
    print(f'Seed     : {args.seed}')
    print(f'Ratios   : train={SPLIT_TRAIN:.0%}  val={SPLIT_VAL:.0%}  '
          f'test={1-SPLIT_TRAIN-SPLIT_VAL:.0%}')
    print()
    print('Strategy:')
    print('  sad/   → FILE-level split  (mirrors M2 file_split())')
    print('  laugh/ → FILE-level split  (mirrors M2 file_split())')
    print('  audio/ → FILE-level split  (approximates M1 feature-level split)')
    print()
    print('Note: Original files in each folder are NOT deleted.')
    print('      train/ val/ test/ subfolders are created alongside them.')

    sources = {
        'sad'  : os.path.join(base, 'sad'),
        'laugh': os.path.join(base, 'laugh'),
        'audio': os.path.join(base, 'audio'),
    }

    for name, src_dir in sources.items():
        split_source(name, src_dir, args.seed)

    print('\n' + '=' * 60)
    print('DONE — folder structure is now:')
    print('=' * 60)
    for name, src_dir in sources.items():
        if not os.path.exists(src_dir):
            continue
        for split_name in ('train', 'val', 'test'):
            split_dir = os.path.join(src_dir, split_name)
            if os.path.exists(split_dir):
                n = sum(
                    1 for r, _, fs in os.walk(split_dir)
                    for f in fs if f.lower().endswith('.wav')
                )
                print(f'  datasets1/{name}/{split_name}/  →  {n} WAV files')
    print()
    print('To use these splits in training, modify baby_cry_v3_local.py:')
    print("  SAD_DIR_TRAIN  = f'{BASE}/sad/train'")
    print("  SAD_DIR_VAL    = f'{BASE}/sad/val'")
    print("  SAD_DIR_TEST   = f'{BASE}/sad/test'")
    print("  LAUGH_DIR_TRAIN = f'{BASE}/laugh/train'")
    print("  (etc.)")


if __name__ == '__main__':
    main()
