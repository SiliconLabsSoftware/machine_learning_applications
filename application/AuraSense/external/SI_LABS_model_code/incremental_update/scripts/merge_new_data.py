#!/usr/bin/env python3
import argparse
import hashlib
import os
import shutil
from pathlib import Path

AUDIO_EXT = {".wav", ".mp3", ".m4a", ".3gp", ".ogg", ".flac"}


def sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(block_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_audio_files(root: Path):
    files = []
    if not root.exists():
        return files
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in AUDIO_EXT:
            files.append(p)
    return files


def build_hash_index(target_root: Path):
    index = {}
    for p in collect_audio_files(target_root):
        try:
            index[sha256_file(p)] = p
        except Exception:
            continue
    return index


def safe_copy(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    candidate = dst_dir / src.name
    if not candidate.exists():
        shutil.copy2(src, candidate)
        return candidate

    stem = src.stem
    suffix = src.suffix
    counter = 1
    while True:
        renamed = dst_dir / f"{stem}__new{counter}{suffix}"
        if not renamed.exists():
            shutil.copy2(src, renamed)
            return renamed
        counter += 1


def merge_class(src_dir: Path, dst_dir: Path, hash_index: dict):
    added, skipped, errors = 0, 0, 0
    for src in collect_audio_files(src_dir):
        try:
            h = sha256_file(src)
            if h in hash_index:
                skipped += 1
                continue
            copied_path = safe_copy(src, dst_dir)
            hash_index[h] = copied_path
            added += 1
        except Exception:
            errors += 1
    return added, skipped, errors


def main():
    parser = argparse.ArgumentParser(
        description="Merge new labeled audio into datasets1 with hash-based dedup (incremental learning prep)."
    )
    parser.add_argument(
        "--new-data-root",
        default="incremental_update/new_data",
        help="Folder containing subfolders: sad, laugh, not_cry, cryceleb_extra",
    )
    parser.add_argument(
        "--dataset-root",
        default="datasets1",
        help="Existing dataset root to merge into",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Analyze only; do not copy files",
    )
    args = parser.parse_args()

    new_root = Path(args.new_data_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    if not dataset_root.exists():
        candidate = Path("dataset").resolve()
        if candidate.exists():
            dataset_root = candidate

    class_map = {
        "sad": dataset_root / "sad",
        "laugh": dataset_root / "laugh",
        "not_cry": dataset_root / "background",
        "cryceleb_extra": dataset_root / "audio",
    }

    if not dataset_root.exists():
        raise SystemExit(
            f"Dataset root not found: {dataset_root}. "
            f"Pass --dataset-root explicitly (e.g. --dataset-root dataset)."
        )

    hash_index = build_hash_index(dataset_root)

    total_added = total_skipped = total_errors = 0

    print("=== Incremental Merge Summary ===")
    print(f"New data root : {new_root}")
    print(f"Dataset root  : {dataset_root}")
    print(f"Dry run       : {args.dry_run}")

    for cls, target_dir in class_map.items():
        source_dir = new_root / cls
        if not source_dir.exists():
            print(f"- {cls:12s} -> missing (skip)")
            continue

        if args.dry_run:
            files = collect_audio_files(source_dir)
            dedup_hits = 0
            for src in files:
                try:
                    if sha256_file(src) in hash_index:
                        dedup_hits += 1
                except Exception:
                    pass
            can_add = len(files) - dedup_hits
            print(f"- {cls:12s} -> total={len(files)} add={can_add} skip={dedup_hits}")
            total_added += can_add
            total_skipped += dedup_hits
            continue

        added, skipped, errors = merge_class(source_dir, target_dir, hash_index)
        total_added += added
        total_skipped += skipped
        total_errors += errors
        print(f"- {cls:12s} -> added={added} skipped={skipped} errors={errors}")

    print("-------------------------------")
    print(f"TOTAL added   : {total_added}")
    print(f"TOTAL skipped : {total_skipped}")
    print(f"TOTAL errors  : {total_errors}")


if __name__ == "__main__":
    main()
