#!/usr/bin/env python3
"""
fix_repo.py — Repair and clean the gesture dataset repository.

Run from the repo root:
    python3 fix_repo.py            # dry run — shows what would change
    python3 fix_repo.py --apply    # actually make the changes

Actions
-------
1. Rename thgh9905's Phyphox "Raw Data N.csv" files into the standard
   dated-subfolder / Accelerometer.csv structure used by every other
   contributor.  Timestamps are taken from the matching Phyphox .zip
   filenames (real capture times, not synthetic).

2. Delete all .zip files anywhere under data/  (Phyphox export artifacts
   from thgh9905, Jagi4310_Idle, ahab5661_idle — the CSVs are what matter).

3. Delete orphaned sidecar files that sit directly in a contributor folder
   rather than inside a dated subfolder:
   AccelerometerUncalibrated.csv, Annotation.csv, Metadata.csv,
   TotalAcceleration.csv, Tags.csv.  These are SensorLogger extras that
   got dropped one level too high; they contain no gesture data.

4. Delete the duplicate train_demo.py at the repo root (the canonical
   copy lives in scripts/).
"""

import argparse
import shutil
import sys
from pathlib import Path

# ── Real capture timestamps for thgh9905, taken from the Phyphox zip names ──
# Each list maps Raw Data N.csv (1-indexed) to a folder name.
# Format: 2026-10-08_HH-MM-SS  (all captures were on 2026-10-08)

THGH_TIMESTAMPS = {
    "gesture_3": [
        "2026-10-08_21-42-53",
        "2026-10-08_21-43-10",
        "2026-10-08_21-43-33",
        "2026-10-08_21-43-42",
        "2026-10-08_21-43-58",
        "2026-10-08_21-44-10",
        "2026-10-08_21-44-20",
    ],
    "gesture_L": [
        "2026-10-08_21-46-39",
        "2026-10-08_21-46-48",
        "2026-10-08_21-46-56",
        "2026-10-08_21-47-03",
        "2026-10-08_21-47-14",
        "2026-10-08_21-47-21",
        "2026-10-08_21-47-27",
        # gesture_L had 12 zips but 7 CSVs — use first 7 timestamps
    ],
    "gesture_U": [
        "2026-10-08_21-49-37",
        "2026-10-08_21-49-46",
        "2026-10-08_21-49-55",
        "2026-10-08_21-50-05",
        "2026-10-08_21-50-14",
        "2026-10-08_21-50-24",
        "2026-10-08_21-50-31",
        "2026-10-08_21-50-39",
        "2026-10-08_21-50-47",
        "2026-10-08_21-50-55",
        "2026-10-08_21-51-02",
    ],
    "gesture_idle": [
        "2026-10-08_21-53-39",
        "2026-10-08_21-53-51",
        "2026-10-08_21-54-01",
        "2026-10-08_21-54-11",
        "2026-10-08_21-54-22",
        "2026-10-08_21-54-33",
        "2026-10-08_21-54-43",
    ],
}

# Contributor subfolder name for each gesture class
THGH_CONTRIB_DIRS = {
    "gesture_3":    "thgh9905_3_Gesture",
    "gesture_L":    "thgh9905_L_Gesture",
    "gesture_U":    "thgh9905_U_Gesture",
    "gesture_idle": "thgh9905",
}

# Sidecar filenames that are safe to delete when found at contributor level
SIDECAR_NAMES = {
    "AccelerometerUncalibrated.csv",
    "Annotation.csv",
    "Metadata.csv",
    "TotalAcceleration.csv",
    "Tags.csv",
}


def log(apply: bool, action: str, src: Path, dst: Path | None = None) -> None:
    verb = action if apply else f"[DRY] {action}"
    if dst:
        print(f"  {verb}: {src}  →  {dst}")
    else:
        print(f"  {verb}: {src}")


def rename_thgh(data_dir: Path, apply: bool) -> int:
    """Move each Raw Data N.csv into a dated subfolder as Accelerometer.csv."""
    count = 0
    for gesture, timestamps in THGH_TIMESTAMPS.items():
        contrib_dir = data_dir / gesture / THGH_CONTRIB_DIRS[gesture]
        if not contrib_dir.exists():
            print(f"  WARN: expected folder not found: {contrib_dir}")
            continue

        # Collect Raw Data files sorted by N
        raw_files = sorted(
            contrib_dir.glob("Raw Data *.csv"),
            key=lambda p: int(p.stem.split()[-1]),
        )

        if len(raw_files) != len(timestamps):
            print(
                f"  WARN: {gesture} — found {len(raw_files)} Raw Data CSVs "
                f"but have {len(timestamps)} timestamps; will pair what we can"
            )

        for raw_path, ts in zip(raw_files, timestamps):
            dest_dir = contrib_dir / ts
            dest = dest_dir / "Accelerometer.csv"
            log(apply, "RENAME", raw_path, dest)
            if apply:
                dest_dir.mkdir(exist_ok=True)
                shutil.move(str(raw_path), dest)
            count += 1

    return count


def delete_zips(data_dir: Path, apply: bool) -> int:
    """Remove all .zip files anywhere under data/."""
    zips = sorted(data_dir.rglob("*.zip"))
    for z in zips:
        log(apply, "DELETE", z)
        if apply:
            z.unlink()
    return len(zips)


def delete_orphan_sidecars(data_dir: Path, apply: bool) -> int:
    """
    Delete sidecar CSVs that sit directly inside a contributor folder
    (depth = data/<gesture>/<contributor>/<file>) rather than inside a
    dated subfolder.  Only removes known sidecar filenames — never
    Accelerometer.csv.
    """
    count = 0
    for gesture_dir in sorted(data_dir.iterdir()):
        if not gesture_dir.is_dir() or not gesture_dir.name.startswith("gesture_"):
            continue
        for contrib_dir in sorted(gesture_dir.iterdir()):
            if not contrib_dir.is_dir():
                continue
            for f in sorted(contrib_dir.iterdir()):
                if f.is_file() and f.name in SIDECAR_NAMES:
                    log(apply, "DELETE orphan sidecar", f)
                    if apply:
                        f.unlink()
                    count += 1
    return count


def delete_root_duplicate(repo_root: Path, apply: bool) -> int:
    """Remove the stray train_demo.py at the repo root (canonical copy is in scripts/)."""
    dup = repo_root / "train_demo.py"
    scripts_copy = repo_root / "scripts" / "train_demo.py"
    if dup.exists() and scripts_copy.exists():
        log(apply, "DELETE duplicate", dup)
        if apply:
            dup.unlink()
        return 1
    if dup.exists() and not scripts_copy.exists():
        print(f"  WARN: {dup} exists but scripts/train_demo.py does NOT — skipping delete")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply", action="store_true",
        help="Actually make changes.  Without this flag, only a dry-run is shown."
    )
    parser.add_argument(
        "--repo", default=".", metavar="DIR",
        help="Path to the repo root (default: current directory)"
    )
    args = parser.parse_args()

    repo_root = Path(args.repo).resolve()
    data_dir  = repo_root / "data"

    if not data_dir.exists():
        print(f"ERROR: data/ not found under {repo_root}")
        sys.exit(1)

    mode = "APPLYING CHANGES" if args.apply else "DRY RUN — pass --apply to make changes"
    print(f"\n{'='*60}")
    print(f"  fix_repo.py  |  {mode}")
    print(f"{'='*60}\n")

    print("── 1. Rename thgh9905 Raw Data CSVs ──")
    n1 = rename_thgh(data_dir, args.apply)
    print(f"   → {n1} file(s)\n")

    print("── 2. Delete .zip files from data/ ──")
    n2 = delete_zips(data_dir, args.apply)
    print(f"   → {n2} file(s)\n")

    print("── 3. Delete orphaned sidecar CSVs ──")
    n3 = delete_orphan_sidecars(data_dir, args.apply)
    print(f"   → {n3} file(s)\n")

    print("── 4. Delete duplicate root-level train_demo.py ──")
    n4 = delete_root_duplicate(repo_root, args.apply)
    print(f"   → {n4} file(s)\n")

    total = n1 + n2 + n3 + n4
    if args.apply:
        print(f"Done. {total} change(s) applied.")
    else:
        print(f"Dry run complete. {total} change(s) pending.  Run with --apply to execute.")
    print()


if __name__ == "__main__":
    main()
