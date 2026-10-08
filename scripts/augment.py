#!/usr/bin/env python3
"""
augment.py — Data augmentation for accelerometer gesture CSVs.

Reads raw captures from data/<gesture>/**/*.csv (searches recursively,
so extra sub-folders like data/gesture_L/thgh9905_L_Gesture/ are handled
automatically) and writes augmented copies to data_augmented/<gesture>/
using four techniques appropriate for 3-axis IMU gesture data:

  1. Noise jitter       — adds small Gaussian noise to x, y, z
  2. Time-shifting      — drops a few samples from the start, zero-pads the end
  3. Amplitude scaling  — multiplies x, y, z by a scalar near 1.0
  4. Axis rotation      — applies a small random 3-D rotation

Each source file produces N_AUGMENTED copies per technique
(default 3 per technique → 12 augmented files per source file).

Supports two column-naming conventions automatically:
  • SensorLogger:  time, seconds_elapsed, x, y, z, absolute_acceleration
  • Phyphox:       Time (s), Acceleration x (m/s^2), …, Absolute acceleration (m/s^2)

The output is always flattened into data_augmented/<gesture>/:
  data/gesture_L/thgh9905_L_Gesture/Raw Data 1.csv
    →  data_augmented/gesture_L/Raw Data 1__noise_00.csv
       data_augmented/gesture_L/Raw Data 1__noise_01.csv
       ...

Usage:
  python3 scripts/augment.py
  python3 scripts/augment.py --data-dir data --out-dir data_augmented --n 3 --seed 42
"""

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np


# ── Configurable defaults ───────────────────────────────────────────────────

DEFAULT_DATA_DIR = Path("data")
DEFAULT_OUT_DIR  = Path("data_augmented")
N_AUGMENTED      = 3   # copies per technique per source file
RANDOM_SEED      = 42

# Technique parameters
NOISE_STD         = 0.05   # Gaussian σ, same units as your sensor (m/s²)
SHIFT_MAX_SAMPLES = 10     # max samples to drop from the start
SCALE_RANGE       = (0.85, 1.15)   # uniform range for amplitude multiplier
ROTATION_MAX_DEG  = 15     # max per-axis rotation angle (degrees)


# ── Column-name aliases ─────────────────────────────────────────────────────
# Maps canonical name → list of accepted variants (strip + lower compared).
# First match in the header wins.

COL_ALIASES: dict[str, list[str]] = {
    "x":   ["x", "acceleration x (m/s^2)"],
    "y":   ["y", "acceleration y (m/s^2)"],
    "z":   ["z", "acceleration z (m/s^2)"],
    "abs": ["absolute_acceleration", "absolute acceleration (m/s^2)"],
}


def _find_col(header: list[str], canonical: str) -> int | None:
    """Return the index of the first header cell matching any alias for
    `canonical`, or None if not found."""
    aliases = COL_ALIASES[canonical]
    for i, cell in enumerate(header):
        if cell.strip().lower() in aliases:
            return i
    return None


# ── CSV I/O ─────────────────────────────────────────────────────────────────

def read_gesture_csv(path: Path):
    """Parse a gesture CSV.  Accepts both SensorLogger and Phyphox column
    naming conventions (see COL_ALIASES above).

    Returns:
        header        list[str]       — column names as read from the file
        comment_lines list[str]       — leading '#' lines (preserved verbatim)
        data_rows     list[list[str]] — string rows (index 0 = first data row)
        xi, yi, zi    int             — column indices for x, y, z
        xyz           np.ndarray      — shape (N, 3), float32
    """
    with open(path, newline="", errors="replace") as f:
        raw = f.readlines()

    comment_lines = [l for l in raw if l.startswith("#")]
    data_lines    = [l for l in raw if not l.startswith("#")]

    reader    = csv.reader(data_lines)
    all_rows  = list(reader)
    if len(all_rows) < 2:
        raise ValueError(f"File has no data rows: {path}")

    header    = all_rows[0]
    data_rows = all_rows[1:]

    xi = _find_col(header, "x")
    yi = _find_col(header, "y")
    zi = _find_col(header, "z")

    missing = [name for name, idx in zip(["x", "y", "z"], [xi, yi, zi]) if idx is None]
    if missing:
        raise ValueError(
            f"{path}: could not find column(s) {missing} — "
            f"header was {header}"
        )

    try:
        xyz = np.array(
            [[float(r[xi]), float(r[yi]), float(r[zi])] for r in data_rows],
            dtype=np.float32,
        )
    except (ValueError, IndexError) as exc:
        raise ValueError(f"{path}: could not parse x/y/z values — {exc}") from exc

    return header, comment_lines, data_rows, xi, yi, zi, xyz


def write_gesture_csv(
    dest: Path,
    header: list,
    comment_lines: list,
    data_rows: list,
    xi: int,
    yi: int,
    zi: int,
    xyz_new: np.ndarray,
) -> None:
    """Write augmented CSV.  Replaces x/y/z with xyz_new values and
    recomputes absolute_acceleration; all other columns (timestamps, etc.)
    are copied verbatim from the source rows.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    abs_i = _find_col(header, "abs")

    with open(dest, "w", newline="") as f:
        # Re-emit leading comment lines
        for cl in comment_lines:
            f.write(cl)
        writer = csv.writer(f)
        writer.writerow(header)
        for row_idx, row in enumerate(data_rows):
            row = list(row)
            if row_idx < len(xyz_new):
                row[xi] = f"{xyz_new[row_idx, 0]:.6f}"
                row[yi] = f"{xyz_new[row_idx, 1]:.6f}"
                row[zi] = f"{xyz_new[row_idx, 2]:.6f}"
                if abs_i is not None:
                    mag = float(np.linalg.norm(xyz_new[row_idx]))
                    row[abs_i] = f"{mag:.6f}"
            writer.writerow(row)


# ── Augmentation techniques ─────────────────────────────────────────────────

def augment_noise(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add zero-mean Gaussian noise to every sample on every axis.
    Simulates natural sensor noise variation across repetitions."""
    noise = rng.normal(0.0, NOISE_STD, size=xyz.shape).astype(np.float32)
    return xyz + noise


def augment_time_shift(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Drop 1–SHIFT_MAX_SAMPLES samples from the start and zero-pad the end.
    Simulates a gesture that starts slightly later in the capture window
    (or the window that starts slightly before the gesture does)."""
    n_drop = int(rng.integers(1, SHIFT_MAX_SAMPLES + 1))
    shifted = xyz[n_drop:]
    pad     = np.zeros((n_drop, 3), dtype=np.float32)
    return np.vstack([shifted, pad])


def augment_amplitude_scale(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Multiply all three axes by a scalar drawn uniformly from SCALE_RANGE.
    Simulates the same gesture performed with more or less force/speed."""
    scale = float(rng.uniform(*SCALE_RANGE))
    return (xyz * scale).astype(np.float32)


def _rotation_matrix_3d(ax: float, ay: float, az: float) -> np.ndarray:
    """3×3 rotation matrix from Euler angles (radians) as Rx @ Ry @ Rz."""
    cx, sx = math.cos(ax), math.sin(ax)
    cy, sy = math.cos(ay), math.sin(ay)
    cz, sz = math.cos(az), math.sin(az)

    Rx = np.array([[1,  0,   0  ],
                   [0,  cx, -sx ],
                   [0,  sx,  cx ]], dtype=np.float64)
    Ry = np.array([[ cy, 0, sy ],
                   [  0, 1,  0 ],
                   [-sy, 0, cy ]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0 ],
                   [sz,  cz, 0 ],
                   [ 0,   0, 1 ]], dtype=np.float64)
    return Rx @ Ry @ Rz


def augment_rotation(xyz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Apply a small random 3-D rotation (up to ROTATION_MAX_DEG per axis).
    Simulates holding the phone at a slightly different wrist/hand angle."""
    max_rad = math.radians(ROTATION_MAX_DEG)
    angles  = rng.uniform(-max_rad, max_rad, size=3)
    R       = _rotation_matrix_3d(*angles)
    return (xyz @ R.T).astype(np.float32)


# Ordered list of (name, function) pairs.  Each produces N_AUGMENTED copies.
TECHNIQUES = [
    ("noise",     augment_noise),
    ("timeshift", augment_time_shift),
    ("scale",     augment_amplitude_scale),
    ("rotation",  augment_rotation),
]


# ── Per-file processing ─────────────────────────────────────────────────────

def process_file(src: Path, dest_dir: Path, rng: np.random.Generator, n: int) -> int:
    """Augment one source CSV.  Writes n copies per technique to dest_dir.
    Returns the count of files written."""
    header, comment_lines, data_rows, xi, yi, zi, xyz = read_gesture_csv(src)
    stem    = src.stem
    written = 0

    for tech_name, tech_fn in TECHNIQUES:
        for i in range(n):
            xyz_aug   = tech_fn(xyz, rng)
            dest_name = f"{stem}__{tech_name}_{i:02d}.csv"
            write_gesture_csv(
                dest_dir / dest_name,
                header, comment_lines, data_rows,
                xi, yi, zi, xyz_aug,
            )
            written += 1

    return written


# ── Entry point ─────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR,
                    help=f"Root data directory containing gesture sub-folders "
                         f"(default: {DEFAULT_DATA_DIR})")
    ap.add_argument("--out-dir",  type=Path, default=DEFAULT_OUT_DIR,
                    help=f"Output directory for augmented files "
                         f"(default: {DEFAULT_OUT_DIR})")
    ap.add_argument("--n",    type=int, default=N_AUGMENTED,
                    help=f"Augmented copies per technique per source file "
                         f"(default: {N_AUGMENTED})")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED,
                    help=f"Random seed for reproducibility (default: {RANDOM_SEED})")
    args = ap.parse_args()

    data_dir: Path = args.data_dir
    out_dir:  Path = args.out_dir
    rng = np.random.default_rng(args.seed)

    if not data_dir.is_dir():
        print(f"ERROR: data directory '{data_dir}' not found.", file=sys.stderr)
        sys.exit(1)

    gesture_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if not gesture_dirs:
        print(f"ERROR: no sub-directories found under '{data_dir}'.", file=sys.stderr)
        sys.exit(1)

    total_src     = 0
    total_written = 0

    for gesture_dir in gesture_dirs:
        gesture   = gesture_dir.name
        # rglob so extra sub-folders (e.g. per-person capture dirs) are included
        csv_files = sorted(gesture_dir.rglob("Accelerometer.csv"))
        if not csv_files:
            print(f"  [{gesture}]  no CSVs found — skipping")
            continue

        dest_dir = out_dir / gesture
        dest_dir.mkdir(parents=True, exist_ok=True)

        n_tech = len(TECHNIQUES)
        print(f"  [{gesture}]  {len(csv_files)} source file(s)  "
              f"→  {len(csv_files) * n_tech * args.n} augmented files  "
              f"({n_tech} techniques × {args.n} copies each)")

        for src in csv_files:
            try:
                count          = process_file(src, dest_dir, rng, args.n)
                total_src     += 1
                total_written += count
            except Exception as exc:
                print(f"      WARNING: skipping {src.name}: {exc}", file=sys.stderr)

    techniques_str = ", ".join(name for name, _ in TECHNIQUES)
    print(f"\nDone.")
    print(f"  Source files processed : {total_src}")
    print(f"  Augmented files written: {total_written}  (techniques: {techniques_str})")
    print(f"  Output root            : {out_dir}/")


if __name__ == "__main__":
    main()
