#!/usr/bin/env python3
"""
train_demo.py — Demo training script for gesture-recognition dataset.

Loads raw accelerometer CSVs from data/ (and augmented files from
data_augmented/ if it exists), extracts per-recording statistical features,
and trains + evaluates a Random Forest classifier.

This is a genuine, correctly-wired demonstration of the full pipeline;
it does not need to achieve high accuracy on its own.

Feature extraction
------------------
Each recording is summarised as a 12-element vector:
  [mean_x, std_x, min_x, max_x,  mean_y, ...,  mean_z, std_z, min_z, max_z]

Model
-----
RandomForestClassifier (100 trees, sklearn) — a sensible baseline that
trains in seconds on a dataset this size.

Setup
-----
  python3 -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt

Usage
-----
  # Raw data only (looks for data/ relative to cwd):
  python3 train_demo.py

  # Raw + augmented data:
  python3 train_demo.py --aug-dir data_augmented

  # Custom paths / options:
  python3 train_demo.py --data-dir data --aug-dir data_augmented \\
                        --test-size 0.25 --seed 0

Column conventions supported (matched by name, not position)
-------------------------------------------------------------
  SensorLogger: time, seconds_elapsed, z, y, x
  Phyphox:      Time (s), Acceleration x (m/s^2), Acceleration y (m/s^2),
                Acceleration z (m/s^2)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


# ── Column-name aliases (mirrors augment.py) ──────────────────────────────────

_ALIASES: dict[str, list[str]] = {
    "x": ["x", "acceleration x (m/s^2)"],
    "y": ["y", "acceleration y (m/s^2)"],
    "z": ["z", "acceleration z (m/s^2)"],
}


def _resolve_xyz_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with columns renamed to 'x', 'y', 'z' (if needed).
    Raises KeyError if any axis column cannot be found."""
    # Normalise column names for matching
    lc = {c: c.strip().lower() for c in df.columns}
    rename: dict[str, str] = {}
    for canonical, aliases in _ALIASES.items():
        if canonical in lc.values():
            continue  # already present
        for raw_col, low_col in lc.items():
            if low_col in aliases:
                rename[raw_col] = canonical
                break
    if rename:
        df = df.rename(columns=rename)
    missing = [ax for ax in ("x", "y", "z") if ax not in df.columns]
    if missing:
        raise KeyError(f"Could not find columns {missing} in header {list(df.columns)}")
    return df


# ── Data loading ───────────────────────────────────────────────────────────────

def load_xyz(path: Path) -> np.ndarray | None:
    """Read one CSV and return shape-(N, 3) float32 [x, y, z], or None."""
    try:
        df = pd.read_csv(path, comment="#")
        df = _resolve_xyz_cols(df)
        xyz = df[["x", "y", "z"]].to_numpy(dtype=np.float32)
        return xyz if len(xyz) >= 2 else None
    except Exception as exc:
        print(f"  WARNING: skipping {path.name}: {exc}", file=sys.stderr)
        return None


def label_from_path(path: Path) -> str:
    """Walk path components for the first one starting with 'gesture_'
    and strip the prefix, e.g. gesture_U → 'U', gesture_3 → '3'."""
    for part in path.parts:
        if part.lower().startswith("gesture_"):
            return part.split("_", 1)[1]
    raise ValueError(f"No gesture_* component in path: {path}")


def collect_dataset(
    data_dir: Path,
    aug_dir: Path | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Gather all Accelerometer.csv files from data_dir (and aug_dir if
    given), extract features, and return (X, y)."""

    sources: list[Path] = []
    if data_dir.is_dir():
        sources.extend(sorted(data_dir.rglob("Accelerometer.csv")))
    if aug_dir and aug_dir.is_dir():
        sources.extend(sorted(aug_dir.rglob("Accelerometer*.csv")))

    if not sources:
        print(f"ERROR: no Accelerometer CSV files found under '{data_dir}'.",
              file=sys.stderr)
        sys.exit(1)

    X_rows: list[np.ndarray] = []
    y_rows: list[str] = []
    n_skipped = 0

    for path in sources:
        try:
            label = label_from_path(path)
        except ValueError as exc:
            print(f"  WARNING: {exc}", file=sys.stderr)
            n_skipped += 1
            continue

        xyz = load_xyz(path)
        if xyz is None:
            n_skipped += 1
            continue

        X_rows.append(_extract_features(xyz))
        y_rows.append(label)

    if not X_rows:
        print("ERROR: no usable samples loaded.", file=sys.stderr)
        sys.exit(1)

    if n_skipped:
        print(f"  ({n_skipped} file(s) skipped due to parse errors)")

    return np.array(X_rows, dtype=np.float32), np.array(y_rows)


# ── Feature extraction ─────────────────────────────────────────────────────────

def _extract_features(xyz: np.ndarray) -> np.ndarray:
    """12-element feature vector: mean/std/min/max for each of x, y, z."""
    feats: list[float] = []
    for col in range(3):  # x=0, y=1, z=2
        axis = xyz[:, col]
        feats += [float(axis.mean()), float(axis.std()),
                  float(axis.min()),  float(axis.max())]
    return np.array(feats, dtype=np.float32)


FEATURE_NAMES = [
    f"{stat}_{ax}"
    for ax in ("x", "y", "z")
    for stat in ("mean", "std", "min", "max")
]


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data"),
                    help="Root data directory containing gesture_* sub-folders "
                         "(default: data)")
    ap.add_argument("--aug-dir", type=Path, default=None,
                    help="Augmented data directory from augment.py "
                         "(default: data_augmented if present, else skipped)")
    ap.add_argument("--no-aug", action="store_true",
                    help="Ignore data_augmented/ even if it exists")
    ap.add_argument("--test-size", type=float, default=0.2,
                    help="Held-out test fraction (default: 0.2)")
    ap.add_argument("--seed", type=int, default=42,
                    help="Random seed (default: 42)")
    args = ap.parse_args()

    # Resolve aug_dir
    aug_dir: Path | None = None
    if not args.no_aug:
        candidate = args.aug_dir if args.aug_dir else Path("data_augmented")
        if candidate.is_dir():
            aug_dir = candidate

    # ── Banner ────────────────────────────────────────────────────────────────
    print("=" * 60)
    print("  Gesture Recognition — Demo Training Script")
    print("=" * 60)
    print(f"  Raw data dir  : {args.data_dir}")
    print(f"  Augmented dir : {aug_dir or '(none — run augment.py first to add)'}")
    print()

    # ── Load ──────────────────────────────────────────────────────────────────
    print("Loading dataset …")
    X, y = collect_dataset(args.data_dir, aug_dir)

    classes, counts = np.unique(y, return_counts=True)
    print(f"  Samples loaded  : {len(X)}")
    print(f"  Classes         : {list(classes)}")
    print(f"  Samples / class : { {c: int(n) for c, n in zip(classes, counts)} }")
    print(f"  Feature vector  : {X.shape[1]} dims "
          f"(mean/std/min/max × x/y/z per recording)")
    print()

    if len(classes) < 2:
        print("ERROR: need at least 2 classes to train a classifier.", file=sys.stderr)
        sys.exit(1)

    # ── Train / test split ────────────────────────────────────────────────────
    # Stratify when every class has ≥2 samples; fall back to random otherwise.
    stratify = y if min(counts) >= 2 else None
    if stratify is None:
        print("  NOTE: one or more classes has only 1 sample — "
              "skipping stratification.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=stratify,
    )
    print(f"Train / test split: {len(X_train)} train / {len(X_test)} test  "
          f"({100*(1-args.test_size):.0f}% / {100*args.test_size:.0f}%)")
    print()

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Training RandomForestClassifier (100 trees) …")
    clf = RandomForestClassifier(n_estimators=100, random_state=args.seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    print("  Fit complete.")
    print()

    # ── Evaluate ──────────────────────────────────────────────────────────────
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("Results on held-out test set")
    print("-" * 40)
    print(f"  Accuracy : {acc:.1%}  "
          f"({int(round(acc * len(y_test)))}/{len(y_test)} correct)")
    print()
    print(classification_report(y_test, y_pred, zero_division=0))

    # ── Feature importances ───────────────────────────────────────────────────
    ranked = sorted(
        zip(clf.feature_importances_, FEATURE_NAMES), reverse=True
    )
    print("Top 6 features by importance:")
    for imp, name in ranked[:6]:
        bar = "█" * int(imp * 40)
        print(f"  {name:<12}  {imp:.3f}  {bar}")
    print()
    print("Pipeline demo complete.  Accuracy depends on dataset size and ")
    print("composition — augment first (scripts/augment.py) to add more samples.")


if __name__ == "__main__":
    main()
