#!/usr/bin/env python3
"""
train_demo.py — Gesture classifier demo for 3-axis accelerometer data.

Loads all Accelerometer.csv files from data/ and data_augmented/ (if present),
extracts 12 statistical features per recording (mean/std/min/max × x/y/z),
trains a Random Forest classifier, and prints accuracy, a classification
report, and feature importances.

Usage:
    python3 scripts/train_demo.py
    python3 scripts/train_demo.py --data-dir data --aug-dir data_augmented --seed 42
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# Column aliases — same convention as augment.py
COL_ALIASES = {
    "x": ["x", "acceleration x (m/s^2)"],
    "y": ["y", "acceleration y (m/s^2)"],
    "z": ["z", "acceleration z (m/s^2)"],
}

def find_col(columns, canonical):
    aliases = COL_ALIASES[canonical]
    for col in columns:
        if col.strip().lower() in aliases:
            return col
    return None


def extract_features(path: Path):
    """Read one Accelerometer.csv and return a 12-element feature vector."""
    df = pd.read_csv(path, comment="#")
    df.columns = [c.strip() for c in df.columns]

    xcol = find_col(df.columns, "x")
    ycol = find_col(df.columns, "y")
    zcol = find_col(df.columns, "z")

    if not all([xcol, ycol, zcol]):
        raise ValueError(f"Missing x/y/z columns in {path} — found: {list(df.columns)}")

    feats = []
    for col in [xcol, ycol, zcol]:
        s = df[col].dropna().astype(float)
        feats += [s.mean(), s.std(), s.min(), s.max()]
    return feats


def label_from_path(path: Path, data_root: Path) -> str:
    """Derive label from the top-level gesture folder name.
    e.g. data/gesture_U/contributor/timestamp/Accelerometer.csv -> 'U'
    """
    rel = path.relative_to(data_root)
    gesture_folder = rel.parts[0]          # e.g. 'gesture_U'
    return gesture_folder.replace("gesture_", "")


def load_dataset(dirs: list[Path]):
    X, y = [], []
    skipped = 0
    for root in dirs:
        if not root.is_dir():
            continue
        for csv_path in sorted(root.rglob("Accelerometer*.csv")):
            if "Uncalibrated" in csv_path.name:
                continue
            try:
                feats = extract_features(csv_path)
                label = label_from_path(csv_path, root)
                X.append(feats)
                y.append(label)
            except Exception as exc:
                print(f"  WARNING: skipping {csv_path.name}: {exc}", file=sys.stderr)
                skipped += 1
    if skipped:
        print(f"  ({skipped} file(s) skipped due to errors)")
    return np.array(X, dtype=np.float32), np.array(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--aug-dir",  type=Path, default=Path("data_augmented"))
    ap.add_argument("--seed",     type=int,  default=42)
    ap.add_argument("--test-size",type=float,default=0.2)
    ap.add_argument("--n-trees",  type=int,  default=100)
    args = ap.parse_args()

    print("Loading dataset...")
    dirs = [args.data_dir, args.aug_dir]
    X, y = load_dataset(dirs)

    if len(X) == 0:
        print("ERROR: no samples loaded. Check that data/ exists and contains Accelerometer.csv files.")
        sys.exit(1)

    classes, counts = np.unique(y, return_counts=True)
    print(f"  Loaded {len(X)} samples across {len(classes)} classes:")
    for cls, cnt in zip(classes, counts):
        print(f"    {cls}: {cnt}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    print(f"\nTrain: {len(X_train)}  |  Test: {len(X_test)}")

    print(f"\nTraining Random Forest ({args.n_trees} trees)...")
    clf = RandomForestClassifier(n_estimators=args.n_trees, random_state=args.seed, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = (y_pred == y_test).mean()
    print(f"\nTest accuracy: {acc:.3f}\n")

    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix (rows=actual, cols=predicted):")
    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    header = "        " + "  ".join(f"{c:>6}" for c in clf.classes_)
    print(header)
    for cls, row in zip(clf.classes_, cm):
        print(f"  {cls:>4}  " + "  ".join(f"{v:>6}" for v in row))

    print("\nTop feature importances (mean/std/min/max × x/y/z):")
    feat_names = [f"{stat}_{axis}" for axis in ["x","y","z"]
                                   for stat in ["mean","std","min","max"]]
    importances = sorted(zip(feat_names, clf.feature_importances_),
                         key=lambda t: t[1], reverse=True)
    for name, imp in importances[:6]:
        print(f"  {name:<12} {imp:.4f}")


if __name__ == "__main__":
    main()
