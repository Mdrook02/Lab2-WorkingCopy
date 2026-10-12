#!/usr/bin/env python3
"""
train_demo.py — Gesture classifier demo for 3-axis accelerometer data.

Loads raw Accelerometer.csv files and, if present, their augmented variants.
It splits raw recordings by capture folder before training; augmented files
are included in training only when their source recording is in the training
split. Evaluation uses only held-out raw recordings.

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


def load_dataset(data_dir: Path, aug_dir: Path):
    raw_X, raw_y, raw_groups = [], [], []
    aug_X, aug_y, aug_groups = [], [], []
    skipped = 0
    for root, is_augmented in ((data_dir, False), (aug_dir, True)):
        if not root.is_dir():
            continue
        pattern = "Accelerometer__*.csv" if is_augmented else "Accelerometer.csv"
        for csv_path in sorted(root.rglob(pattern)):
            try:
                feats = extract_features(csv_path)
                label = label_from_path(csv_path, root)
                group = str(csv_path.relative_to(root).parent)
                if is_augmented:
                    aug_X.append(feats)
                    aug_y.append(label)
                    aug_groups.append(group)
                else:
                    raw_X.append(feats)
                    raw_y.append(label)
                    raw_groups.append(group)
            except Exception as exc:
                print(f"  WARNING: skipping {csv_path.name}: {exc}", file=sys.stderr)
                skipped += 1
    if skipped:
        print(f"  ({skipped} file(s) skipped due to errors)")
    return (
        np.array(raw_X, dtype=np.float32),
        np.array(raw_y),
        np.array(raw_groups),
        np.array(aug_X, dtype=np.float32),
        np.array(aug_y),
        np.array(aug_groups),
    )


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
    X_raw, y_raw, raw_groups, X_aug, y_aug, aug_groups = load_dataset(
        args.data_dir, args.aug_dir
    )

    if len(X_raw) == 0:
        print("ERROR: no samples loaded. Check that data/ exists and contains Accelerometer.csv files.")
        sys.exit(1)

    group_labels = {}
    for group, label in zip(raw_groups, y_raw):
        if group in group_labels and group_labels[group] != label:
            print(f"ERROR: capture group {group!r} has conflicting labels.", file=sys.stderr)
            sys.exit(1)
        group_labels[group] = label

    unique_groups = np.array(sorted(group_labels))
    labels_by_group = np.array([group_labels[group] for group in unique_groups])
    classes, group_counts = np.unique(labels_by_group, return_counts=True)
    if len(group_counts) == 0 or group_counts.min() < 2:
        print("ERROR: need at least two raw capture groups per class for a stratified split.", file=sys.stderr)
        sys.exit(1)

    train_groups, test_groups = train_test_split(
        unique_groups,
        test_size=args.test_size,
        random_state=args.seed,
        stratify=labels_by_group,
    )
    train_group_set = set(train_groups)
    test_group_set = set(test_groups)

    raw_train_mask = np.array([group in train_group_set for group in raw_groups])
    raw_test_mask = np.array([group in test_group_set for group in raw_groups])
    aug_train_mask = np.array([group in train_group_set for group in aug_groups])

    X_train_parts = [X_raw[raw_train_mask]]
    y_train_parts = [y_raw[raw_train_mask]]
    if aug_train_mask.any():
        X_train_parts.append(X_aug[aug_train_mask])
        y_train_parts.append(y_aug[aug_train_mask])

    X_train = np.concatenate(X_train_parts)
    y_train = np.concatenate(y_train_parts)
    X_test = X_raw[raw_test_mask]
    y_test = y_raw[raw_test_mask]

    unmatched_augmented = len(aug_groups) - int(aug_train_mask.sum()) - sum(
        group in test_group_set for group in aug_groups
    )
    print(f"  Raw recordings: {len(X_raw)} across {len(unique_groups)} capture groups")
    print(f"  Augmented recordings found: {len(X_aug)}")
    if unmatched_augmented:
        print(f"  Augmented recordings without a matching raw capture: {unmatched_augmented} (excluded)")
    print(f"  Classes: {', '.join(classes)}")
    print(f"  Training groups: {len(train_groups)}  |  Test groups: {len(test_groups)}")
    print(f"  Training samples (raw + training-group augmentations): {len(X_train)}")
    print(f"  Test samples (held-out raw only): {len(X_test)}")

    if not len(X_test):
        print("ERROR: no held-out raw recordings available for testing.", file=sys.stderr)
        sys.exit(1)

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
