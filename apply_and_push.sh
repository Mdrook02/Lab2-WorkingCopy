#!/usr/bin/env bash
# apply_and_push.sh
# Run from the repo root (embeddedai-fall2026-lab2-dataset-group-5-main/).
# Usage:  bash apply_and_push.sh /path/to/train_demo.py /path/to/requirements.txt
#
# What it does:
#   1. Removes 30 Windows Zone.Identifier sidecar files from git tracking.
#   2. Removes 3 stray CSVs directly under data/gesture_3/3GesturesECEN5003/
#      (they have no sibling Accelerometer.csv — misplaced during capture).
#   3. Adds a .gitignore rule so Zone.Identifier files can't creep back.
#   4. Copies the new train_demo.py and updated requirements.txt into the repo.
#   5. Commits and pushes everything.

set -euo pipefail

TRAIN_DEMO_SRC="${1:-train_demo.py}"
REQUIREMENTS_SRC="${2:-requirements.txt}"

# ── Guard: must be at repo root ───────────────────────────────────────────────
if [[ ! -d data ]] || [[ ! -f scripts/augment.py ]]; then
  echo "ERROR: run this from the repo root (where data/ and scripts/ live)."
  exit 1
fi

# ── 1. Remove Zone.Identifier files ──────────────────────────────────────────
echo "==> Removing Zone.Identifier sidecar files …"

ZONE_FILES=(
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-22/Accelerometer.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-22/AccelerometerUncalibrated.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-22/Annotation.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-22/Metadata.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-22/Tags.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-22/TotalAcceleration.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-31/Accelerometer.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-31/AccelerometerUncalibrated.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-31/Annotation.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-31/Metadata.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-31/Tags.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-31/TotalAcceleration.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-37/Accelerometer.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-37/AccelerometerUncalibrated.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-37/Annotation.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-37/Metadata.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-37/Tags.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-37/TotalAcceleration.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-44/Accelerometer.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-44/AccelerometerUncalibrated.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-44/Annotation.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-44/Metadata.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-44/Tags.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-44/TotalAcceleration.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-54/Accelerometer.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-54/AccelerometerUncalibrated.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-54/Annotation.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-54/Metadata.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-54/Tags.csv:Zone.Identifier"
  "data/gesture_3/Jagi4310_Gesture_3/2026-10-06_17-24-54/TotalAcceleration.csv:Zone.Identifier"
)

for f in "${ZONE_FILES[@]}"; do
  if git ls-files --error-unmatch -- "$f" &>/dev/null; then
    git rm --cached -f -- "$f"
    rm -f -- "$f"
  fi
done
echo "    Done (${#ZONE_FILES[@]} files)."

# ── 2. Remove stray CSVs from 3GesturesECEN5003/ ─────────────────────────────
echo "==> Removing stray CSVs from data/gesture_3/3GesturesECEN5003/ …"

STRAY_FILES=(
  "data/gesture_3/3GesturesECEN5003/AccelerometerUncalibrated.csv"
  "data/gesture_3/3GesturesECEN5003/Annotation.csv"
  "data/gesture_3/3GesturesECEN5003/Metadata.csv"
)

for f in "${STRAY_FILES[@]}"; do
  if git ls-files --error-unmatch -- "$f" &>/dev/null; then
    git rm --cached -f -- "$f"
    rm -f -- "$f"
  fi
done
echo "    Done (${#STRAY_FILES[@]} files)."

# ── 3. Update .gitignore ──────────────────────────────────────────────────────
echo "==> Updating .gitignore …"
if ! grep -qF "Zone.Identifier" .gitignore 2>/dev/null; then
  printf '\n# Windows Zone.Identifier sidecar files (WSL artefacts)\n*:Zone.Identifier\n' >> .gitignore
  echo "    Added Zone.Identifier rule."
else
  echo "    Rule already present — no change."
fi

# ── 4. Copy new files into repo ───────────────────────────────────────────────
echo "==> Copying train_demo.py …"
cp -- "$TRAIN_DEMO_SRC" train_demo.py

echo "==> Copying requirements.txt …"
cp -- "$REQUIREMENTS_SRC" requirements.txt

# ── 5. Stage and commit ───────────────────────────────────────────────────────
echo "==> Staging …"
git add train_demo.py requirements.txt .gitignore

echo "==> Committing …"
git commit -m "$(cat <<'EOF'
Add train_demo.py, update requirements, remove junk files

- Add train_demo.py: loads Accelerometer.csv files from data/ (and
  data_augmented/ if present), extracts 12-dim statistical features
  (mean/std/min/max per axis), trains a RandomForestClassifier, and
  prints accuracy + classification report + feature importances.
- Add scikit-learn>=1.3 and pandas>=2.0 to requirements.txt.
- Remove 30 Windows Zone.Identifier sidecar files committed from WSL.
- Remove 3 stray CSVs from data/gesture_3/3GesturesECEN5003/ (no
  sibling Accelerometer.csv; misplaced during capture).
- Add *:Zone.Identifier to .gitignore so they can't creep back.

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01RynhceLS4uxeghzwcmUA1e
EOF
)"

# ── 6. Push ───────────────────────────────────────────────────────────────────
echo "==> Pushing …"
git push

echo ""
echo "All done."
