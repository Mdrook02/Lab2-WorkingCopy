# Gesture Recognition Accelerometer Dataset
## Embedded AI Lab 2 — ECEN 5003

A 3-axis accelerometer dataset captured with smartphones for training a
gesture classifier that distinguishes four classes: three air-drawn letter
shapes (**U**, **L**, **3**) and an **idle/background** class.  The dataset
is organized in the style of the TensorFlow Lite "Magic Wand" example and
is intended to be used with the accompanying augmentation and demo-training
scripts.

---

## Table of Contents

1. [Gesture set and goal](#1-gesture-set-and-goal)
2. [Capture hardware and software](#2-capture-hardware-and-software)
3. [Physical capture procedure](#3-physical-capture-procedure)
4. [Repository layout](#4-repository-layout)
5. [Data format](#5-data-format)
6. [Deriving a label](#6-deriving-a-label)
7. [Dataset statistics](#7-dataset-statistics)
8. [Known limitations](#8-known-limitations)
9. [Setup and dependencies](#9-setup-and-dependencies)
10. [Running the augmentation script](#10-running-the-augmentation-script)
11. [Running the demo training script](#11-running-the-demo-training-script)

---

## 1. Gesture set and goal

The dataset contains accelerometer recordings of four classes captured
while holding a smartphone:

| Label | Class | Description |
|-------|-------|-------------|
| `U`   | Letter U | Stroke down, curve right, stroke back up |
| `L`   | Letter L | Stroke down, then stroke right |
| `3`   | Digit 3  | Two stacked rightward arcs |
| `idle` | Idle/background | Phone held stationary; no gesture performed |

The goal is to train a classifier that can distinguish the three active
gestures from each other and from background motion — analogous to the
TensorFlow Lite Magic Wand demo.  The idle class represents the "no gesture
happening" state that a real deployed detector must handle.

---

## 2. Capture hardware and software

**Hardware:** Contributors' personal smartphones (iOS and Android).  No
STM32F4 Discovery board or external IMU was used.

**Primary app:** [SensorLogger](https://www.tszheichoi.com/sensorlogger)
(free, available on iOS and Android).  Default settings were used: the app
streams accelerometer readings at the device's native sampling rate
(typically 100 Hz) and writes them to a timestamped folder containing one
`Accelerometer.csv` per recording session.

**Secondary app:** One contributor (thgh9905) used
[Phyphox](https://phyphox.org/) for their idle-class captures.  Phyphox
produces a different CSV column layout (see §5); both scripts handle this
automatically.

**How SensorLogger creates files:** Each time you tap "Record," the app
creates a new folder named with an ISO-8601 timestamp
(e.g. `2026-10-06_14-23-11`) and writes a fresh `Accelerometer.csv`
inside it.  One tap-record–tap-stop cycle = one gesture repetition = one
CSV file.

---

## 3. Physical capture procedure

- **Contributors:** 7 student contributors across the full dataset (not
  every contributor recorded every class — see §7).
- **Sessions:** Captures were collected across multiple independent sessions
  spanning **2026-10-05 through 2026-10-10**.  Each contributor recorded
  their gestures in at least one sitting; several contributed across
  multiple days, giving natural within-person variation.
- **Repetitions:** Each contributor performed the gesture multiple times,
  one repetition per CSV file (see §7 for per-class counts).
- **Physical technique (gesture classes):** The contributor held the phone
  naturally in one hand, oriented in portrait mode, and drew the gesture
  shape in the air in front of them at a comfortable, conversational speed.
  No specific wrist angle or gesture size was mandated; normal variation in
  speed and amplitude across contributors is intentional and contributes to
  the dataset's representativeness.
- **Physical technique (idle class):** The contributor held the phone
  stationary in hand (as if about to perform a gesture but not moving) and
  recorded a brief window of near-zero motion.
- **Triggering:** The contributor tapped "Record" immediately before
  beginning the motion and tapped "Stop" immediately after completing it.
  Each file therefore contains the accelerometer trace for exactly one
  repetition.

---

## 4. Repository layout

```
gesture-dataset/
├── README.md                 ← this file
├── requirements.txt          ← Python dependencies for both scripts
├── data/
│   ├── gesture_3/            ← class "3"
│   │   ├── <contributor_id>/
│   │   │   ├── <timestamp>/
│   │   │   │   └── Accelerometer.csv
│   │   │   └── ...
│   │   └── ...
│   ├── gesture_L/            ← class "L"
│   │   └── (same structure)
│   ├── gesture_U/            ← class "U"
│   │   └── (same structure)
│   └── gesture_idle/         ← class "idle"
│       └── (same structure)
└── scripts/
    ├── augment.py            ← augmentation script
    └── train_demo.py         ← demo training / evaluation script
```

`data_augmented/` is **not committed** to the repository.  It is generated
locally by running `scripts/augment.py` and contains synthetic recordings
organized by class in flat per-class folders (e.g. `data_augmented/gesture_U/`).

---

## 5. Data format

Each `Accelerometer.csv` uses one of two column layouts depending on which
app captured it.  Both scripts select columns by name, so either format
parses correctly without any manual intervention.

### SensorLogger format (all contributors except thgh9905 idle data)

```
time,seconds_elapsed,z,y,x
1728216191123456789,0.000000,-2.341,8.812,4.203
1728216191133456789,0.010000,-2.287,8.799,4.251
...
```

| Column | Type | Description |
|--------|------|-------------|
| `time` | integer | Unix epoch time in nanoseconds |
| `seconds_elapsed` | float | Seconds since the start of this recording |
| `z` | float | Accelerometer Z-axis, **m/s²** |
| `y` | float | Accelerometer Y-axis, **m/s²** |
| `x` | float | Accelerometer X-axis, **m/s²** |

### Phyphox format (thgh9905 idle recordings only)

```
"Time (s)","Acceleration x (m/s^2)","Acceleration y (m/s^2)","Acceleration z (m/s^2)","Absolute acceleration (m/s^2)"
7.00E-3,-0.132,2.696,9.670,10.040
...
```

| Column | Type | Description |
|--------|------|-------------|
| `Time (s)` | float | Seconds since recording start |
| `Acceleration x (m/s^2)` | float | X-axis, **m/s²** |
| `Acceleration y (m/s^2)` | float | Y-axis, **m/s²** |
| `Acceleration z (m/s^2)` | float | Z-axis, **m/s²** |
| `Absolute acceleration (m/s^2)` | float | Vector magnitude, **m/s²** |

**Notes applicable to both formats:**

- Scripts should select the x, y, z axes **by column name**, not by
  position.  `augment.py` and `train_demo.py` both do this automatically.
- Each file contains approximately 200–400 data rows (≈ 2–4 seconds at
  ~100 Hz), corresponding to one gesture repetition or one idle window.
- There are no comment lines or metadata headers in either format; the
  first line of every file is the column header.
- SensorLogger also writes `AccelerometerUncalibrated.csv`,
  `Annotation.csv`, `Metadata.csv`, and sometimes `TotalAcceleration.csv`
  and `Tags.csv` alongside each `Accelerometer.csv`.  These are not used
  and can be ignored.

---

## 6. Deriving a label

**The label for any recording is the top-level gesture folder name with
the `gesture_` prefix stripped.**

Given the path:

```
data/gesture_U/augo9400/2026-10-06_09-14-52/Accelerometer.csv
```

The label is **`U`** (from `gesture_U`, dropping `gesture_`).

Similarly:

```
data/gesture_idle/Jagi4310_Idle/2026-10-09_16-58-32/Accelerometer.csv
```

→ label **`idle`**

More precisely, when iterating files recursively:

1. Find every file named `Accelerometer.csv` anywhere under `data/`.
2. Walk up its ancestor directories until you reach one whose name begins
   with `gesture_`.
3. Strip the `gesture_` prefix from that directory name to get the label.

In Python:

```python
from pathlib import Path

def get_label(csv_path: Path, data_root: Path = Path("data")) -> str:
    """Return the gesture label for a given Accelerometer.csv path."""
    rel = csv_path.relative_to(data_root)
    gesture_folder = rel.parts[0]          # e.g. 'gesture_U'
    return gesture_folder.replace("gesture_", "")
```

The `contributor_id` and `timestamp` path components carry no label
information and can be ignored for classification purposes.

---

## 7. Dataset statistics

| Class | Label | Recordings | Contributors |
|-------|-------|------------|--------------|
| gesture_3 | `3` | 39 | 3GesturesECEN5003, Jagi4310, Lifa9737, adwa4856, augo9400, madr3349 |
| gesture_L | `L` | 39 | Jagi4310, LGesturesECEN5003, adwa4856, augo9400, Lifa9737, madr3349 |
| gesture_U | `U` | 46 | Jagi4310, Lifa9737, UGesturesECEN5003, adwa4856, augo9400, madr3349 |
| gesture_idle | `idle` | 32 | Jagi4310, madr3349, ahab5661, thgh9905 |
| **Total** | — | **156** | 7 distinct contributors |

Captures span 2026-10-05 through 2026-10-10 across multiple independent
recording sessions.  Not every contributor recorded every class (see §8).

**Note on contributor folder names:** The contributor IDs in this table are
canonical identifiers used for reference.  The actual subfolder names inside
each `gesture_*` directory were chosen by each contributor independently and
may differ in casing or have a class-specific suffix (e.g. `Lifa9737_gesture_3`,
`lifa9737_L_Gesture`, `Lifa9737`).  This has no effect on label derivation,
which depends only on the top-level `gesture_*` folder (see §6).

---

## 8. Known limitations

1. **Uneven contributor coverage across classes:** Not all contributors
   recorded every class.  The gesture_idle class has 4 contributors
   (Jagi4310, madr3349, ahab5661, thgh9905), while the active gesture
   classes have 6 contributors each, with no overlap from ahab5661 or
   thgh9905.  Conversely, adwa4856 and augo9400 have no idle-class data.

2. **Missing active-gesture data (thgh9905):** This contributor submitted
   their gesture_3, gesture_L, and gesture_U data as Excel (`.xlsx`) files,
   which are not compatible with the CSV format used here.  Their active
   gesture recordings are absent from the dataset entirely.  They are
   represented only in the idle class (via Phyphox-format CSVs).

3. **Class imbalance:** `gesture_U` has 46 recordings versus 39 each for
   `gesture_3` and `gesture_L` and 32 for `gesture_idle` (≈ 44% more
   than idle).  A production training pipeline should apply class weights
   or resampling if the imbalance causes issues; the demo script trains
   without correction.

4. **Mixed capture applications:** The majority of recordings use
   SensorLogger; the thgh9905 idle recordings use Phyphox.  The two apps
   produce different CSV column layouts (see §5).  Both scripts handle this
   automatically by selecting columns by name.

5. **Phone-only captures:** All recordings were made with smartphones.  No
   STM32F4 Discovery board data is included.  The axis orientation, sampling
   rate, and sensor noise characteristics may differ from the course board's
   onboard accelerometer.

6. **Single device orientation:** Contributors held the phone in portrait
   mode.  Gestures performed with a different phone orientation (landscape,
   face-down, etc.) may produce substantially different raw traces.

---

## 9. Setup and dependencies

Python 3.8 or later is required.  Create a virtual environment and install
dependencies before running either script:

```bash
python3 -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

`requirements.txt` declares:

```
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
```

All subsequent commands assume the virtual environment is active.

---

## 10. Running the augmentation script

```bash
python3 scripts/augment.py
```

**What it does:** Reads every `Accelerometer.csv` recursively from `data/`
and writes augmented copies into `data_augmented/`, organized into flat
per-class subfolders (e.g. `data_augmented/gesture_U/`).  Four augmentation
techniques are applied:

| Technique | Description |
|-----------|-------------|
| Noise jitter | Adds small Gaussian noise (σ = 0.05 m/s²) to x, y, z values |
| Time-shifting | Drops a few samples from the start and zero-pads the end |
| Amplitude scaling | Multiplies all axes by a random factor in [0.85, 1.15] |
| 3-D rotation | Applies a small random rotation matrix (up to ±15°) to the (x, y, z) vectors |

Three synthetic copies are generated per technique per source file,
producing 12 augmented files per original recording (1 872 files total from
156 source files).

The script requires no arguments.  On re-run it overwrites any previously
generated `data_augmented/` content.

Optional arguments:

```bash
python3 scripts/augment.py --data-dir data --out-dir data_augmented --n 3 --seed 42
```

---

## 11. Running the demo training script

```bash
python3 scripts/train_demo.py
```

**What it does:** Loads all `Accelerometer.csv` files from `data/` and, if
present, from `data_augmented/`.  Derives the label for each file using the
`gesture_*` folder convention described in §6.  Extracts 12 statistical
features per file (mean, std, min, max for each of the x, y, z axes) and
trains a 100-tree Random Forest classifier (scikit-learn).  Prints accuracy,
a full per-class classification report, a confusion matrix, and per-feature
importances to stdout.

For best results, run the augmentation script first:

```bash
python3 scripts/augment.py
python3 scripts/train_demo.py
```

The demo script also works on raw data alone (if `data_augmented/` does not
exist, it trains only on the 156 source recordings across 4 classes).

Optional arguments:

```bash
python3 scripts/train_demo.py --data-dir data --aug-dir data_augmented --seed 42 --test-size 0.2 --n-trees 100
```
