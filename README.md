# avcomfort: occupant comfort assessment for autonomous vehicles

[![tests](https://github.com/echod-wmg/av-comfort-assessment/actions/workflows/tests.yml/badge.svg)](https://github.com/echod-wmg/av-comfort-assessment/actions/workflows/tests.yml)

`avcomfort` rates ride comfort from vehicle IMU data. It classifies every
sample by **acceleration and jerk** into seven comfort levels (A* to F, after
the VeriCAV occupant-comfort scale), computes a **0–100 comfort score**, and
summarises a whole drive. It reads:

* **nuScenes** autonomous-vehicle drives (CAN bus expansion: 100 Hz IMU and pose),
* drives logged by a **Raspberry Pi + BerryGPS-IMU** data acquisition unit
  (DAU), including real-time assessment on the vehicle,
* any **CSV** with `time_s, acc_lat, acc_long`.

The tool started as my MSc dissertation at the University of Bath (2022),
*occupant comfort assessment of autonomous vehicles*. The original 2022
scripts are referred to here as AssessTool-V1. The full dissertation is in
[`docs/report/`](docs/report/2022_MSc_dissertation_av_comfort.pdf). This version (v2) is a
cleaned-up, tested package. It reproduces V1's
published outputs exactly, and fixes the issues listed under
[Corrections](#corrections-to-v1).

![nuScenes scene-0103 report](docs/img/nuscenes_0103_report.png)

*nuScenes scene-0103 (Boston Seaport). Each sample is placed on the
acceleration comfort regions and the route is coloured by comfort level. The
two black clusters near −2.8 m/s² are firm braking; they rate E.*

## Install

```bash
git clone https://github.com/echod-wmg/av-comfort-assessment.git
cd av-comfort-assessment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # add ,maps for web-map backgrounds (contextily)
```

## Usage

```bash
# nuScenes: point at the extracted CAN bus expansion folder
avcomfort nuscenes --can-bus ~/data/nuscenes/can_bus --scene 103 --out results

# a drive logged by the BerryGPS-IMU DAU (2022 file layout: Scene02-Lateral.txt, ...)
avcomfort berry --dir ~/drives/bath --scene Scene02

# any CSV (time_s, acc_lat, acc_long [, lon, lat | x_m, y_m, speed_mps])
avcomfort csv tests/data/bath_scene02.csv

# options
avcomfort csv drive.csv --jerk-window 0.5     # jerk smoothing window [s]
avcomfort csv drive.csv --legacy              # exact AssessTool-V1 behaviour
avcomfort model > my_model.json               # dump thresholds, edit, then --model my_model.json
```

Each run writes `<scene>_comfort.csv` (per-sample levels and scores),
`<scene>_summary.json` and `<scene>_report.png`.

```python
from avcomfort import load_nuscenes, assess, summarise

drive = load_nuscenes("~/data/nuscenes/can_bus", 103)
result = assess(drive)              # adds jerk, level, rating, score per sample
summarise(result)                   # {'mean_score': 68.49, 'overall_rating': 'C', 'time_share_pct': {...}, ...}
```

## Comfort model

Each level is a rhombus ("kite") in the lateral/longitudinal plane, so
lateral, accelerating and braking motion get different limits. A sample takes
the first level whose acceleration kite **and** jerk kite both contain it. In
practice that is the worse of the two.

| Level | Rating | Lateral [m/s²] | Accel [m/s²] | Braking [m/s²] | Jerk [m/s³] | Score base |
|---|---|---|---|---|---|---|
| Extremely comfortable | A* | < 0.3 | < 0.3 | < 0.3 | < 0.2 | 100 |
| Very comfortable | A | < 0.6 | < 0.6 | < 0.6 | < 0.4 | 90 |
| Comfortable | B | < 0.9 | < 0.9 | < 0.9 | < 0.6 | 80 |
| Acceptable | C | < 3.6 | < 1.3 | < 1.8 | < 0.8 | 75 |
| Poor | D | < 4.2 | < 1.5 | < 2.4 | < 0.95 | 65 |
| Uncomfortable | E | < 5.6 | < 3.07 | < 5.08 | < 2.0 | 60 |
| Extremely uncomfortable | F | beyond | | | | 35 |

The thresholds come from Bae et al. (2019), Karjanto et al. (2017) and VeriCAV
(Connected Places Catapult, 2021). Within a level the score falls with the
total acceleration:
`score = base − points · (|a_lat| + |a_long|) / divisor`.
The level-specific `points` and `divisor` (with a separate braking divisor)
are in `avcomfort/model.py`. The drive's overall rating comes from its
time-weighted mean score: >90 A*, >80 A, >75 B, >65 C, >60 D, >35 E, else F.
Run `avcomfort model` to see the full parameter set.

## Corrections to V1

| Issue in AssessTool-V1 | Evidence | v2 |
|---|---|---|
| **nuScenes axes swapped.** IMU `x` (forward) was classified against the *lateral* limits and `y` (left) against the *longitudinal* ones. The plotting scripts used the right axes, so the two disagreed. | Over scenes 0061, 0103 and 0916, IMU `x` correlates 0.95–0.99 with d(speed)/dt and IMU `y` correlates 0.94–0.98 with yaw-rate × speed. | `load_nuscenes` maps `x` to longitudinal and `y` to lateral. |
| **Jerk never used for nuScenes.** It was hard-coded to 0. Raw differentiation (as in the DAU script) gives a median of about 5–7 m/s³ from sensor noise, above every jerk threshold. | Raw 100 Hz IMU data | Jerk is the derivative of a 1 s moving average (median about 0.2–0.3 m/s³). Adjust with `--jerk-window`. |
| GPS (50 Hz) and IMU (100 Hz) samples weren't aligned, so comfort couldn't be mapped to position. | Comment in `AV_nuscenes.py` | Pose is interpolated onto the IMU timestamps. |
| Hard-coded Windows paths, output files opened in append mode (re-runs duplicated rows), thresholds copy-pasted across 4 scripts | | One package, CLI, JSON-configurable model, tests |
| Dissertation Table 3 lists lateral C/D limits of 3.7 / 4.3 m/s²; the code used 3.6 / 4.2. The text says ±2 g accelerometer range; the driver configured ±8 g (0.244 mg/LSB). | `IMU.py` sets `CTRL1_XL = 0b10011111` | Code values kept, so published results reproduce. Both are documented here. |

Effect on the three nuScenes scenes from the dissertation (`docs/make_figures.py`):

| Scene | V1 (2022) | V1 + axis fix | v2 default (axis fix + jerk) |
|---|---|---|---|
| scene-0061 Singapore One-North | 77.5 (B) | 77.5 (B) | 71.3 (C) |
| scene-0103 Boston Seaport | 78.0 (B) | 76.2 (B) | 68.5 (C) |
| scene-0916 Singapore Queenstown | 72.5 (C) | 74.7 (C) | 65.4 (C) |

The axis fix changes the per-sample classification considerably even where
the mean barely moves. In scene-0103, samples rated E (uncomfortable) go from
5 to 212 because hard braking is now judged against the braking limits. Adding
jerk lowers the mean scores by a further 6–9 points.

`tests/test_assess.py::test_reproduces_assesstool_v1_exactly` checks that
`--legacy` reproduces V1's per-sample levels and scores bit-for-bit, using the
Bath scene-02 drive in `tests/data/`.

## On-vehicle data acquisition unit

![Bath test drive](docs/img/bath_scene02_gg.png)

The DAU was a Raspberry Pi 4B with an OzzMaker BerryGPS-IMU v4 (LSM6DSL
accelerometer, u-blox CAM-M8 GPS, 10 Hz), an external GPS antenna, a USB
camera and a 3-D printed case. I used it on a Honda Civic around Bath (UK).
[`daq/berry_logger.py`](daq/berry_logger.py) is the cleaned-up logger. It
assesses comfort live, reads GPS in a background thread, and writes one CSV
per drive that `avcomfort csv` can re-assess. Run it with `--simulate` to try
it without hardware. The setup steps are in the file header.

## Data

* **nuScenes** is not redistributed here (CC BY-NC-SA 4.0). Download the
  *CAN bus expansion* from <https://www.nuscenes.org/download> and pass its
  folder to `--can-bus`. Scenes 0061, 0103 and 0916 were used in the
  dissertation. The figures above are derived from nuScenes data,
  © Motional, used under CC BY-NC-SA 4.0.
* `tests/data/bath_scene02.csv` is a 50 s extract of my own test drive
  (acceleration and speed only; GPS removed).

## Limitations

* Only horizontal acceleration and jerk are scored. Vertical vibration
  (ISO 2631), motion sickness, noise and thermal comfort are not.
* The score formula and thresholds are heuristic, synthesised from the
  literature. They have not been calibrated against occupant ratings.
* The IMU must be aligned with the vehicle axes. The logger has axis and sign
  flags for the mounting, but no automatic alignment.

## Dissertation

*Design of the model for the real-time occupant's comfort assessment within
the autonomous vehicle*, MSc dissertation, University of Bath, 2022:
[PDF](docs/report/2022_MSc_dissertation_av_comfort.pdf). It covers the
literature review, the DAU design, the comfort model and the original V1
results. Some literature figures are reproduced from the cited sources.

## License

Code: [MIT](LICENSE). nuScenes-derived figures: CC BY-NC-SA 4.0 (see Data).
