"""Regenerate README figures and the results table.

    python docs/make_figures.py /path/to/nuscenes/can_bus

nuScenes data is not redistributed in this repository (CC BY-NC-SA 4.0);
download the CAN bus expansion from https://www.nuscenes.org/download.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

from avcomfort import AssessOptions, assess, load_csv, load_nuscenes, summarise
from avcomfort.plots import gg_diagram, report_figure

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
SCENES = {61: "Singapore One-North", 103: "Boston Seaport", 916: "Singapore Queenstown"}


def v1_swapped(drive: pd.DataFrame) -> pd.DataFrame:
    """Reproduce AssessTool-V1's axis mix-up (IMU x treated as lateral)."""
    d = drive.copy()
    d["acc_lat"], d["acc_long"] = drive["acc_long"].to_numpy(), drive["acc_lat"].to_numpy()
    return d


def main(can_bus: Path) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene, place in SCENES.items():
        drive = load_nuscenes(can_bus, scene)
        variants = {
            "V1 (2022)": assess(v1_swapped(drive), options=AssessOptions.legacy()),
            "V1 + axis fix": assess(drive, options=AssessOptions.legacy()),
            "v2 default (axis fix + jerk)": assess(drive),
        }
        for label, res in variants.items():
            s = summarise(res)
            rows.append(
                {
                    "scene": f"scene-{scene:04d} ({place})",
                    "method": label,
                    "mean score": s["mean_score"],
                    "rating": s["overall_rating"],
                    "time in C or worse [%]": round(sum(v for k, v in s["time_share_pct"].items() if k in "CDEF"), 1),
                }
            )
        if scene == 103:
            fig = report_figure(variants["v2 default (axis fix + jerk)"], title="nuScenes scene-0103 (Boston Seaport)")
            fig.savefig(IMG / "nuscenes_0103_report.png", dpi=100)
    table = pd.DataFrame(rows)
    try:
        print(table.to_markdown(index=False))  # needs the optional 'tabulate' package
    except ImportError:
        print(table.to_string(index=False))
    table.to_csv(ROOT / "docs" / "nuscenes_results.csv", index=False)

    bath = assess(load_csv(ROOT / "tests" / "data" / "bath_scene02.csv"))
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    gg_diagram(bath, ax=ax, title="Bath test drive, scene 02 (BerryGPS-IMU DAU)")
    fig.tight_layout()
    fig.savefig(IMG / "bath_scene02_gg.png", dpi=110)
    print(json.dumps(summarise(bath), indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
