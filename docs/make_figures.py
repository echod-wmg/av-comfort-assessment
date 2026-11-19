"""Regenerate the README figures and the results table.

    python docs/make_figures.py /path/to/nuscenes/can_bus

nuScenes data is not redistributed in this repository (CC BY-NC-SA 4.0);
download the CAN bus expansion from https://www.nuscenes.org/download.
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from avcomfort import assess, load_csv, load_nuscenes, summarise
from avcomfort.plots import gg_diagram, report_figure

ROOT = Path(__file__).resolve().parents[1]
IMG = ROOT / "docs" / "img"
HERO = 446  # S-bend: comfortable on the straights, uncomfortable in the corners
SCENES = {
    61: "Singapore One-North",
    103: "Boston Seaport",
    446: None,
    916: "Singapore Queenstown",
}


def main(can_bus: Path) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    rows = []
    for scene, place in SCENES.items():
        result = assess(load_nuscenes(can_bus, scene))
        s = summarise(result)
        rows.append(
            {
                "scene": f"scene-{scene:04d}" + (f" ({place})" if place else ""),
                "mean score": s["mean_score"],
                "rating": s["overall_rating"],
                "time in C or worse [%]": round(sum(v for k, v in s["time_share_pct"].items() if k in "CDEF"), 1),
            }
        )
        if scene == HERO:
            fig = report_figure(result, title=f"nuScenes scene-{scene:04d}")
            fig.savefig(IMG / f"nuscenes_{scene:04d}_report.png", dpi=100)
            plt.close(fig)
    table = pd.DataFrame(rows)
    print(table.to_string(index=False))
    table.to_csv(ROOT / "docs" / "nuscenes_results.csv", index=False)

    bath = assess(load_csv(ROOT / "tests" / "data" / "bath_scene02.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    gg_diagram(bath, ax=ax, title="Bath test drive, scene 02 (BerryGPS-IMU DAU)")
    fig.tight_layout()
    fig.savefig(IMG / "bath_scene02_gg.png", dpi=110)
    print(json.dumps(summarise(bath), indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(Path(sys.argv[1]))
