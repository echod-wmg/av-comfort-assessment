"""``avcomfort`` command line interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .assess import AssessOptions, assess, summarise
from .model import ComfortModel, default_model


def _common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--out", type=Path, default=Path("results"), help="output directory (default: results)")
    p.add_argument("--model", type=Path, help="custom comfort model JSON (see `avcomfort model`)")
    p.add_argument("--jerk-window", type=float, default=1.0, help="smoothing window for jerk [s] (default 1.0)")
    p.add_argument("--accel-window", type=float, default=0.0, help="smoothing window for acceleration [s] (default 0)")
    p.add_argument("--no-jerk", action="store_true", help="classify on acceleration only")
    p.add_argument("--legacy", action="store_true", help="AssessTool-V1 behaviour: raw acceleration, no jerk")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--basemap", action="store_true", help="draw a web map under lon/lat routes (needs contextily)")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="avcomfort", description="Occupant comfort assessment from vehicle IMU data")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("nuscenes", help="assess a nuScenes CAN bus scene")
    p.add_argument("--can-bus", type=Path, required=True, help="directory with scene-XXXX_ms_imu.json / _pose.json")
    p.add_argument("--scene", required=True, help="scene number or name, e.g. 61 or scene-0061")
    _common(p)

    p = sub.add_parser("berry", help="assess a drive logged by the BerryGPS-IMU DAU")
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--scene", required=True, help="file prefix, e.g. Scene02")
    _common(p)

    p = sub.add_parser("csv", help="assess a CSV with time_s, acc_lat, acc_long [, lon, lat | x_m, y_m]")
    p.add_argument("path", type=Path)
    _common(p)

    p = sub.add_parser("model", help="print the default comfort model as JSON (to customise)")

    args = parser.parse_args(argv)
    if args.command == "model":
        print(json.dumps(default_model().to_dict(), indent=2))
        return 0

    from . import io

    if args.command == "nuscenes":
        drive = io.load_nuscenes(args.can_bus, args.scene)
        name = io._scene_name(args.scene)
    elif args.command == "berry":
        drive = io.load_berry_scene(args.dir, args.scene)
        name = args.scene
    else:
        drive = io.load_csv(args.path)
        name = args.path.stem

    model = ComfortModel.from_json(args.model) if args.model else default_model()
    if args.legacy:
        opts = AssessOptions.legacy()
    else:
        opts = AssessOptions(args.accel_window, args.jerk_window, not args.no_jerk)
    result = assess(drive, model, opts)
    summary = summarise(result, model)
    summary["source"] = drive.attrs.get("source", str(name))
    summary["options"] = vars(opts)

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    result.to_csv(out / f"{name}_comfort.csv", index=False, float_format="%.5g")
    (out / f"{name}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))

    if not args.no_plots:
        import matplotlib

        matplotlib.use("Agg")
        from .plots import report_figure, route_map

        fig = report_figure(
            result, model, title=f"{name} - mean score {summary['mean_score']} ({summary['overall_rating']})"
        )
        fig.savefig(out / f"{name}_report.png", dpi=120)
        if args.basemap and {"lon", "lat"} <= set(result.columns):
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(9, 9))
            route_map(result, model, ax=ax, basemap=True)
            fig.savefig(out / f"{name}_map.png", dpi=120)
    print(f"results written to {out}/")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
