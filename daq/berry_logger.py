#!/usr/bin/env python3
"""Real-time comfort logger for the Raspberry Pi + BerryGPS-IMU data acquisition unit.

This is a cleaned-up version of the 2022 on-vehicle script. It reads the
LSM6DSL accelerometer through OzzMaker's ``IMU.py`` driver, takes GPS fixes
from ``gpsd`` in a background thread (the 2022 script blocked the IMU loop
while it waited for GPS), assesses comfort live with
:class:`avcomfort.RealtimeAssessor`, and writes one CSV file per drive. You
can re-assess that file offline with ``avcomfort csv``.

Setup on the Pi:

1. Enable I2C, then install gpsd and the Python client:
   ``sudo apt install gpsd gpsd-clients python3-gps python3-smbus``.
2. Copy ``IMU.py``, ``LSM6DSL.py``, ``LIS3MDL.py``, ``LSM9DS0.py`` and
   ``LSM9DS1.py`` from https://github.com/ozzmaker/BerryIMU
   (python-BerryIMU-gryo-accel-compass) next to this file. They are not
   redistributed here.
3. ``pip install -e .`` (from the repository root), then run
   ``python3 daq/berry_logger.py --out ~/drives``.

The hardware path has not been re-tested since the 2022 refactor. The
assessment logic is covered by the unit tests, and ``--simulate`` runs the
full loop without hardware.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

from avcomfort import RealtimeAssessor

G = 9.80665
#: LSM6DSL sensitivity at +/-8 g full scale (IMU.py sets CTRL1_XL = 0b10011111)
MG_PER_LSB = 0.244

FIELDS = [
    "unix_time",
    "time_s",
    "acc_lat",
    "acc_long",
    "acc_vert",
    "jerk_lat",
    "jerk_long",
    "level",
    "rating",
    "score",
    "lon",
    "lat",
    "speed_mps",
    "alt_m",
]


class GpsReader(threading.Thread):
    """Keeps the latest gpsd TPV fix without blocking the IMU loop."""

    def __init__(self):
        super().__init__(daemon=True)
        self.fix = {"lon": math.nan, "lat": math.nan, "speed_mps": math.nan, "alt_m": math.nan}

    def run(self):  # pragma: no cover - needs gpsd
        from gps import WATCH_ENABLE, WATCH_NEWSTYLE, gps

        session = gps(mode=WATCH_ENABLE | WATCH_NEWSTYLE)
        for report in session:
            if report.get("class") == "TPV":
                self.fix = {
                    "lon": getattr(report, "lon", math.nan),
                    "lat": getattr(report, "lat", math.nan),
                    "speed_mps": getattr(report, "speed", math.nan),
                    "alt_m": getattr(report, "alt", math.nan),
                }


class BerryIMU:  # pragma: no cover - needs hardware
    def __init__(self):
        import IMU  # OzzMaker driver, see module docstring

        IMU.detectIMU()
        if IMU.BerryIMUversion == 99:
            sys.exit("No BerryIMU found")
        IMU.initIMU()
        self._imu = IMU

    def read_g(self):
        imu = self._imu
        return tuple(v * MG_PER_LSB / 1000.0 for v in (imu.readACCx(), imu.readACCy(), imu.readACCz()))


class SimulatedIMU:
    """Smooth random manoeuvres, for testing the logger without hardware."""

    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        self.t0 = time.time()

    def read_g(self):
        t = time.time() - self.t0
        lat = 0.05 * math.sin(2 * math.pi * t / 7) + self.rng.gauss(0, 0.005)
        lon = 0.03 * math.sin(2 * math.pi * t / 11) + self.rng.gauss(0, 0.005)
        return lat, lon, 1.0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", type=Path, default=Path("drives"))
    p.add_argument("--rate", type=float, default=10.0, help="sample rate in Hz (default 10, the GPS rate)")
    p.add_argument("--lat-axis", choices="xyz", default="x", help="IMU axis pointing to the vehicle's left")
    p.add_argument("--long-axis", choices="xyz", default="y", help="IMU axis pointing forward")
    p.add_argument("--flip-lat", action="store_true")
    p.add_argument("--flip-long", action="store_true")
    p.add_argument("--jerk-window", type=float, default=1.0)
    p.add_argument("--duration", type=float, default=0.0, help="stop after N seconds (0 = until Ctrl-C)")
    p.add_argument("--simulate", action="store_true", help="no hardware: synthetic IMU, no GPS")
    args = p.parse_args(argv)

    imu = SimulatedIMU() if args.simulate else BerryIMU()
    gps = GpsReader()
    if not args.simulate:
        gps.start()
    axes = {"x": 0, "y": 1, "z": 2}
    i_lat, i_long = axes[args.lat_axis], axes[args.long_axis]
    i_vert = ({0, 1, 2} - {i_lat, i_long}).pop()
    s_lat = -1.0 if args.flip_lat else 1.0
    s_long = -1.0 if args.flip_long else 1.0

    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{datetime.now():%Y-%m-%d_%H-%M-%S}_drive.csv"
    assessor = RealtimeAssessor(jerk_window_s=args.jerk_window)
    period = 1.0 / args.rate
    t_start = time.time()
    print(f"logging to {path} (Ctrl-C to stop)")
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        try:
            while True:
                now = time.time()
                t = now - t_start
                g = imu.read_g()
                acc_lat, acc_long = s_lat * g[i_lat] * G, s_long * g[i_long] * G
                acc_vert = g[i_vert] * G - G
                s = assessor.update(t, acc_lat, acc_long)
                row = {
                    "unix_time": f"{now:.3f}",
                    "time_s": f"{t:.3f}",
                    "acc_lat": f"{acc_lat:.4f}",
                    "acc_long": f"{acc_long:.4f}",
                    "acc_vert": f"{acc_vert:.4f}",
                    "jerk_lat": f"{s.jerk_lat:.4f}",
                    "jerk_long": f"{s.jerk_long:.4f}",
                    "level": s.level,
                    "rating": s.rating,
                    "score": f"{s.score:.2f}",
                    **gps.fix,
                }
                writer.writerow(row)
                fh.flush()
                print(f"t={t:7.1f}s  lat={acc_lat:+5.2f}  long={acc_long:+5.2f}  {s.rating:>2}  score={s.score:5.1f}")
                if args.duration and t >= args.duration:
                    break
                time.sleep(max(0.0, period - (time.time() - now)))
        except KeyboardInterrupt:
            pass
    print(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
