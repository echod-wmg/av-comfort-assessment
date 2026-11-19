"""Data loaders. Every loader returns a DataFrame with the standard columns.

``time_s`` (from 0), ``acc_lat`` (m/s^2, + = left), ``acc_long`` (m/s^2,
+ = accelerating) and, when available, ``speed_mps``, ``x_m``/``y_m``
(local map frame) or ``lon``/``lat`` (WGS84).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

PathLike = Union[str, Path]

# Latitude / longitude of each nuScenes map origin (from the nuScenes devkit).
NUSCENES_MAP_ORIGINS = {
    "boston-seaport": (42.336849169438615, -71.05785369873047),
    "singapore-onenorth": (1.2882100868743724, 103.78475189208984),
    "singapore-hollandvillage": (1.2993652317780957, 103.78217697143555),
    "singapore-queenstown": (1.2782562240223188, 103.76741409301758),
}
_EARTH_RADIUS_M = 6378137.0


def load_csv(path: PathLike) -> pd.DataFrame:
    """Generic CSV with at least ``time_s, acc_lat, acc_long`` (``#`` comments allowed)."""
    df = pd.read_csv(path, comment="#")
    df["time_s"] = df["time_s"] - df["time_s"].iloc[0]
    return df


def load_nuscenes(can_bus_dir: PathLike, scene: Union[str, int], location: Optional[str] = None) -> pd.DataFrame:
    """Load one scene of the nuScenes CAN bus expansion.

    Uses ``scene-XXXX_ms_imu.json`` (100 Hz). ``linear_accel`` is in the
    vehicle frame with x forward and y left, so ``x`` is longitudinal and ``y``
    is lateral. AssessTool-V1 had these two swapped. Positions and speed from
    ``scene-XXXX_pose.json`` (50 Hz) are interpolated onto the IMU
    timestamps, so every comfort sample has a location.

    With ``location`` (a key of ``NUSCENES_MAP_ORIGINS``, e.g.
    ``"singapore-queenstown"``) the local positions are also converted to
    ``lon``/``lat``, so the route can be drawn on a web map.
    """
    can_bus_dir = Path(can_bus_dir)
    name = _scene_name(scene)
    imu = _read_json(can_bus_dir / f"{name}_ms_imu.json")
    t_us = np.array([m["utime"] for m in imu], dtype=np.int64)
    acc = np.array([m["linear_accel"] for m in imu], dtype=float)
    t0 = t_us[0]
    df = pd.DataFrame({"time_s": (t_us - t0) / 1e6, "acc_long": acc[:, 0], "acc_lat": acc[:, 1]})
    pose_path = can_bus_dir / f"{name}_pose.json"
    if pose_path.exists():
        pose = _read_json(pose_path)
        tp = (np.array([m["utime"] for m in pose], dtype=np.int64) - t0) / 1e6
        pos = np.array([m["pos"] for m in pose], dtype=float)
        vel = np.array([m["vel"][0] for m in pose], dtype=float)
        df["x_m"] = np.interp(df["time_s"], tp, pos[:, 0])
        df["y_m"] = np.interp(df["time_s"], tp, pos[:, 1])
        df["speed_mps"] = np.interp(df["time_s"], tp, vel)
        if location is not None:
            lat0, lon0 = NUSCENES_MAP_ORIGINS[location]
            df["lat"] = lat0 + np.degrees(df["y_m"] / _EARTH_RADIUS_M)
            df["lon"] = lon0 + np.degrees(df["x_m"] / (_EARTH_RADIUS_M * np.cos(np.radians(lat0))))
    df.attrs["source"] = f"nuScenes CAN bus {name}"
    return df


def load_berry_scene(directory: PathLike, scene: str) -> pd.DataFrame:
    """Load a drive logged by the Raspberry Pi / BerryGPS-IMU DAU (2022 format).

    Expects ``<scene>-Lateral.txt`` and ``<scene>-Longitudinal.txt``
    (``time,value``) and optionally ``<scene>-Speed.txt`` (m/s) and
    ``<scene>-GPS.txt`` (``lon,lat,flag``; ``0,0`` means no fix).
    """
    d = Path(directory)
    lat = np.loadtxt(d / f"{scene}-Lateral.txt", delimiter=",", ndmin=2)
    lon = np.loadtxt(d / f"{scene}-Longitudinal.txt", delimiter=",", ndmin=2)
    if lat.shape[0] != lon.shape[0] or not np.allclose(lat[:, 0], lon[:, 0]):
        raise ValueError("lateral and longitudinal logs are not sample-aligned")
    df = pd.DataFrame({"time_s": lat[:, 0] - lat[0, 0], "acc_lat": lat[:, 1], "acc_long": lon[:, 1]})
    speed = d / f"{scene}-Speed.txt"
    if speed.exists():
        s = np.genfromtxt(speed, delimiter=",", ndmin=2)
        if s.shape[0] == len(df):
            df["speed_mps"] = s[:, 1]
    gps = d / f"{scene}-GPS.txt"
    if gps.exists():
        g = np.genfromtxt(gps, delimiter=",", ndmin=2)
        if g.shape[0] == len(df):
            no_fix = (g[:, 0] == 0) & (g[:, 1] == 0)
            df["lon"] = np.where(no_fix, np.nan, g[:, 0])
            df["lat"] = np.where(no_fix, np.nan, g[:, 1])
    df.attrs["source"] = f"BerryIMU {scene}"
    return df


def _scene_name(scene: Union[str, int]) -> str:
    s = str(scene)
    if s.startswith("scene-"):
        return s
    return f"scene-{int(s):04d}"


def _read_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)
