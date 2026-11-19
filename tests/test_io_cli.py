import json
from pathlib import Path

import numpy as np

from avcomfort import load_berry_scene, load_nuscenes
from avcomfort.cli import main

DATA = Path(__file__).parent / "data"


def _write_nuscenes(tmp_path, scene="scene-0042"):
    t0 = 1_532_402_927_000_000
    imu = [
        {"utime": t0 + i * 10_000, "linear_accel": [0.5, -0.2, 9.81], "q": [1, 0, 0, 0], "rotation_rate": [0, 0, 0]}
        for i in range(300)
    ]
    pose = [
        {"utime": t0 + i * 20_000, "pos": [100.0 + i, 200.0, 0.0], "vel": [5.0, 0.0, 0.0], "accel": [0, 0, 0]}
        for i in range(150)
    ]
    (tmp_path / f"{scene}_ms_imu.json").write_text(json.dumps(imu))
    (tmp_path / f"{scene}_pose.json").write_text(json.dumps(pose))


def test_nuscenes_axes_and_interpolation(tmp_path):
    _write_nuscenes(tmp_path)
    df = load_nuscenes(tmp_path, 42)
    # IMU x is forward -> longitudinal, y is left -> lateral
    assert np.allclose(df["acc_long"], 0.5) and np.allclose(df["acc_lat"], -0.2)
    assert df["time_s"].iloc[1] == 0.01
    assert df["x_m"].iloc[2] == 101.0  # 20 ms -> second pose sample
    assert np.allclose(df["speed_mps"], 5.0)


def test_berry_loader(tmp_path):
    rows = [(0.5, 0.1, -0.2), (0.8, 0.2, -0.1), (1.2, 0.3, 0.0)]
    (tmp_path / "S-Lateral.txt").write_text("\n".join(f"{t},{a}" for t, a, _ in rows))
    (tmp_path / "S-Longitudinal.txt").write_text("\n".join(f"{t},{b}" for t, _, b in rows))
    (tmp_path / "S-GPS.txt").write_text("-2.35,51.38,1\n0.0,0.0,1\n-2.36,51.39,1")
    (tmp_path / "S-Speed.txt").write_text("0.5,9.9\n0.8,nan\n1.2,10.1")
    df = load_berry_scene(tmp_path, "S")
    assert df["time_s"].tolist() == [0.0, 0.30000000000000004, 0.7]
    assert np.isnan(df["lon"].iloc[1]) and df["lat"].iloc[2] == 51.39
    assert np.isnan(df["speed_mps"].iloc[1])


def test_cli_nuscenes(tmp_path, capsys):
    _write_nuscenes(tmp_path)
    out = tmp_path / "out"
    assert main(["nuscenes", "--can-bus", str(tmp_path), "--scene", "42", "--out", str(out)]) == 0
    assert (out / "scene-0042_comfort.csv").exists()
    assert (out / "scene-0042_report.png").exists()
    summary = json.loads((out / "scene-0042_summary.json").read_text())
    assert summary["overall_rating"] == "B"  # steady 0.5 / -0.2 m/s^2 -> "comfortable" (score 78)


def test_cli_csv_legacy(tmp_path):
    out = tmp_path / "o"
    assert main(["csv", str(DATA / "bath_scene02.csv"), "--legacy", "--no-plots", "--out", str(out)]) == 0
    s = json.loads((out / "bath_scene02_summary.json").read_text())
    assert s["options"]["use_jerk"] is False and s["samples"] == 101


def test_cli_model_dump(capsys):
    assert main(["model"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert len(data["levels"]) == 7
