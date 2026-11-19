"""Plots: g-g classification diagram, score timeline, accelerations, route map."""

from __future__ import annotations

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .model import ComfortModel, default_model


def gg_diagram(result: pd.DataFrame, model: Optional[ComfortModel] = None, ax=None, title: Optional[str] = None):
    """Lateral vs longitudinal acceleration with the comfort regions."""
    model = model or default_model()
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.5))
    for lv in model.levels:
        if lv.accel is None:
            continue
        x, y = lv.accel.vertices()
        ax.plot(x, y, color=lv.colour, linewidth=1.8, label=f"{lv.rating}  {lv.name}")
    ax.scatter(result["acc_lat_f"], result["acc_long_f"], s=4, c="k", alpha=0.5, linewidths=0)
    ax.set_xlabel("Lateral acceleration [m/s²]")
    ax.set_ylabel("Longitudinal acceleration [m/s²]")
    ax.set_title(title or "Lateral / longitudinal acceleration classification")
    ax.axhline(0, color="0.7", lw=0.6)
    ax.axvline(0, color="0.7", lw=0.6)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_aspect("equal", adjustable="datalim")
    return ax


def score_timeline(result: pd.DataFrame, model: Optional[ComfortModel] = None, ax=None, title: Optional[str] = None):
    """Comfort score vs time, shaded by rating band."""
    model = model or default_model()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.8))
    bands = [b for b, _ in model.rating_bands]
    tops = [100.0] + bands[:-1]
    colours = [lv.colour for lv in model.levels]
    y_min = max(0.0, float(result["score"].min()) - 5)
    for (lo, rating), hi, colour in zip(model.rating_bands, tops, colours):
        lo = max(lo, 0.0)
        ax.axhspan(lo, hi, color=colour, alpha=0.12, lw=0)
        if hi > y_min:  # label only the visible part of the band
            mid = (max(lo, y_min) + hi) / 2
            ax.text(1.005, mid, rating, transform=ax.get_yaxis_transform(), va="center", fontsize=8)
    ax.plot(result["time_s"], result["score"], color="mediumblue", lw=1)
    ax.set_ylim(y_min, 100)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Comfort score [0-100]")
    ax.set_title(title or "Comfort score vs time")
    ax.grid(alpha=0.3)
    return ax


def acceleration_timeline(result: pd.DataFrame, ax=None, title: Optional[str] = None):
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.8))
    ax.plot(result["time_s"], result["acc_lat_f"], lw=1, label="lateral")
    ax.plot(result["time_s"], result["acc_long_f"], lw=1, label="longitudinal")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Acceleration [m/s²]")
    ax.set_title(title or "Acceleration vs time")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def route_map(
    result: pd.DataFrame,
    model: Optional[ComfortModel] = None,
    ax=None,
    title: Optional[str] = None,
    basemap: bool = False,
):
    """Route coloured by comfort level.

    Uses ``x_m``/``y_m`` (local metres, e.g. nuScenes) or ``lon``/``lat``. With
    ``basemap=True`` and lon/lat data a web map is drawn underneath via the
    optional ``contextily`` package.
    """
    model = model or default_model()
    if {"lon", "lat"} <= set(result.columns):
        xs, ys, xl, yl, geo = result["lon"], result["lat"], "Longitude [deg]", "Latitude [deg]", True
    elif {"x_m", "y_m"} <= set(result.columns):
        xs, ys, xl, yl, geo = result["x_m"], result["y_m"], "x [m]", "y [m]", False
    else:
        raise ValueError("no position columns (lon/lat or x_m/y_m) in the data")
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 7))
    colours = np.array([lv.colour for lv in model.levels], dtype=object)[result["level"].to_numpy()]
    ax.scatter(xs, ys, c=list(colours), s=9, linewidths=0, zorder=3)
    if basemap and geo:
        try:
            import contextily as cx

            cx.add_basemap(ax, crs="EPSG:4326", source=cx.providers.CartoDB.Voyager)
        except ImportError:  # pragma: no cover
            pass
    handles = [Line2D([], [], marker="o", ls="", color=lv.colour, label=lv.rating) for lv in model.levels]
    ax.legend(handles=handles, fontsize=8, loc="best")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_title(title or "Comfort along the route")
    if not geo:
        ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.3)
    return ax


def report_figure(result: pd.DataFrame, model: Optional[ComfortModel] = None, title: Optional[str] = None):
    """One-page overview: g-g diagram, route (if available), accelerations and score."""
    has_route = {"x_m", "y_m"} <= set(result.columns) or {"lon", "lat"} <= set(result.columns)
    fig = plt.figure(figsize=(13, 9))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1])
    gg_diagram(result, model, ax=fig.add_subplot(gs[0, 0]))
    if has_route:
        route_map(result, model, ax=fig.add_subplot(gs[0, 1]))
    else:
        acceleration_timeline(result, ax=fig.add_subplot(gs[0, 1]))
    score_timeline(result, model, ax=fig.add_subplot(gs[1, :]))
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
