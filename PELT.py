"""
Robust multiscale river segmentation with PELT.

Includes:
- Optional LineString-derived geometry metrics (curvature, sinuosity)
- Missing-aware moving-window slope from WSE
- Automated fixed, width-based, or raw window selection
- PELT per window + penalty sweep
- Breakpoint support clustering across windows and penalties
- Final breakpoint selection from clustered candidates
- Optional timing output for bottleneck inspection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from pathlib import Path


FEATURE_COLS_ALL: Tuple[str, ...] = ("slope", "curv_int", "sinu", "width_s", "nch_s")


def normalize_feature_cols(cols: Sequence[str]) -> Tuple[str, ...]:
    seen = set()
    out: List[str] = []
    for c in cols:
        if c not in FEATURE_COLS_ALL:
            raise ValueError(f"Unknown feature column '{c}'. Valid options: {FEATURE_COLS_ALL}")
        if c not in seen:
            seen.add(c)
            out.append(c)
    if not out:
        raise ValueError("At least one feature column must be selected.")
    return tuple(out)


def unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for v in values:
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


def window_label(window_m: float) -> str:
    if float(window_m) <= 0.0:
        return "Wraw"
    return f"W{window_m / 1000.0:g}km"


def report_feature_health(df: pd.DataFrame, label: str, cols: Sequence[str]) -> None:
    n = len(df)
    # print(f"\n[{label}] feature health (n={n})")
    for c in cols:
        x = df[c].to_numpy(dtype=float)
        nan_frac = float(np.mean(~np.isfinite(x)))
        finite = int(np.isfinite(x).sum())
        uniq = np.unique(x[np.isfinite(x)]).size if finite > 0 else 0
        xmin = float(np.nanmin(x)) if finite > 0 else np.nan
        xmax = float(np.nanmax(x)) if finite > 0 else np.nan
        # print(
        #     f"  {c:10s} nan_frac={nan_frac:5.2f} finite={finite:5d} "
        #     f"uniq={uniq:4d} min={xmin:.3g} max={xmax:.3g}"
        # )


def _has_external_window_feature(
    external_by_window: Optional[Dict[str, pd.DataFrame]],
    feature_col: str,
) -> bool:
    if external_by_window is None:
        return False
    return any(feature_col in fdf.columns for fdf in external_by_window.values())


def _aligned_external_window_feature(
    external_df: pd.DataFrame,
    dist: np.ndarray,
    dist_col: str,
    feature_col: str,
    window_label_used: str,
) -> np.ndarray:
    require_cols(external_df, [dist_col, feature_col])
    ext = ensure_sorted_by_dist(external_df[[dist_col, feature_col]].copy(), dist_col)
    ext_dist = ext[dist_col].to_numpy(dtype=float)
    ext_values = ext[feature_col].to_numpy(dtype=float)

    if len(ext_dist) == len(dist) and np.allclose(ext_dist, dist, rtol=0.0, atol=1e-6):
        return ext_values

    merged = pd.DataFrame({dist_col: dist}).merge(ext, on=dist_col, how="left")
    out = merged[feature_col].to_numpy(dtype=float)
    missing = ~np.isfinite(out)
    if not np.any(missing):
        return out

    finite = np.isfinite(ext_dist) & np.isfinite(ext_values)
    if int(np.sum(finite)) >= 2:
        interp = np.interp(
            dist,
            ext_dist[finite],
            ext_values[finite],
            left=np.nan,
            right=np.nan,
        )
        out[missing] = interp[missing]
    elif int(np.sum(finite)) == 1:
        out[missing] = float(ext_values[finite][0])

    if np.all(~np.isfinite(out)):
        raise ValueError(
            f"{window_label_used}: external geometry feature '{feature_col}' "
            "could not be aligned to node distances."
        )
    return out


def rolling_sinuosity_from_xy_nodes(
    x: np.ndarray,
    y: np.ndarray,
    window_pts: int,
    min_pts: int = 5,
) -> np.ndarray:
    """
    Sinuosity(window) = polyline_length(window) / straight_distance(window endpoints)
    Computed on node-scale x,y.
    """
    n = len(x)
    half = window_pts // 2
    out = np.full(n, np.nan, dtype=float)

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        xx = x[i0:i1]
        yy = y[i0:i1]
        m = np.isfinite(xx) & np.isfinite(yy)
        xx = xx[m]
        yy = yy[m]
        if xx.size < min_pts:
            continue

        dx = np.diff(xx)
        dy = np.diff(yy)
        ch_len = float(np.nansum(np.sqrt(dx * dx + dy * dy)))
        straight = float(np.sqrt((xx[-1] - xx[0]) ** 2 + (yy[-1] - yy[0]) ** 2))
        if ch_len <= 0 or straight <= 0:
            continue
        out[i] = ch_len / straight

    return out


def add_node_xy_from_linestring(nodes_df: pd.DataFrame, linestring, dist_col: str = "dist_m") -> pd.DataFrame:
    """
    Adds x,y columns to nodes_df by interpolating the LineString at each node dist_m.
    dist_m must be in the same units as linestring.length (meters if projected).
    """
    nodes_df = nodes_df.sort_values(dist_col).reset_index(drop=True)
    d = nodes_df[dist_col].to_numpy(dtype=float)

    xs = np.empty_like(d)
    ys = np.empty_like(d)

    for i, di in enumerate(d):
        p = linestring.interpolate(float(di))
        xs[i] = p.x
        ys[i] = p.y

    out = nodes_df.copy()
    out["x"] = xs
    out["y"] = ys
    return out


def add_node_xy_from_geom_df(
    nodes_df: pd.DataFrame,
    geom_df_10m: pd.DataFrame,
    dist_col: str = "dist_m",
) -> pd.DataFrame:
    """
    Adds x,y columns to nodes_df by interpolating from a sampled geometry dataframe.
    """
    require_cols(nodes_df, [dist_col])
    require_cols(geom_df_10m, ["dist_m", "x", "y"])

    nodes_df = ensure_sorted_by_dist(nodes_df, dist_col).reset_index(drop=True)
    geom_df_10m = ensure_sorted_by_dist(geom_df_10m, "dist_m").reset_index(drop=True)

    d = nodes_df[dist_col].to_numpy(dtype=float)
    dg = geom_df_10m["dist_m"].to_numpy(dtype=float)
    xg = geom_df_10m["x"].to_numpy(dtype=float)
    yg = geom_df_10m["y"].to_numpy(dtype=float)

    out = nodes_df.copy()
    out["x"] = np.interp(d, dg, xg)
    out["y"] = np.interp(d, dg, yg)
    return out


# =============================================================================
# Basic utilities
# =============================================================================


def require_cols(df: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def ensure_sorted_by_dist(df: pd.DataFrame, dist_col: str = "dist_m") -> pd.DataFrame:
    if not df[dist_col].is_monotonic_increasing:
        df = df.sort_values(dist_col).reset_index(drop=True)
    return df


def infer_spacing_m(dist: np.ndarray) -> float:
    diffs = np.diff(dist)
    diffs = diffs[np.isfinite(diffs) & (diffs > 0)]
    if diffs.size == 0:
        raise ValueError("Cannot infer spacing: distance column has no positive diffs.")
    return float(np.median(diffs))


def zscore_df(
    df: pd.DataFrame,
    cols: Sequence[str],
) -> Tuple[pd.DataFrame, Dict[str, Tuple[float, float]]]:
    out = df.copy()
    stats: Dict[str, Tuple[float, float]] = {}
    for c in cols:
        x = out[c].to_numpy(dtype=float)
        mu = np.nanmean(x)
        sd = np.nanstd(x, ddof=0)
        if not np.isfinite(sd) or sd == 0:
            out[c] = (x - mu) * 0.0
            stats[c] = (mu, sd if np.isfinite(sd) else np.nan)
        else:
            out[c] = (x - mu) / sd
            stats[c] = (mu, sd)
    return out, stats


def sanitize_array(X: np.ndarray) -> np.ndarray:
    """
    Ensure finite values for scoring / PELT input.
    NaNs are filled with column medians; all-NaN columns become zeros.
    """
    X2 = X.copy().astype(float)
    for j in range(X2.shape[1]):
        col = X2[:, j]
        m = np.isfinite(col)
        if not np.any(m):
            X2[:, j] = 0.0
            continue
        med = np.nanmedian(col)
        if not np.isfinite(med):
            med = 0.0
        col[~m] = med
        X2[:, j] = col
    return X2


def print_timings(timings: Dict[str, float]) -> None:
    print("\n[Timings]")
    for key, value in timings.items():
        print(f"  {key:24s} {value:8.3f} s")


# =============================================================================
# Geometry: optional LineString sampling + curvature + sinuosity at 10 m
# =============================================================================


def sample_linestring_xy(linestring, ds: float = 10.0) -> pd.DataFrame:
    """
    Sample a shapely LineString at uniform spacing ds and return dist_m,x,y.
    """
    try:
        from shapely.geometry import LineString  # noqa: F401
    except ImportError as e:
        raise ImportError("shapely is required for LineString sampling. pip install shapely") from e

    L = float(linestring.length)
    if not np.isfinite(L) or L <= 0:
        raise ValueError("LineString length must be positive.")

    dists = np.arange(0.0, L + 0.5 * ds, ds)
    xs = np.empty_like(dists)
    ys = np.empty_like(dists)

    for i, d in enumerate(dists):
        p = linestring.interpolate(float(d))
        xs[i] = p.x
        ys[i] = p.y

    return pd.DataFrame({"dist_m": dists, "x": xs, "y": ys})


def _moving_average(a: np.ndarray, window: int | None) -> np.ndarray:
    if window is None or window <= 1:
        return a.astype(float)
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ap = np.pad(a.astype(float), (pad, pad), mode="edge")
    kernel = np.ones(window, dtype=float) / window
    return np.convolve(ap, kernel, mode="valid")


def curvature_from_xy_uniform(
    x: np.ndarray,
    y: np.ndarray,
    ds: float = 10.0,
    smooth_window: int | None = 9,
    return_signed: bool = True,
    eps: float = 1e-12,
) -> np.ndarray:
    """
    Curvature kappa = (x' y'' - y' x'') / (x'^2 + y'^2)^(3/2)
    using central differences with uniform spacing ds.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = x.size
    if n < 5:
        raise ValueError("Need at least 5 points for curvature.")

    xs = _moving_average(x, smooth_window)
    ys = _moving_average(y, smooth_window)

    dx = np.empty(n, dtype=float)
    dy = np.empty(n, dtype=float)
    dx[1:-1] = (xs[2:] - xs[:-2]) / (2 * ds)
    dy[1:-1] = (ys[2:] - ys[:-2]) / (2 * ds)
    dx[0] = (xs[1] - xs[0]) / ds
    dy[0] = (ys[1] - ys[0]) / ds
    dx[-1] = (xs[-1] - xs[-2]) / ds
    dy[-1] = (ys[-1] - ys[-2]) / ds

    ddx = np.empty(n, dtype=float)
    ddy = np.empty(n, dtype=float)
    ddx[1:-1] = (xs[2:] - 2 * xs[1:-1] + xs[:-2]) / (ds * ds)
    ddy[1:-1] = (ys[2:] - 2 * ys[1:-1] + ys[:-2]) / (ds * ds)
    ddx[0] = ddx[1]
    ddy[0] = ddy[1]
    ddx[-1] = ddx[-2]
    ddy[-1] = ddy[-2]

    num = dx * ddy - dy * ddx
    denom = (dx * dx + dy * dy) ** 1.5
    denom = np.maximum(denom, eps)

    kappa = num / denom
    if not return_signed:
        kappa = np.abs(kappa)
    return kappa


def rolling_sinuosity_from_xy(
    x: np.ndarray,
    y: np.ndarray,
    window_pts: int,
    min_pts: int = 5,
) -> np.ndarray:
    """
    Sinuosity in a centered window:
      sinu = polyline_length(window) / straight_distance(endpoints)
    """
    n = len(x)
    half = window_pts // 2
    out = np.full(n, np.nan, dtype=float)

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)
        xx = x[i0:i1]
        yy = y[i0:i1]
        m = np.isfinite(xx) & np.isfinite(yy)
        xx = xx[m]
        yy = yy[m]
        if xx.size < min_pts:
            continue

        dx = np.diff(xx)
        dy = np.diff(yy)
        ch_len = float(np.nansum(np.sqrt(dx * dx + dy * dy)))
        straight = float(np.sqrt((xx[-1] - xx[0]) ** 2 + (yy[-1] - yy[0]) ** 2))
        if ch_len <= 0 or straight <= 0:
            continue
        out[i] = ch_len / straight

    return out


def sample_metric_to_nodes(
    geom_df_10m: pd.DataFrame,
    node_dist_m: np.ndarray,
    metric_col: str,
    agg: str = "median",
    half_window_m: float = 100.0,
) -> np.ndarray:
    """
    Aggregate 10 m metric within ±half_window_m around each node distance.
    """
    dist10 = geom_df_10m["dist_m"].to_numpy(dtype=float)
    val10 = geom_df_10m[metric_col].to_numpy(dtype=float)
    out = np.full(len(node_dist_m), np.nan, dtype=float)

    for i, d in enumerate(node_dist_m):
        m = (dist10 >= d - half_window_m) & (dist10 <= d + half_window_m) & np.isfinite(val10)
        w = val10[m]
        if w.size == 0:
            continue
        if agg == "median":
            out[i] = float(np.median(w))
        elif agg == "mean":
            out[i] = float(np.mean(w))
        elif agg == "rms":
            out[i] = float(np.sqrt(np.mean(w * w)))
        elif agg == "p90":
            out[i] = float(np.percentile(w, 90))
        else:
            raise ValueError(f"Unknown agg={agg}")
    return out


def attach_geometry_metrics_to_nodes(
    nodes_df_200m: pd.DataFrame,
    linestring=None,
    geom_df_10m: Optional[pd.DataFrame] = None,
    ds_geom: float = 10.0,
    curvature_smooth_window: int = 9,
    sinuosity_window_m: float = 5000.0,
    node_agg_half_window_m: float = 100.0,
    curvature_agg: str = "rms",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Attach curvature and sinuosity to the 200 m nodes.

    Provide EITHER:
    - linestring (shapely LineString), OR
    - geom_df_10m with columns dist_m,x,y (already sampled)

    Returns:
      nodes_out (nodes_df with curvature,sinuosity)
      geom_df (10 m geometry df with curvature,sinuosity_10m)
    """
    require_cols(nodes_df_200m, ["dist_m"])
    nodes_df_200m = ensure_sorted_by_dist(nodes_df_200m, "dist_m")

    if geom_df_10m is None:
        if linestring is None:
            raise ValueError("Provide either 'linestring' or 'geom_df_10m'.")
        geom_df = sample_linestring_xy(linestring, ds=ds_geom)
    else:
        geom_df = geom_df_10m.copy()

    require_cols(geom_df, ["dist_m", "x", "y"])
    geom_df = ensure_sorted_by_dist(geom_df, "dist_m")

    x = geom_df["x"].to_numpy(dtype=float)
    y = geom_df["y"].to_numpy(dtype=float)

    geom_df["curvature"] = curvature_from_xy_uniform(
        x,
        y,
        ds=ds_geom,
        smooth_window=curvature_smooth_window,
        return_signed=True,
    )

    window_pts = int(np.round(sinuosity_window_m / ds_geom))
    window_pts = max(window_pts, 5)
    geom_df["sinuosity_10m"] = rolling_sinuosity_from_xy(x, y, window_pts=window_pts, min_pts=5)

    node_dist = nodes_df_200m["dist_m"].to_numpy(dtype=float)
    nodes_out = nodes_df_200m.copy()

    nodes_out["curvature"] = sample_metric_to_nodes(
        geom_df,
        node_dist,
        metric_col="curvature",
        agg=curvature_agg,
        half_window_m=node_agg_half_window_m,
    )
    nodes_out["sinuosity"] = sample_metric_to_nodes(
        geom_df,
        node_dist,
        metric_col="sinuosity_10m",
        agg="median",
        half_window_m=node_agg_half_window_m,
    )
    return nodes_out, geom_df


# =============================================================================
# Missing-aware slope + short-gap interpolation
# =============================================================================


def rolling_linear_slope_missing_aware(
    dist: np.ndarray,
    wse: np.ndarray,
    window_pts: int,
    min_valid_frac: float = 0.8,
    min_pts: int = 10,
) -> np.ndarray:
    """
    Moving-window slope from linear regression of WSE vs distance.
    Uses only finite WSE in the window.
    Returns NaN if not enough valid points.
    """
    n = len(dist)
    half = window_pts // 2
    out = np.full(n, np.nan, dtype=float)

    min_valid = max(int(np.ceil(min_valid_frac * window_pts)), min_pts)

    for i in range(n):
        i0 = max(0, i - half)
        i1 = min(n, i + half + 1)

        x = dist[i0:i1]
        y = wse[i0:i1]
        m = np.isfinite(x) & np.isfinite(y)
        x = x[m]
        y = y[m]

        if x.size < min_valid:
            continue

        x0 = x - x.mean()
        y0 = y - y.mean()
        denom = np.dot(x0, x0)
        if denom <= 0 or not np.isfinite(denom):
            continue

        b = np.dot(x0, y0) / denom
        out[i] = -b
    return out


def interpolate_short_gaps(series: pd.Series, max_gap_pts: int = 5) -> pd.Series:
    """
    Linear interpolation only for NaN runs up to max_gap_pts.
    Longer NaN runs remain NaN as much as pandas limit rules allow.
    """
    s = series.copy()
    if not s.isna().any():
        return s

    is_na = s.isna().to_numpy()
    n = len(s)
    sentinel = 1e308
    i = 0
    while i < n:
        if not is_na[i]:
            i += 1
            continue
        j = i
        while j < n and is_na[j]:
            j += 1
        run_len = j - i
        if run_len > max_gap_pts:
            s.iloc[i:j] = sentinel
        i = j

    s = s.replace(sentinel, np.nan)
    s = s.interpolate(method="linear", limit=max_gap_pts, limit_direction="both")
    return s


# =============================================================================
# Rolling summaries for other variables
# =============================================================================


def window_indices(i: int, half: int, n: int) -> slice:
    return slice(max(0, i - half), min(n, i + half + 1))


def rolling_robust_summary(
    x: np.ndarray,
    window_pts: int,
    func: str = "median",
    min_pts: int = 5,
) -> np.ndarray:
    n = len(x)
    half = window_pts // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        sl = window_indices(i, half, n)
        w = x[sl]
        w = w[np.isfinite(w)]
        if w.size < min_pts:
            continue
        if func == "median":
            out[i] = float(np.median(w))
        elif func == "mean":
            out[i] = float(np.mean(w))
        elif func == "rms":
            out[i] = float(np.sqrt(np.mean(w * w)))
        elif func == "p90":
            out[i] = float(np.percentile(w, 90))
        else:
            raise ValueError(f"Unknown func={func}")
    return out


def rolling_mode_int(x: np.ndarray, window_pts: int, min_pts: int = 5) -> np.ndarray:
    n = len(x)
    half = window_pts // 2
    out = np.full(n, np.nan, dtype=float)
    for i in range(n):
        sl = window_indices(i, half, n)
        w = x[sl]
        w = w[np.isfinite(w)]
        if w.size < min_pts:
            continue
        w_int = np.round(w).astype(int)
        vals, counts = np.unique(w_int, return_counts=True)
        out[i] = float(vals[np.argmax(counts)])
    return out


# =============================================================================
# Configs
# =============================================================================


@dataclass
class FeatureConfig:
    dist_col: str = "dist_m"
    wse_col: str = "wse"
    curvature_col: str = "curvature"
    sinuosity_col: str = "sinuosity"
    width_col: str = "width"
    nch_col: str = "n_channels"

    curvature_summary: str = "rms"
    width_summary: str = "median"
    nch_summary: str = "mode"
    multi_chan_treatment: bool = True
    multi_chan_threshold: float = 1.0
    use_abs_curvature: bool = True
    log_width: bool = True

    slope_min_valid_frac: float = 0.8
    slope_min_pts: int = 10
    slope_interp_max_gap_pts: int = 5

    ds_geom: float = 10.0
    curvature_smooth_window: int = 9
    sinuosity_window_m: float = 5000.0
    node_agg_half_window_m: float = 100.0
    curvature_agg: str = "rms"


@dataclass
class PeltConfig:
    feature_cols: Tuple[str, ...] = FEATURE_COLS_ALL
    min_size_pts: Optional[int] = None
    jump: int = 1


@dataclass
class WindowSelectionConfig:
    method: str = "fixed"  # "fixed", "width_quantile_log", or "raw"
    windows_m: Optional[Tuple[float, ...]] = (5000.0, 10000.0, 20000.0)
    n_windows: int = 5
    width_col: str = "width"
    low_quantile: float = 0.25
    high_quantile: float = 0.75
    min_width_multiplier: float = 25.0
    max_width_multiplier: float = 100.0
    min_window_m: Optional[float] = None
    max_window_m: Optional[float] = None
    max_window_fraction_of_length: Optional[float] = 0.25
    min_window_pts: int = 5
    round_to_spacing: bool = True


@dataclass
class BreakSelectionResult:
    breaks_m: List[float]
    segments: List[Tuple[int, int]]
    history: pd.DataFrame
    candidates_used_m: List[float]
    windows_used: List[str]
    feature_cols_used: List[str]


@dataclass
class BreakSelectionGridResult:
    individual_results: Dict[str, BreakSelectionResult]
    summary: pd.DataFrame
    consensus: pd.DataFrame
    stable_breaks_m: List[float]
    stable_segments: List[Tuple[int, int]]
    stable_support_frac_min: float
    stable_support_count: Optional[int] = None
    consensus_method: str = "complete_linkage"
    merge_threshold_m: float = 11_500.0
    n_settings_valid: int = 0


@dataclass
class BreakSelectionConfig:
    enabled: bool = True
    windows: Optional[Tuple[str, ...]] = None
    feature_cols: Optional[Tuple[str, ...]] = None
    candidate_source: str = "stability"  # "stability" or "all_runs"
    candidate_freq_min: Optional[float] = None  # backward-compatible raw-cluster frequency
    min_support_frac_runs: float = 0.20
    min_windows_supported: int = 1
    min_support_frac_windows: float = 0.0
    stability_tolerance_m: Optional[float] = None
    min_spacing_m: float = 10_000.0
    min_reach_len_m: Optional[float] = None
    max_breaks: int = 30
    stop_rel_improvement: float = 0.01
    stop_abs_improvement: float = 0.0
    window_weights: Optional[Dict[str, float]] = None
    verbose: bool = False


@dataclass
class ConsensusConfig:
    method: str = "complete_linkage"
    merge_threshold_m: float = 11_500.0
    calibration_label: str = "full_grid_first_stable_plateau_u20_u30"


@dataclass
class BreakSelectionGridConfig:
    enabled: bool = False
    min_support_frac_runs_values: Tuple[float, ...] = (0.15, 0.20)
    min_windows_supported_values: Tuple[int, ...] = (1, 2)
    stop_rel_improvement_values: Tuple[float, ...] = (0.015, 0.02)
    consensus: ConsensusConfig = field(default_factory=ConsensusConfig)
    stable_support_frac_min: float = 0.75
    verbose: bool = False


@dataclass
class PipelineConfig:
    windows_m: Optional[Tuple[float, ...]] = None
    window_selection_method: Optional[str] = None
    penalties: Tuple[float, ...] = (5.0, 10.0, 20.0, 40.0, 80.0)
    standardize: bool = True
    stability_tolerance_m: float = 1000.0
    min_pts_in_window: int = 5
    report_feature_health: bool = True
    print_timings: bool = False
    window_selection: WindowSelectionConfig = field(default_factory=WindowSelectionConfig)
    break_selection: BreakSelectionConfig = field(default_factory=BreakSelectionConfig)
    break_selection_grid: BreakSelectionGridConfig = field(default_factory=BreakSelectionGridConfig)


DEFAULT_FROZEN_CONSENSUS_CONFIG = ConsensusConfig()
DEFAULT_TUNING_STABLE_SUPPORT_FRAC_MIN = 0.75
DEFAULT_FINAL_STABLE_SUPPORT_COUNT = 4


def get_effective_window_selection_config(pipe_cfg: PipelineConfig) -> WindowSelectionConfig:
    window_cfg = WindowSelectionConfig(**vars(pipe_cfg.window_selection))
    if pipe_cfg.windows_m is not None:
        window_cfg.windows_m = tuple(float(w) for w in pipe_cfg.windows_m)
    if pipe_cfg.window_selection_method is not None:
        window_cfg.method = str(pipe_cfg.window_selection_method)
    return window_cfg


# =============================================================================
# Window selection
# =============================================================================


def resolve_window_sizes(
    nodes_df: pd.DataFrame,
    feat_cfg: FeatureConfig = FeatureConfig(),
    window_cfg: WindowSelectionConfig = WindowSelectionConfig(),
) -> Tuple[Tuple[float, ...], Dict[str, object]]:
    require_cols(nodes_df, [feat_cfg.dist_col])
    df = ensure_sorted_by_dist(nodes_df, feat_cfg.dist_col).reset_index(drop=True)

    dist = df[feat_cfg.dist_col].to_numpy(dtype=float)
    spacing = infer_spacing_m(dist)
    river_length_m = float(dist[-1] - dist[0])

    if window_cfg.method == "fixed":
        if not window_cfg.windows_m:
            raise ValueError("WindowSelectionConfig.method='fixed' requires windows_m.")
        raw_windows = np.asarray(window_cfg.windows_m, dtype=float)
        meta = {"method": "fixed"}

    elif window_cfg.method == "raw":
        raw_windows = np.array([0.0], dtype=float)
        meta = {"method": "raw"}

    elif window_cfg.method == "width_quantile_log":
        width_col = window_cfg.width_col
        require_cols(df, [width_col])
        width = df[width_col].to_numpy(dtype=float)
        width = width[np.isfinite(width) & (width > 0)]
        if width.size == 0:
            raise ValueError("Width-based window selection requires finite positive widths.")

        width_lo = float(np.quantile(width, window_cfg.low_quantile))
        width_hi = float(np.quantile(width, window_cfg.high_quantile))
        raw_wmin = (
            float(window_cfg.min_window_m)
            if window_cfg.min_window_m is not None
            else float(window_cfg.min_width_multiplier * width_lo)
        )
        raw_wmax = (
            float(window_cfg.max_window_m)
            if window_cfg.max_window_m is not None
            else float(window_cfg.max_width_multiplier * width_hi)
        )

        if window_cfg.max_window_fraction_of_length is not None:
            raw_wmax = min(raw_wmax, float(window_cfg.max_window_fraction_of_length) * river_length_m)

        min_window_from_spacing = max(int(window_cfg.min_window_pts), 3) * spacing
        raw_wmin = max(raw_wmin, min_window_from_spacing)
        raw_wmax = max(raw_wmax, raw_wmin)

        if int(window_cfg.n_windows) <= 1 or np.isclose(raw_wmin, raw_wmax):
            raw_windows = np.array([raw_wmin], dtype=float)
        else:
            raw_windows = np.exp(np.linspace(np.log(raw_wmin), np.log(raw_wmax), int(window_cfg.n_windows)))

        meta = {
            "method": "width_quantile_log",
            "width_low_quantile": float(window_cfg.low_quantile),
            "width_high_quantile": float(window_cfg.high_quantile),
            "width_lo_m": width_lo,
            "width_hi_m": width_hi,
            "raw_window_min_m": raw_wmin,
            "raw_window_max_m": raw_wmax,
        }

    else:
        raise ValueError("window_cfg.method must be 'fixed', 'width_quantile_log', or 'raw'.")

    if window_cfg.method != "raw" and window_cfg.round_to_spacing:
        raw_windows = np.round(raw_windows / spacing) * spacing

    if window_cfg.method != "raw":
        min_window = max(int(window_cfg.min_window_pts), 3) * spacing
        raw_windows = np.maximum(raw_windows, min_window)
    raw_windows = raw_windows[np.isfinite(raw_windows) & (raw_windows > 0)]
    if window_cfg.method == "raw":
        raw_windows = np.array([0.0], dtype=float)
    windows = tuple(float(w) for w in np.unique(raw_windows))
    if len(windows) == 0:
        raise ValueError("Window selection produced no valid windows.")

    meta.update(
        {
            "spacing_m": spacing,
            "river_length_m": river_length_m,
            "windows_m": windows,
            "window_labels": [window_label(w) for w in windows],
        }
    )
    return windows, meta


def window_pts_from_window_m(window_m: float, spacing_m: float) -> int:
    if float(window_m) <= 0.0:
        return 1
    window_pts = int(np.round(float(window_m) / float(spacing_m)))
    return max(window_pts, 3)


def summarize_window_feasibility(
    nodes_df: pd.DataFrame,
    windows_m: Sequence[float],
    dist_col: str = "dist_m",
) -> Dict[str, object]:
    require_cols(nodes_df, [dist_col])
    df = ensure_sorted_by_dist(nodes_df, dist_col).reset_index(drop=True)
    dist = df[dist_col].to_numpy(dtype=float)
    spacing_m = infer_spacing_m(dist)
    n_nodes = int(len(df))

    rows: List[Dict[str, object]] = []
    for order, window_m in enumerate(tuple(float(w) for w in windows_m)):
        window_pts = window_pts_from_window_m(window_m, spacing_m)
        is_feasible = bool(window_pts <= n_nodes)
        rows.append(
            {
                "window_order": int(order),
                "window_m": float(window_m),
                "window_km": float(window_m) / 1000.0,
                "window_label": window_label(float(window_m)),
                "window_pts": int(window_pts),
                "n_nodes": int(n_nodes),
                "spacing_m": float(spacing_m),
                "is_feasible": is_feasible,
                "is_dropped": bool(not is_feasible),
            }
        )

    summary_df = pd.DataFrame(rows)
    feasible_windows_m = tuple(
        float(v) for v in summary_df.loc[summary_df["is_feasible"], "window_m"].tolist()
    )
    infeasible_windows_m = tuple(
        float(v) for v in summary_df.loc[~summary_df["is_feasible"], "window_m"].tolist()
    )
    first_window_feasible = bool(summary_df["is_feasible"].iloc[0]) if not summary_df.empty else False

    return {
        "summary_df": summary_df,
        "spacing_m": float(spacing_m),
        "n_nodes": int(n_nodes),
        "nominal_windows_m": tuple(float(w) for w in windows_m),
        "feasible_windows_m": feasible_windows_m,
        "infeasible_windows_m": infeasible_windows_m,
        "n_windows_nominal": int(len(windows_m)),
        "n_windows_feasible": int(len(feasible_windows_m)),
        "n_windows_infeasible": int(len(infeasible_windows_m)),
        "any_infeasible": bool(len(infeasible_windows_m) > 0),
        "all_infeasible": bool(len(feasible_windows_m) == 0),
        "first_window_feasible": first_window_feasible,
    }


# =============================================================================
# Feature building at 200 m nodes
# =============================================================================


def build_multiscale_features(
    nodes_df: pd.DataFrame,
    windows_m: Sequence[float],
    cfg: FeatureConfig = FeatureConfig(),
    min_pts_in_window: int = 5,
    feature_cols_to_compute: Optional[Sequence[str]] = None,
    geometry_features_by_window: Optional[Dict[str, pd.DataFrame]] = None,
    report_health: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Build per-window features on the nodes.

    Output columns per window always include:
      dist_m, window_m, window_pts
    plus only the requested feature columns.
    """
    selected = normalize_feature_cols(feature_cols_to_compute or FEATURE_COLS_ALL)
    require_cols(nodes_df, [cfg.dist_col])
    external_geometry_cols = {
        c
        for c in ("sinu", "curv_int")
        if _has_external_window_feature(geometry_features_by_window, c)
    }

    if "slope" in selected:
        require_cols(nodes_df, [cfg.wse_col])
    if "width_s" in selected:
        require_cols(nodes_df, [cfg.width_col])
    if "nch_s" in selected:
        require_cols(nodes_df, [cfg.nch_col])
    if "curv_int" in selected and "curv_int" not in external_geometry_cols and cfg.curvature_col not in nodes_df.columns:
        raise ValueError(
            f"Requested 'curv_int' but nodes_df is missing '{cfg.curvature_col}'. "
            "Provide curvature or a centerline / geom_df_10m upstream."
        )
    if "sinu" in selected and "sinu" not in external_geometry_cols and not {"x", "y"}.issubset(nodes_df.columns):
        raise ValueError(
            "Requested 'sinu' but nodes_df is missing x/y columns. "
            "Provide node x/y directly or provide a centerline / geom_df_10m upstream."
        )

    df = ensure_sorted_by_dist(nodes_df, cfg.dist_col).reset_index(drop=True)
    dist = df[cfg.dist_col].to_numpy(dtype=float)
    spacing = infer_spacing_m(dist)
    out: Dict[str, pd.DataFrame] = {}

    wse = df[cfg.wse_col].to_numpy(dtype=float) if "slope" in selected else None
    curv = None
    if "curv_int" in selected and "curv_int" not in external_geometry_cols:
        curv = df[cfg.curvature_col].to_numpy(dtype=float)
        if cfg.use_abs_curvature:
            curv = np.abs(curv)
    width = None
    if "width_s" in selected:
        width = df[cfg.width_col].to_numpy(dtype=float)
        if cfg.log_width:
            width = np.where(width > 0, np.log(width), np.nan)
    nch = None
    if "nch_s" in selected:
        nch_raw = df[cfg.nch_col].to_numpy(dtype=float)
        if cfg.multi_chan_treatment:
            nch = np.where(
                np.isfinite(nch_raw),
                (nch_raw > float(cfg.multi_chan_threshold)).astype(float),
                np.nan,
            )
        else:
            nch = nch_raw
    x = df["x"].to_numpy(dtype=float) if "sinu" in selected and "sinu" not in external_geometry_cols else None
    y = df["y"].to_numpy(dtype=float) if "sinu" in selected and "sinu" not in external_geometry_cols else None

    for Wm in windows_m:
        is_raw_window = float(Wm) <= 0.0
        label = window_label(float(Wm))
        if is_raw_window:
            unsupported = [c for c in selected if c not in {"width_s", "nch_s"}]
            if unsupported:
                raise ValueError(
                    "Raw window mode only supports width_s and nch_s. "
                    "Geometry features require positive window sizes. "
                    f"Unsupported raw features requested: {unsupported}"
                )
            window_pts = 1
        else:
            window_pts = window_pts_from_window_m(float(Wm), spacing)

        fdf = pd.DataFrame({cfg.dist_col: dist})
        external_fdf = None
        if geometry_features_by_window is not None:
            external_fdf = geometry_features_by_window.get(label)

        if "slope" in selected:
            slope = rolling_linear_slope_missing_aware(
                dist=dist,
                wse=wse,
                window_pts=window_pts,
                min_valid_frac=cfg.slope_min_valid_frac,
                min_pts=cfg.slope_min_pts,
            )
            slope = interpolate_short_gaps(
                pd.Series(slope),
                max_gap_pts=cfg.slope_interp_max_gap_pts,
            ).to_numpy()
            slope = pd.Series(slope).ffill().bfill().to_numpy()
            fdf["slope"] = slope

        if "curv_int" in selected:
            if external_fdf is not None and "curv_int" in external_fdf.columns:
                fdf["curv_int"] = _aligned_external_window_feature(
                    external_fdf,
                    dist,
                    cfg.dist_col,
                    "curv_int",
                    label,
                )
            else:
                fdf["curv_int"] = rolling_robust_summary(
                    curv,
                    window_pts=window_pts,
                    func=cfg.curvature_summary,
                    min_pts=min_pts_in_window,
                )

        if "sinu" in selected:
            if external_fdf is not None and "sinu" in external_fdf.columns:
                fdf["sinu"] = _aligned_external_window_feature(
                    external_fdf,
                    dist,
                    cfg.dist_col,
                    "sinu",
                    label,
                )
            else:
                fdf["sinu"] = rolling_sinuosity_from_xy_nodes(
                    x,
                    y,
                    window_pts=window_pts,
                    min_pts=min_pts_in_window,
                )

        if "width_s" in selected:
            if is_raw_window:
                fdf["width_s"] = width.copy()
            else:
                fdf["width_s"] = rolling_robust_summary(
                    width,
                    window_pts=window_pts,
                    func=cfg.width_summary,
                    min_pts=min_pts_in_window,
                )

        if "nch_s" in selected:
            if is_raw_window:
                fdf["nch_s"] = nch.copy()
            elif cfg.multi_chan_treatment:
                fdf["nch_s"] = rolling_robust_summary(
                    nch,
                    window_pts=window_pts,
                    func="mean",
                    min_pts=min_pts_in_window,
                )
            else:
                if cfg.nch_summary == "mode":
                    fdf["nch_s"] = rolling_mode_int(
                        nch,
                        window_pts=window_pts,
                        min_pts=min_pts_in_window,
                    )
                elif cfg.nch_summary == "mean":
                    fdf["nch_s"] = rolling_robust_summary(
                        nch,
                        window_pts=window_pts,
                        func="mean",
                        min_pts=min_pts_in_window,
                    )
                else:
                    raise ValueError(f"Unknown nch_summary={cfg.nch_summary}")

        fdf["window_m"] = float(Wm)
        fdf["window_pts"] = int(window_pts)
        if report_health:
            report_feature_health(fdf, label, selected)
        out[label] = fdf

    return out


# =============================================================================
# PELT + support clustering
# =============================================================================


def _prepare_pelt_run(
    features_df: pd.DataFrame,
    pelt_cfg: PeltConfig,
):
    try:
        import ruptures as rpt
    except ImportError as e:
        raise ImportError("ruptures is required. pip install ruptures") from e

    cols = list(normalize_feature_cols(pelt_cfg.feature_cols))
    require_cols(features_df, cols + ["dist_m", "window_pts"])

    X = features_df[cols].to_numpy(dtype=float)
    X_imp = sanitize_array(X)

    window_pts = int(features_df["window_pts"].iloc[0])
    min_size = int(pelt_cfg.min_size_pts if pelt_cfg.min_size_pts is not None else window_pts)
    algo = rpt.Pelt(model="l2", min_size=min_size, jump=int(pelt_cfg.jump)).fit(X_imp)
    n = len(features_df)
    dist = features_df["dist_m"].to_numpy(dtype=float)
    window_m = float(features_df["window_m"].iloc[0])

    return algo, dist, cols, window_pts, min_size, n, window_m


def _predict_pelt_breaks(
    algo,
    dist: np.ndarray,
    cols: Sequence[str],
    window_pts: int,
    min_size: int,
    n: int,
    window_m: float,
    penalty: float,
) -> Dict[str, object]:
    bkps = algo.predict(pen=float(penalty))
    bkps_idx = [b for b in bkps if b < n]
    bkps_dist = [float(dist[b]) for b in bkps_idx]

    return {
        "penalty": float(penalty),
        "min_size_pts": int(min_size),
        "break_indices": bkps_idx,
        "break_dist_m": bkps_dist,
        "n_points": int(n),
        "feature_cols": cols,
        "window_pts": int(window_pts),
        "window_m": float(window_m),
    }


def run_pelt(features_df: pd.DataFrame, pelt_cfg: PeltConfig, penalty: float) -> Dict[str, object]:
    algo, dist, cols, window_pts, min_size, n, window_m = _prepare_pelt_run(features_df, pelt_cfg)
    return _predict_pelt_breaks(
        algo=algo,
        dist=dist,
        cols=cols,
        window_pts=window_pts,
        min_size=min_size,
        n=n,
        window_m=window_m,
        penalty=penalty,
    )


def penalty_sweep(
    features_df: pd.DataFrame,
    pelt_cfg: PeltConfig,
    penalties: Sequence[float],
) -> List[Dict[str, object]]:
    algo, dist, cols, window_pts, min_size, n, window_m = _prepare_pelt_run(features_df, pelt_cfg)
    return [
        _predict_pelt_breaks(
            algo=algo,
            dist=dist,
            cols=cols,
            window_pts=window_pts,
            min_size=min_size,
            n=n,
            window_m=window_m,
            penalty=float(p),
        )
        for p in penalties
    ]


def compute_breakpoint_stability(
    all_runs: List[Dict[str, object]],
    tolerance_m: float = 1000.0,
) -> pd.DataFrame:
    """
    Greedy 1D clustering of breakpoint locations within tolerance_m.

    Preserves the original columns:
      cluster_center_m, count, runs_total, freq, members_m
    and adds support-aware fields:
      n_windows_supported, support_frac_windows,
      n_run_pairs_supported, support_frac_runs, etc.
    """
    rows_pts: List[Dict[str, object]] = []
    for run_id, r in enumerate(all_runs):
        window = str(r.get("window_label", f"run_{run_id}"))
        penalty = float(r.get("penalty", np.nan))
        for d in r.get("break_dist_m", []):
            if np.isfinite(d):
                rows_pts.append(
                    {
                        "break_m": float(d),
                        "run_id": int(run_id),
                        "window_label": window,
                        "penalty": penalty,
                        "run_key": (window, penalty),
                    }
                )

    runs_total = len(all_runs)
    if len(rows_pts) == 0:
        return pd.DataFrame(
            columns=[
                "cluster_center_m",
                "count",
                "runs_total",
                "freq",
                "members_m",
                "n_windows_supported",
                "support_frac_windows",
                "n_run_pairs_supported",
                "support_frac_runs",
            ]
        )

    pts_df = pd.DataFrame(rows_pts).sort_values("break_m").reset_index(drop=True)
    total_windows = len({str(r.get("window_label", "")) for r in all_runs})

    clusters: List[List[int]] = []
    cur = [0]
    for i in range(1, len(pts_df)):
        if abs(float(pts_df.loc[i, "break_m"]) - float(pts_df.loc[i - 1, "break_m"])) <= tolerance_m:
            cur.append(i)
        else:
            clusters.append(cur)
            cur = [i]
    clusters.append(cur)

    rows = []
    for cluster_id, idx in enumerate(clusters):
        c = pts_df.loc[idx].copy()
        members = [float(v) for v in c["break_m"].tolist()]
        run_pairs = sorted(set(c["run_key"].tolist()))
        windows = sorted(set(c["window_label"].tolist()))
        penalties = sorted(set(float(v) for v in c["penalty"].tolist()))
        support_by_window = (
            c.groupby("window_label")["penalty"]
            .apply(lambda s: sorted(set(float(v) for v in s.tolist())))
            .to_dict()
        )
        support_counts_by_window = {k: len(v) for k, v in support_by_window.items()}

        rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_center_m": float(np.median(np.asarray(members, dtype=float))),
                "count": int(len(members)),
                "runs_total": int(runs_total),
                "freq": float(len(members) / max(runs_total, 1)),
                "members_m": members,
                "windows_total": int(total_windows),
                "n_windows_supported": int(len(windows)),
                "support_frac_windows": float(len(windows) / max(total_windows, 1)),
                "windows_supported": windows,
                "penalties_supported": penalties,
                "n_penalties_supported": int(len(penalties)),
                "n_run_pairs_supported": int(len(run_pairs)),
                "support_frac_runs": float(len(run_pairs) / max(runs_total, 1)),
                "run_pairs_supported": run_pairs,
                "support_by_window": support_by_window,
                "support_counts_by_window": support_counts_by_window,
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            ["n_windows_supported", "support_frac_runs", "freq", "count"],
            ascending=False,
        )
        .reset_index(drop=True)
    )


# =============================================================================
# Final breakpoint selection
# =============================================================================


def select_segment_breaks_from_results(
    results: Dict[str, object],
    windows: Optional[Sequence[str]] = None,
    feature_cols: Sequence[str] = FEATURE_COLS_ALL,
    candidate_source: str = "stability",
    candidate_freq_min: Optional[float] = None,
    min_support_frac_runs: float = 0.20,
    min_windows_supported: int = 1,
    min_support_frac_windows: float = 0.0,
    stability_tolerance_m: Optional[float] = None,
    min_spacing_m: float = 10_000.0,
    min_reach_len_m: Optional[float] = None,
    max_breaks: int = 30,
    stop_rel_improvement: float = 0.01,
    stop_abs_improvement: float = 0.0,
    window_weights: Optional[Dict[str, float]] = None,
    verbose: bool = True,
) -> BreakSelectionResult:
    """
    Select a single set of breakpoints that is supported across multiple windows
    while minimizing total within-segment SSE across the chosen windows.
    """

    def _require_key(d: Dict[str, object], k: str) -> None:
        if k not in d:
            raise KeyError(f"results is missing required key: '{k}'")

    def _validate_window_df(df: pd.DataFrame, wlab: str) -> None:
        for c in ["dist_m", "window_m", "window_pts"]:
            if c not in df.columns:
                raise ValueError(f"{wlab}: missing column '{c}'")
        dist = df["dist_m"].to_numpy(dtype=float)
        if not np.all(np.isfinite(dist)):
            raise ValueError(f"{wlab}: dist_m contains NaNs/infs")
        if not np.all(np.diff(dist) > 0):
            raise ValueError(f"{wlab}: dist_m must be strictly increasing.")

    def _validate_breaks(dist: np.ndarray, breaks_m: List[float]) -> None:
        if len(breaks_m) == 0:
            return
        b = np.array(breaks_m, dtype=float)
        if np.any(~np.isfinite(b)):
            raise ValueError("Selected breaks contain NaN/inf.")
        if not np.all(np.diff(np.sort(b)) > 0):
            raise ValueError("Selected breaks must be strictly increasing.")
        if b.min() <= dist[0] or b.max() >= dist[-1]:
            raise ValueError("Selected breaks must be inside the distance range.")

        if min_spacing_m is not None and len(b) > 1:
            if np.min(np.diff(np.sort(b))) < min_spacing_m - 1e-9:
                raise ValueError("Selected breaks violate min_spacing_m.")

        if min_reach_len_m is not None:
            cuts = np.sort(b)
            endpoints = np.concatenate([[dist[0]], cuts, [dist[-1]]])
            if np.min(np.diff(endpoints)) < min_reach_len_m - 1e-9:
                raise ValueError("Selected breaks violate min_reach_len_m.")

    def _breaks_to_segments(dist: np.ndarray, breaks_m: List[float]) -> List[Tuple[int, int]]:
        idx = np.searchsorted(dist, np.array(breaks_m, dtype=float), side="left")
        idx = [int(i) for i in idx if 0 < i < len(dist)]
        idx = sorted(set(idx))
        cuts = [0] + idx + [len(dist)]
        return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    def _within_segment_sse(dist: np.ndarray, X: np.ndarray, breaks_m: List[float]) -> float:
        segs = _breaks_to_segments(dist, breaks_m)
        total = 0.0
        for a, b in segs:
            Xi = X[a:b, :]
            mu = Xi.mean(axis=0)
            total += float(((Xi - mu) ** 2).sum())
        return total

    def _total_weighted_sse(breaks_m: List[float]) -> float:
        s = 0.0
        for wlab in windows_used:
            dfw = std_by_window[wlab]
            distw = dfw["dist_m"].to_numpy(dtype=float)
            Xw = sanitize_array(dfw[list(feature_cols_used)].to_numpy(dtype=float))
            s += float(weights.get(wlab, 1.0)) * _within_segment_sse(distw, Xw, breaks_m)
        return float(s)

    def _compatible_with_constraints(breaks_m: List[float], cand: float) -> bool:
        if any(abs(cand - b) < min_spacing_m for b in breaks_m):
            return False

        if min_reach_len_m is not None:
            trial = sorted(breaks_m + [cand])
            endpoints = np.concatenate([[ref_dist[0]], np.array(trial), [ref_dist[-1]]])
            if np.min(np.diff(endpoints)) < min_reach_len_m - 1e-9:
                return False

        return True

    _require_key(results, "standardized_by_window")
    std_by_window: Dict[str, pd.DataFrame] = results["standardized_by_window"]  # type: ignore

    if windows is None:
        windows_used = sorted(list(std_by_window.keys()))
    else:
        windows_used = list(windows)
        missing = [w for w in windows_used if w not in std_by_window]
        if missing:
            raise ValueError(f"Requested windows not found in results['standardized_by_window']: {missing}")

    for w in windows_used:
        _validate_window_df(std_by_window[w], w)

    requested = normalize_feature_cols(feature_cols)
    common = [c for c in requested if all(c in std_by_window[w].columns for w in windows_used)]
    if len(common) == 0:
        raise ValueError("None of the requested feature_cols exist in all selected windows.")

    feature_cols_used = []
    for c in common:
        useful_somewhere = False
        for w in windows_used:
            x = std_by_window[w][c].to_numpy(dtype=float)
            if np.nanstd(x) > 1e-12:
                useful_somewhere = True
                break
        if useful_somewhere:
            feature_cols_used.append(c)

    if len(feature_cols_used) == 0:
        raise ValueError("All selected feature columns are constant/zero across the chosen windows.")

    ref_w = windows_used[0]
    ref_dist = std_by_window[ref_w]["dist_m"].to_numpy(dtype=float)

    if window_weights is None:
        weights = {w: 1.0 for w in windows_used}
    else:
        weights = {w: float(window_weights.get(w, 1.0)) for w in windows_used}

    candidates_m: List[float] = []

    if candidate_source == "stability":
        _require_key(results, "stability")
        stab: pd.DataFrame = results["stability"]  # type: ignore
        if "cluster_center_m" not in stab.columns:
            raise ValueError("results['stability'] must have 'cluster_center_m'.")

        mask = pd.Series(True, index=stab.index)
        if candidate_freq_min is not None and "freq" in stab.columns:
            mask &= stab["freq"] >= float(candidate_freq_min)
        if "support_frac_runs" in stab.columns:
            mask &= stab["support_frac_runs"] >= float(min_support_frac_runs)
        elif "freq" in stab.columns:
            mask &= stab["freq"] >= float(min_support_frac_runs)
        if "n_windows_supported" in stab.columns:
            mask &= stab["n_windows_supported"] >= int(min_windows_supported)
        if "support_frac_windows" in stab.columns:
            mask &= stab["support_frac_windows"] >= float(min_support_frac_windows)

        candidates_m = stab.loc[mask, "cluster_center_m"].astype(float).tolist()

    elif candidate_source == "all_runs":
        _require_key(results, "all_runs")
        all_runs: List[Dict[str, object]] = results["all_runs"]  # type: ignore
        pts = []
        for r in all_runs:
            pts.extend([float(d) for d in r.get("break_dist_m", []) if np.isfinite(d)])
        pts = sorted(pts)

        if stability_tolerance_m is None:
            candidates_m = pts
        else:
            clusters = []
            if len(pts) > 0:
                cur = [pts[0]]
                for d in pts[1:]:
                    if abs(d - cur[-1]) <= stability_tolerance_m:
                        cur.append(d)
                    else:
                        clusters.append(cur)
                        cur = [d]
                clusters.append(cur)
            candidates_m = [float(np.median(c)) for c in clusters]

    else:
        raise ValueError("candidate_source must be 'stability' or 'all_runs'.")

    lo, hi = float(ref_dist[0]), float(ref_dist[-1])
    candidates_m = [float(c) for c in candidates_m if lo < float(c) < hi]
    candidates_m = sorted(set(round(c, 6) for c in candidates_m))

    if len(candidates_m) == 0:
        raise ValueError("No candidates available after filtering.")

    base_sse = _total_weighted_sse([])
    selected: List[float] = []
    rows = [
        {
            "k": 0,
            "added_m": np.nan,
            "added_km": np.nan,
            "total_sse": base_sse,
            "improvement": np.nan,
            "rel_improvement": np.nan,
        }
    ]

    if verbose:
        print(f"Windows used: {windows_used}")
        print(f"Feature cols used: {feature_cols_used}")
        print(f"Candidates: {len(candidates_m)}")
        print(f"Initial total SSE: {base_sse:,.3f}")

    current_sse = base_sse
    for _ in range(1, max_breaks + 1):
        best_cand = None
        best_sse = current_sse
        best_imp = 0.0

        for cand in candidates_m:
            if cand in selected:
                continue
            if not _compatible_with_constraints(selected, cand):
                continue

            trial = sorted(selected + [cand])
            sse = _total_weighted_sse(trial)
            imp = current_sse - sse
            if imp > best_imp + 1e-12:
                best_imp = imp
                best_sse = sse
                best_cand = cand

        if best_cand is None:
            if verbose:
                print("No more candidates satisfy constraints / improve SSE.")
            break

        rel_imp = best_imp / base_sse if base_sse > 0 else 0.0
        if best_imp < stop_abs_improvement or rel_imp < stop_rel_improvement:
            if verbose:
                print(
                    f"Stopping at k={len(selected)}: best improvement {best_imp:,.3f} "
                    f"(rel {rel_imp:.3%}) below thresholds."
                )
            break

        selected.append(float(best_cand))
        selected = sorted(selected)
        current_sse = best_sse

        rows.append(
            {
                "k": len(selected),
                "added_m": float(best_cand),
                "added_km": float(best_cand) / 1000.0,
                "total_sse": float(current_sse),
                "improvement": float(best_imp),
                "rel_improvement": float(rel_imp),
            }
        )

        if verbose:
            print(
                f"k={len(selected):2d}  +{best_cand / 1000:8.2f} km   "
                f"SSE={current_sse:,.3f}   gain={best_imp:,.3f} ({rel_imp:.2%})"
            )

    _validate_breaks(ref_dist, selected)
    segments = _breaks_to_segments(ref_dist, selected)

    if len(segments) == 0:
        segments = [(0, len(ref_dist))]
    if segments[0][0] != 0 or segments[-1][1] != len(ref_dist):
        raise RuntimeError("Internal error: segments do not cover full profile.")
    if any(b <= a for a, b in segments):
        raise RuntimeError("Internal error: invalid segments produced.")

    hist = pd.DataFrame(rows)
    return BreakSelectionResult(
        breaks_m=selected,
        segments=segments,
        history=hist,
        candidates_used_m=[float(c) for c in candidates_m],
        windows_used=windows_used,
        feature_cols_used=feature_cols_used,
    )


def _breaks_to_segments_from_dist(dist: np.ndarray, breaks_m: Sequence[float]) -> List[Tuple[int, int]]:
    idx = np.searchsorted(dist, np.array(list(breaks_m), dtype=float), side="left")
    idx = [int(i) for i in idx if 0 < i < len(dist)]
    idx = sorted(set(idx))
    cuts = [0] + idx + [len(dist)]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def _cluster_break_rows_complete_linkage(
    breaks_df: pd.DataFrame,
    merge_threshold_m: float,
) -> List[List[int]]:
    if merge_threshold_m <= 0:
        raise ValueError("Consensus merge_threshold_m must be > 0.")
    if breaks_df.empty:
        return []

    clusters: List[List[int]] = []
    start = 0
    for i in range(1, len(breaks_df)):
        cluster_min = float(breaks_df.loc[start, "break_m"])
        next_break = float(breaks_df.loc[i, "break_m"])
        if (next_break - cluster_min) <= float(merge_threshold_m):
            continue
        clusters.append(list(range(start, i)))
        start = i
    clusters.append(list(range(start, len(breaks_df))))
    return clusters


def build_grid_consensus_from_break_rows(
    break_rows: Sequence[Dict[str, object]] | pd.DataFrame,
    ref_dist: np.ndarray,
    n_settings_valid: int,
    consensus_cfg: ConsensusConfig = DEFAULT_FROZEN_CONSENSUS_CONFIG,
    stable_support_frac_min: Optional[float] = DEFAULT_TUNING_STABLE_SUPPORT_FRAC_MIN,
    stable_support_count: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[float], List[Tuple[int, int]], float, Optional[int]]:
    if stable_support_frac_min is not None and stable_support_count is not None:
        raise ValueError("Specify either stable_support_frac_min or stable_support_count, not both.")
    if consensus_cfg.method != "complete_linkage":
        raise ValueError(
            f"Unsupported consensus method '{consensus_cfg.method}'. "
            "Only 'complete_linkage' is currently implemented."
        )

    if isinstance(break_rows, pd.DataFrame):
        breaks_df = break_rows.copy()
    else:
        breaks_df = pd.DataFrame(list(break_rows))

    if stable_support_frac_min is None and stable_support_count is None:
        stable_support_frac_min = DEFAULT_TUNING_STABLE_SUPPORT_FRAC_MIN

    if stable_support_frac_min is not None and not (0.0 < float(stable_support_frac_min) <= 1.0):
        raise ValueError("stable_support_frac_min must be in (0, 1].")

    if breaks_df.empty:
        effective_support_count = (
            min(int(stable_support_count), max(n_settings_valid, 1))
            if stable_support_count is not None
            else int(np.ceil(float(stable_support_frac_min) * max(n_settings_valid, 1) - 1e-12))
        )
        effective_support_frac = (
            float(effective_support_count / max(n_settings_valid, 1))
            if stable_support_count is not None
            else float(stable_support_frac_min)
        )
        empty_df = pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_center_m",
                "cluster_center_km",
                "cluster_min_m",
                "cluster_max_m",
                "cluster_span_m",
                "n_members",
                "n_settings_supported",
                "support_frac_grid",
                "settings_supported",
                "members_m",
                "consensus_method",
                "merge_threshold_m",
                "merge_threshold_km",
                "n_settings_valid_total",
            ]
        )
        return empty_df, [], [(0, len(ref_dist))], effective_support_frac, effective_support_count

    breaks_df = breaks_df.sort_values("break_m").reset_index(drop=True)
    clusters = _cluster_break_rows_complete_linkage(
        breaks_df=breaks_df,
        merge_threshold_m=float(consensus_cfg.merge_threshold_m),
    )

    consensus_rows: List[Dict[str, object]] = []
    for cluster_id, idx in enumerate(clusters):
        c = breaks_df.loc[idx].copy()
        settings_supported = unique_preserve_order(c["setting"].tolist())
        members_m = [float(v) for v in c["break_m"].tolist()]
        center_m = float(np.median(np.asarray(members_m, dtype=float)))
        consensus_rows.append(
            {
                "cluster_id": int(cluster_id),
                "cluster_center_m": center_m,
                "cluster_center_km": center_m / 1000.0,
                "cluster_min_m": float(np.min(members_m)),
                "cluster_max_m": float(np.max(members_m)),
                "cluster_span_m": float(np.max(members_m) - np.min(members_m)),
                "n_members": int(len(members_m)),
                "n_settings_supported": int(len(settings_supported)),
                "support_frac_grid": float(len(settings_supported) / max(n_settings_valid, 1)),
                "settings_supported": settings_supported,
                "members_m": members_m,
                "consensus_method": str(consensus_cfg.method),
                "merge_threshold_m": float(consensus_cfg.merge_threshold_m),
                "merge_threshold_km": float(consensus_cfg.merge_threshold_m) / 1000.0,
                "n_settings_valid_total": int(n_settings_valid),
            }
        )

    consensus_df = pd.DataFrame(consensus_rows).sort_values(
        ["n_settings_supported", "cluster_center_m"],
        ascending=[False, True],
    ).reset_index(drop=True)

    if stable_support_frac_min is not None:
        effective_support_frac = float(stable_support_frac_min)
        effective_support_count = int(np.ceil(effective_support_frac * max(n_settings_valid, 1) - 1e-12))
        stable_mask = consensus_df["support_frac_grid"] >= effective_support_frac
    else:
        effective_support_count = min(int(stable_support_count), max(n_settings_valid, 1))
        effective_support_frac = float(effective_support_count / max(n_settings_valid, 1))
        stable_mask = consensus_df["n_settings_supported"] >= effective_support_count

    stable_breaks_m = consensus_df.loc[stable_mask, "cluster_center_m"].astype(float).tolist()
    stable_breaks_m = sorted(set(round(float(b), 6) for b in stable_breaks_m))
    stable_segments = (
        _breaks_to_segments_from_dist(ref_dist, stable_breaks_m)
        if stable_breaks_m
        else [(0, len(ref_dist))]
    )

    return consensus_df, stable_breaks_m, stable_segments, effective_support_frac, effective_support_count


def select_segment_breaks_grid_from_results(
    results: Dict[str, object],
    windows: Optional[Sequence[str]] = None,
    feature_cols: Sequence[str] = FEATURE_COLS_ALL,
    candidate_source: str = "stability",
    candidate_freq_min: Optional[float] = None,
    min_support_frac_runs_values: Sequence[float] = (0.15, 0.20),
    min_windows_supported_values: Sequence[int] = (1, 2),
    min_support_frac_windows: float = 0.0,
    stability_tolerance_m: Optional[float] = None,
    min_spacing_m: float = 10_000.0,
    min_reach_len_m: Optional[float] = None,
    max_breaks: int = 30,
    stop_rel_improvement_values: Sequence[float] = (0.015, 0.02),
    stop_abs_improvement: float = 0.0,
    window_weights: Optional[Dict[str, float]] = None,
    consensus_cfg: ConsensusConfig = DEFAULT_FROZEN_CONSENSUS_CONFIG,
    stable_support_frac_min: float = DEFAULT_TUNING_STABLE_SUPPORT_FRAC_MIN,
    verbose: bool = False,
) -> BreakSelectionGridResult:
    """
    Run a small selector grid on a single PELT/stability result and derive a
    stable breakpoint set from consensus across selector settings.
    """

    def _format_float_label(x: float) -> str:
        s = f"{float(x):.3f}".rstrip("0").rstrip(".")
        return s.replace(".", "p")

    min_support_vals = sorted(set(float(v) for v in min_support_frac_runs_values))
    min_window_vals = sorted(set(int(v) for v in min_windows_supported_values))
    stop_rel_vals = sorted(set(float(v) for v in stop_rel_improvement_values))

    if len(min_support_vals) == 0 or len(min_window_vals) == 0 or len(stop_rel_vals) == 0:
        raise ValueError("Grid values for support/windows/stop_rel must be non-empty.")
    if not (0.0 < stable_support_frac_min <= 1.0):
        raise ValueError("stable_support_frac_min must be in (0, 1].")
    if consensus_cfg.method != "complete_linkage":
        raise ValueError(
            f"Unsupported consensus method '{consensus_cfg.method}'. "
            "Only 'complete_linkage' is currently implemented."
        )

    individual_results: Dict[str, BreakSelectionResult] = {}
    summary_rows: List[Dict[str, object]] = []
    break_rows: List[Dict[str, object]] = []

    n_settings_total = len(min_support_vals) * len(min_window_vals) * len(stop_rel_vals)
    n_settings_valid = 0
    if verbose:
        print(f"Running break-selection grid with {n_settings_total} selector settings.")

    for min_windows_supported in min_window_vals:
        for min_support_frac_runs in min_support_vals:
            for stop_rel_improvement in stop_rel_vals:
                label = (
                    f"mw{int(min_windows_supported)}_"
                    f"runs{_format_float_label(min_support_frac_runs)}_"
                    f"stop{_format_float_label(stop_rel_improvement)}"
                )
                try:
                    sel = select_segment_breaks_from_results(
                        results=results,
                        windows=windows,
                        feature_cols=feature_cols,
                        candidate_source=candidate_source,
                        candidate_freq_min=candidate_freq_min,
                        min_support_frac_runs=min_support_frac_runs,
                        min_windows_supported=min_windows_supported,
                        min_support_frac_windows=min_support_frac_windows,
                        stability_tolerance_m=stability_tolerance_m,
                        min_spacing_m=min_spacing_m,
                        min_reach_len_m=min_reach_len_m,
                        max_breaks=max_breaks,
                        stop_rel_improvement=stop_rel_improvement,
                        stop_abs_improvement=stop_abs_improvement,
                        window_weights=window_weights,
                        verbose=False,
                    )
                except ValueError as exc:
                    err_msg = str(exc)
                    if err_msg == "No candidates available after filtering.":
                        summary_rows.append(
                            {
                                "setting": label,
                                "min_windows_supported": int(min_windows_supported),
                                "min_support_frac_runs": float(min_support_frac_runs),
                                "stop_rel_improvement": float(stop_rel_improvement),
                                "n_breaks": 0,
                                "n_candidates": 0,
                                "final_sse": np.nan,
                                "breaks_m": [],
                                "status": "no_candidates_after_filtering",
                                "error_message": err_msg,
                            }
                        )
                        continue
                    raise

                individual_results[label] = sel
                n_settings_valid += 1

                final_sse = float(sel.history["total_sse"].iloc[-1]) if len(sel.history) else np.nan
                summary_rows.append(
                    {
                        "setting": label,
                        "min_windows_supported": int(min_windows_supported),
                        "min_support_frac_runs": float(min_support_frac_runs),
                        "stop_rel_improvement": float(stop_rel_improvement),
                        "n_breaks": int(len(sel.breaks_m)),
                        "n_candidates": int(len(sel.candidates_used_m)),
                        "final_sse": final_sse,
                        "breaks_m": [float(b) for b in sel.breaks_m],
                        "status": "ok",
                        "error_message": "",
                    }
                )
                for b in sel.breaks_m:
                    break_rows.append(
                        {
                            "setting": label,
                            "break_m": float(b),
                            "min_windows_supported": int(min_windows_supported),
                            "min_support_frac_runs": float(min_support_frac_runs),
                            "stop_rel_improvement": float(stop_rel_improvement),
                        }
                    )

    summary_df = pd.DataFrame(summary_rows).sort_values(
        ["min_windows_supported", "min_support_frac_runs", "stop_rel_improvement"]
    ).reset_index(drop=True)

    std_by_window: Dict[str, pd.DataFrame] = results["standardized_by_window"]  # type: ignore
    if windows is None:
        windows_used = sorted(list(std_by_window.keys()))
    else:
        windows_used = list(windows)
    ref_dist = std_by_window[windows_used[0]]["dist_m"].to_numpy(dtype=float)

    if len(break_rows) == 0:
        empty_consensus_df, _, stable_segments, effective_support_frac, effective_support_count = build_grid_consensus_from_break_rows(
            break_rows=[],
            ref_dist=ref_dist,
            n_settings_valid=n_settings_valid,
            consensus_cfg=consensus_cfg,
            stable_support_frac_min=stable_support_frac_min,
            stable_support_count=None,
        )
        return BreakSelectionGridResult(
            individual_results=individual_results,
            summary=summary_df,
            consensus=empty_consensus_df,
            stable_breaks_m=[],
            stable_segments=stable_segments,
            stable_support_frac_min=float(effective_support_frac),
            stable_support_count=effective_support_count,
            consensus_method=str(consensus_cfg.method),
            merge_threshold_m=float(consensus_cfg.merge_threshold_m),
            n_settings_valid=int(n_settings_valid),
        )

    consensus_df, stable_breaks_m, stable_segments, effective_support_frac, effective_support_count = (
        build_grid_consensus_from_break_rows(
            break_rows=break_rows,
            ref_dist=ref_dist,
            n_settings_valid=n_settings_valid,
            consensus_cfg=consensus_cfg,
            stable_support_frac_min=stable_support_frac_min,
            stable_support_count=None,
        )
    )

    if verbose:
        print(
            f"Grid consensus produced {len(stable_breaks_m)} stable breaks "
            f"using {consensus_cfg.method} at {consensus_cfg.merge_threshold_m / 1000.0:.2f} km "
            f"with support_frac_grid >= {effective_support_frac:.2f}."
        )

    return BreakSelectionGridResult(
        individual_results=individual_results,
        summary=summary_df,
        consensus=consensus_df,
        stable_breaks_m=[float(b) for b in stable_breaks_m],
        stable_segments=stable_segments,
        stable_support_frac_min=float(effective_support_frac),
        stable_support_count=effective_support_count,
        consensus_method=str(consensus_cfg.method),
        merge_threshold_m=float(consensus_cfg.merge_threshold_m),
        n_settings_valid=int(n_settings_valid),
    )


# =============================================================================
# Pipeline orchestration
# =============================================================================


def prepare_nodes_for_requested_features(
    nodes_df: pd.DataFrame,
    requested_feature_cols: Sequence[str],
    feat_cfg: FeatureConfig = FeatureConfig(),
    centerline=None,
    geom_df_10m: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    nodes_out = ensure_sorted_by_dist(nodes_df.copy(), feat_cfg.dist_col).reset_index(drop=True)
    requested = normalize_feature_cols(requested_feature_cols)
    need_curvature = "curv_int" in requested
    need_sinu = "sinu" in requested

    geom_out = geom_df_10m.copy() if geom_df_10m is not None else None

    if need_sinu and not {"x", "y"}.issubset(nodes_out.columns):
        if centerline is not None:
            nodes_out = add_node_xy_from_linestring(nodes_out, centerline, dist_col=feat_cfg.dist_col)
        elif geom_out is not None:
            nodes_out = add_node_xy_from_geom_df(nodes_out, geom_out, dist_col=feat_cfg.dist_col)
        else:
            raise ValueError(
                "Requested 'sinu' but nodes_df has no x/y and no centerline / geom_df_10m was provided."
            )

    if need_curvature and feat_cfg.curvature_col not in nodes_out.columns:
        if centerline is None and geom_out is None:
            raise ValueError(
                "Requested 'curv_int' but nodes_df has no curvature column and "
                "no centerline / geom_df_10m was provided."
            )
        nodes_out, geom_out = attach_geometry_metrics_to_nodes(
            nodes_df_200m=nodes_out,
            linestring=centerline,
            geom_df_10m=geom_out,
            ds_geom=feat_cfg.ds_geom,
            curvature_smooth_window=feat_cfg.curvature_smooth_window,
            sinuosity_window_m=feat_cfg.sinuosity_window_m,
            node_agg_half_window_m=feat_cfg.node_agg_half_window_m,
            curvature_agg=feat_cfg.curvature_agg,
        )

    return nodes_out, geom_out


def run_full_pipeline(
        nodes_df: pd.DataFrame,
        feat_cfg: FeatureConfig = FeatureConfig(),
        pelt_cfg: PeltConfig = PeltConfig(),
        pipe_cfg: PipelineConfig = PipelineConfig(),
        centerline=None,
        geom_df_10m: Optional[pd.DataFrame] = None,
        geometry_feature_cfg=None,
    ) -> Dict[str, object]:
    
    timings: Dict[str, float] = {}
    t_total = perf_counter()

    pelt_feature_cols = normalize_feature_cols(pelt_cfg.feature_cols)
    break_feature_cols = (
        normalize_feature_cols(pipe_cfg.break_selection.feature_cols)
        if pipe_cfg.break_selection.feature_cols is not None
        else pelt_feature_cols
    )
    feature_cols_to_compute = tuple(unique_preserve_order(pelt_feature_cols + break_feature_cols))
    geometry_feature_cols = tuple(c for c in feature_cols_to_compute if c in {"sinu", "curv_int"})

    t0 = perf_counter()
    effective_window_cfg = get_effective_window_selection_config(pipe_cfg)
    resolved_windows_m_nominal, window_selection_info = resolve_window_sizes(
        nodes_df=nodes_df,
        feat_cfg=feat_cfg,
        window_cfg=effective_window_cfg,
    )
    timings["window_selection"] = perf_counter() - t0

    t0 = perf_counter()
    feature_cols_for_node_prepare = tuple(
        c for c in feature_cols_to_compute if c not in {"sinu", "curv_int"}
    ) if geometry_feature_cols else feature_cols_to_compute
    if feature_cols_for_node_prepare:
        prepared_nodes_df, geom_df_used = prepare_nodes_for_requested_features(
            nodes_df=nodes_df,
            requested_feature_cols=feature_cols_for_node_prepare,
            feat_cfg=feat_cfg,
            centerline=centerline,
            geom_df_10m=geom_df_10m,
        )
    else:
        prepared_nodes_df = ensure_sorted_by_dist(nodes_df.copy(), feat_cfg.dist_col).reset_index(drop=True)
        geom_df_used = geom_df_10m.copy() if geom_df_10m is not None else None
    timings["prepare_nodes"] = perf_counter() - t0

    window_feasibility = summarize_window_feasibility(
        prepared_nodes_df,
        resolved_windows_m_nominal,
        dist_col=feat_cfg.dist_col,
    )
    resolved_windows_m = tuple(window_feasibility["feasible_windows_m"])
    window_selection_info = dict(window_selection_info)
    window_selection_info["nominal_windows_m"] = tuple(window_feasibility["nominal_windows_m"])
    window_selection_info["nominal_window_labels"] = [
        window_label(w) for w in window_feasibility["nominal_windows_m"]
    ]
    window_selection_info["windows_m"] = resolved_windows_m
    window_selection_info["window_labels"] = [window_label(w) for w in resolved_windows_m]
    window_selection_info["n_windows_nominal"] = int(window_feasibility["n_windows_nominal"])
    window_selection_info["n_windows_feasible"] = int(window_feasibility["n_windows_feasible"])
    window_selection_info["dropped_windows_m"] = tuple(window_feasibility["infeasible_windows_m"])
    window_selection_info["dropped_window_labels"] = [
        window_label(w) for w in window_feasibility["infeasible_windows_m"]
    ]

    if not resolved_windows_m:
        timings["total"] = perf_counter() - t_total
        if pipe_cfg.print_timings:
            print_timings(timings)
        return {
            "run_status": "skipped",
            "run_status_reason": "no_feasible_windows",
            "run_status_detail": (
                "Resolved minimum window exceeds the feasible reach length / node count."
                if not window_feasibility["first_window_feasible"]
                else "Window selection produced no feasible windows for this reach."
            ),
            "nodes_used": prepared_nodes_df,
            "geom_df_10m": geom_df_used,
            "geometry_features_by_window": None,
            "geometry_feature_diagnostics": {},
            "geometry_feature_source": (
                "PELT_geometry_features" if geometry_feature_cols else None
            ),
            "dist_col_used": feat_cfg.dist_col,
            "width_col_used": feat_cfg.width_col,
            "pelt_jump": pelt_cfg.jump,
            "spacing_m": float(window_feasibility["spacing_m"]),
            "window_selection": window_selection_info,
            "window_feasibility": window_feasibility,
            "resolved_windows_m": resolved_windows_m,
            "resolved_windows_m_nominal": tuple(window_feasibility["nominal_windows_m"]),
            "feature_cols_computed": feature_cols_to_compute,
            "pelt_feature_cols": pelt_feature_cols,
            "final_selection_feature_cols": break_feature_cols,
            "features_by_window": {},
            "standardized_by_window": {},
            "zstats_by_window": {},
            "per_window_runs": {},
            "all_runs": [],
            "stability": pd.DataFrame(),
            "final_selection": None,
            "final_breaks_m": [],
            "final_segments": [(0, len(prepared_nodes_df))],
            "final_selection_grid": None,
            "final_selection_grid_meta": {},
            "stable_breaks_m": [],
            "stable_segments": [(0, len(prepared_nodes_df))],
            "timings": timings,
        }

    geometry_features_by_window = None
    geometry_feature_diagnostics = {}
    if geometry_feature_cols:
        raw_windows = [float(w) for w in resolved_windows_m if float(w) <= 0.0]
        if raw_windows:
            raise ValueError(
                "Requested geometry features with raw window selection. "
                "Use positive PELT windows for 'sinu' and 'curv_int'."
            )
        if centerline is None:
            raise ValueError(
                "Requested geometry features require a concatenated LineString centerline. "
                "The newer PELT_geometry_features definitions are used for 'sinu' and "
                "'curv_int'; the older PELT.py geometry hooks are not used by run_full_pipeline."
            )

        t0 = perf_counter()
        import PELT_geometry_features as pgf

        geom_feat_cfg = geometry_feature_cfg
        if geom_feat_cfg is None:
            geom_feat_cfg = pgf.GeometryFeatureConfig(
                dist_col=feat_cfg.dist_col,
                width_col=feat_cfg.width_col,
            )

        geometry_features_by_window, geometry_feature_diagnostics = (
            pgf.compute_geometry_features_for_windows(
                centerline=centerline,
                nodes_df=prepared_nodes_df,
                windows_m=resolved_windows_m,
                cfg=geom_feat_cfg,
                return_diagnostics=True,
            )
        )
        if geom_df_used is None:
            geom_df_used = geometry_feature_diagnostics.get("geom_base")
        timings["geometry_features"] = perf_counter() - t0

    t0 = perf_counter()
    features_by_window = build_multiscale_features(
        nodes_df=prepared_nodes_df,
        windows_m=resolved_windows_m,
        cfg=feat_cfg,
        min_pts_in_window=pipe_cfg.min_pts_in_window,
        feature_cols_to_compute=feature_cols_to_compute,
        geometry_features_by_window=geometry_features_by_window,
        report_health=pipe_cfg.report_feature_health,
    )
    timings["feature_building"] = perf_counter() - t0

    t0 = perf_counter()
    standardized_by_window: Dict[str, pd.DataFrame] = {}
    zstats_by_window: Dict[str, Dict[str, Tuple[float, float]]] = {}
    for wlab, fdf in features_by_window.items():
        if pipe_cfg.standardize:
            fdf_z, stats = zscore_df(fdf, feature_cols_to_compute)
            standardized_by_window[wlab] = fdf_z
            zstats_by_window[wlab] = stats
        else:
            standardized_by_window[wlab] = fdf.copy()
            zstats_by_window[wlab] = {}
    timings["standardization"] = perf_counter() - t0

    t0 = perf_counter()
    per_window_runs: Dict[str, List[Dict[str, object]]] = {}
    all_runs: List[Dict[str, object]] = []
    for wlab, fdfz in standardized_by_window.items():
        runs = penalty_sweep(fdfz, pelt_cfg=PeltConfig(feature_cols=pelt_feature_cols, min_size_pts=pelt_cfg.min_size_pts, jump=pelt_cfg.jump), penalties=pipe_cfg.penalties)
        for r in runs:
            r["window_label"] = wlab
        per_window_runs[wlab] = runs
        all_runs.extend(runs)
    timings["pelt_sweeps"] = perf_counter() - t0

    t0 = perf_counter()
    stability = compute_breakpoint_stability(all_runs, tolerance_m=pipe_cfg.stability_tolerance_m)
    timings["breakpoint_support"] = perf_counter() - t0

    spacing_m = infer_spacing_m(prepared_nodes_df[feat_cfg.dist_col].to_numpy(dtype=float))
    n_windows_effective = int(len(standardized_by_window))
    break_min_windows_supported_effective = min(
        int(pipe_cfg.break_selection.min_windows_supported),
        max(n_windows_effective, 1),
    )
    grid_min_windows_supported_values_effective = tuple(
        sorted(
            {
                min(int(v), max(n_windows_effective, 1))
                for v in pipe_cfg.break_selection_grid.min_windows_supported_values
            }
        )
    ) if pipe_cfg.break_selection_grid.min_windows_supported_values else tuple()

    partial_results: Dict[str, object] = {
        "nodes_used": prepared_nodes_df,
        "dist_col_used": feat_cfg.dist_col,
        "width_col_used": feat_cfg.width_col,
        "pelt_jump": pelt_cfg.jump,
        "spacing_m": spacing_m,
        "window_feasibility": window_feasibility,
        "features_by_window": features_by_window,
        "standardized_by_window": standardized_by_window,
        "zstats_by_window": zstats_by_window,
        "per_window_runs": per_window_runs,
        "all_runs": all_runs,
        "stability": stability,
    }

    final_selection: Optional[BreakSelectionResult] = None
    final_breaks_m: List[float] = []
    final_segments: List[Tuple[int, int]] = [(0, len(prepared_nodes_df))]
    final_selection_grid: Optional[BreakSelectionGridResult] = None
    stable_breaks_m: List[float] = []
    stable_segments: List[Tuple[int, int]] = [(0, len(prepared_nodes_df))]

    if pipe_cfg.break_selection.enabled:
        t0 = perf_counter()
        final_selection = select_segment_breaks_from_results(
            results=partial_results,
            windows=pipe_cfg.break_selection.windows,
            feature_cols=break_feature_cols,
            candidate_source=pipe_cfg.break_selection.candidate_source,
            candidate_freq_min=pipe_cfg.break_selection.candidate_freq_min,
            min_support_frac_runs=pipe_cfg.break_selection.min_support_frac_runs,
            min_windows_supported=break_min_windows_supported_effective,
            min_support_frac_windows=pipe_cfg.break_selection.min_support_frac_windows,
            stability_tolerance_m=pipe_cfg.break_selection.stability_tolerance_m,
            min_spacing_m=pipe_cfg.break_selection.min_spacing_m,
            min_reach_len_m=pipe_cfg.break_selection.min_reach_len_m,
            max_breaks=pipe_cfg.break_selection.max_breaks,
            stop_rel_improvement=pipe_cfg.break_selection.stop_rel_improvement,
            stop_abs_improvement=pipe_cfg.break_selection.stop_abs_improvement,
            window_weights=pipe_cfg.break_selection.window_weights,
            verbose=pipe_cfg.break_selection.verbose,
        )
        final_breaks_m = final_selection.breaks_m
        final_segments = final_selection.segments
        timings["final_break_selection"] = perf_counter() - t0

    if pipe_cfg.break_selection_grid.enabled:
        t0 = perf_counter()
        final_selection_grid = select_segment_breaks_grid_from_results(
            results=partial_results,
            windows=pipe_cfg.break_selection.windows,
            feature_cols=break_feature_cols,
            candidate_source=pipe_cfg.break_selection.candidate_source,
            candidate_freq_min=pipe_cfg.break_selection.candidate_freq_min,
            min_support_frac_runs_values=pipe_cfg.break_selection_grid.min_support_frac_runs_values,
            min_windows_supported_values=grid_min_windows_supported_values_effective,
            min_support_frac_windows=pipe_cfg.break_selection.min_support_frac_windows,
            stability_tolerance_m=pipe_cfg.break_selection.stability_tolerance_m,
            min_spacing_m=pipe_cfg.break_selection.min_spacing_m,
            min_reach_len_m=pipe_cfg.break_selection.min_reach_len_m,
            max_breaks=pipe_cfg.break_selection.max_breaks,
            stop_rel_improvement_values=pipe_cfg.break_selection_grid.stop_rel_improvement_values,
            stop_abs_improvement=pipe_cfg.break_selection.stop_abs_improvement,
            window_weights=pipe_cfg.break_selection.window_weights,
            consensus_cfg=pipe_cfg.break_selection_grid.consensus,
            stable_support_frac_min=pipe_cfg.break_selection_grid.stable_support_frac_min,
            verbose=pipe_cfg.break_selection_grid.verbose,
        )
        stable_breaks_m = final_selection_grid.stable_breaks_m
        stable_segments = final_selection_grid.stable_segments
        timings["final_break_selection_grid"] = perf_counter() - t0

    timings["total"] = perf_counter() - t_total
    if pipe_cfg.print_timings:
        print_timings(timings)

    return {
        "run_status": "ok",
        "run_status_reason": "",
        "run_status_detail": "",
        "nodes_used": prepared_nodes_df,
        "geom_df_10m": geom_df_used,
        "geometry_features_by_window": geometry_features_by_window,
        "geometry_feature_diagnostics": geometry_feature_diagnostics,
        "geometry_feature_source": (
            "PELT_geometry_features" if geometry_feature_cols else None
        ),
        "dist_col_used": feat_cfg.dist_col,
        "width_col_used": feat_cfg.width_col,
        "pelt_jump": pelt_cfg.jump,
        "spacing_m": spacing_m,
        "window_selection": window_selection_info,
        "window_feasibility": window_feasibility,
        "resolved_windows_m": resolved_windows_m,
        "resolved_windows_m_nominal": tuple(window_feasibility["nominal_windows_m"]),
        "feature_cols_computed": feature_cols_to_compute,
        "pelt_feature_cols": pelt_feature_cols,
        "final_selection_feature_cols": break_feature_cols,
        "features_by_window": features_by_window,
        "standardized_by_window": standardized_by_window,
        "zstats_by_window": zstats_by_window,
        "per_window_runs": per_window_runs,
        "all_runs": all_runs,
        "stability": stability,
        "final_selection": final_selection,
        "final_breaks_m": final_breaks_m,
        "final_segments": final_segments,
        "final_selection_grid": final_selection_grid,
        "final_selection_grid_meta": (
            {
                "consensus_method": final_selection_grid.consensus_method,
                "merge_threshold_m": final_selection_grid.merge_threshold_m,
                "stable_support_frac_min_effective": final_selection_grid.stable_support_frac_min,
                "stable_support_count": final_selection_grid.stable_support_count,
                "n_settings_valid": final_selection_grid.n_settings_valid,
                "min_windows_supported_effective": break_min_windows_supported_effective,
                "grid_min_windows_supported_values_effective": list(grid_min_windows_supported_values_effective),
            }
            if final_selection_grid is not None
            else {
                "min_windows_supported_effective": break_min_windows_supported_effective,
                "grid_min_windows_supported_values_effective": list(grid_min_windows_supported_values_effective),
            }
        ),
        "stable_breaks_m": stable_breaks_m,
        "stable_segments": stable_segments,
        "timings": timings,
    }


# =============================================================================
# Optional plotting helper (quick QA)
# =============================================================================


def plot_features_with_breaks(
    features_df: pd.DataFrame,
    break_dist_m: Sequence[float],
    cols: Sequence[str] = FEATURE_COLS_ALL,
    dist_col: str = "dist_m",
    title: str = "",
    ):
    import matplotlib.pyplot as plt

    cols = [c for c in cols if c in features_df.columns]
    if len(cols) == 0:
        raise ValueError("No requested feature columns are present in features_df.")
    dist_km = features_df[dist_col].to_numpy(dtype=float) / 1000.0
    n = len(cols)
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, c in zip(axes, cols):
        ax.plot(dist_km, features_df[c].to_numpy(dtype=float))
        for bd in break_dist_m:
            ax.axvline(float(bd) / 1000.0, linewidth=1)
        ax.set_ylabel(c)

    axes[-1].set_xlabel("Distance downstream (km)")
    if title:
        fig.suptitle(title)
    plt.tight_layout()
    plt.show()



def compare_pelt_results(results_compare, tol_m=2000, make_plot=True, plot_title = ''):
    """
    Compare multiple PELT run outputs.

    Parameters
    ----------
    results_compare : dict
        Mapping like:
        {
            "support2": results_support2,
            "stop002": results_stop002,
            ...
        }
    tol_m : float
        Distance tolerance for clustering final breaks across runs.
    make_plot : bool
        If True, draw a breakpoint comparison plot.

    Returns
    -------
    summary_df : pd.DataFrame
    cluster_df : pd.DataFrame
    candidate_df : pd.DataFrame
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    def cluster_breaks_across_runs(results_compare, tol_m=2000):
        rows = []
        for name, r in results_compare.items():
            for b in r["final_breaks_m"]:
                rows.append({"run": name, "break_m": float(b)})

        df = pd.DataFrame(rows).sort_values("break_m").reset_index(drop=True)
        if df.empty:
            return pd.DataFrame(
                columns=["cluster_id", "center_km", "n_runs_supported", "runs_supported", "members_km"]
            )

        clusters = []
        cur = [0]
        for i in range(1, len(df)):
            if abs(df.loc[i, "break_m"] - df.loc[i - 1, "break_m"]) <= tol_m:
                cur.append(i)
            else:
                clusters.append(cur)
                cur = [i]
        clusters.append(cur)

        out = []
        for cid, idx in enumerate(clusters):
            c = df.loc[idx]
            out.append({
                "cluster_id": cid,
                "center_km": c["break_m"].median() / 1000.0,
                "n_runs_supported": c["run"].nunique(),
                "runs_supported": sorted(c["run"].unique().tolist()),
                "members_km": [round(x / 1000.0, 2) for x in c["break_m"].tolist()],
            })

        return pd.DataFrame(out).sort_values(
            ["n_runs_supported", "center_km"],
            ascending=[False, True]
        ).reset_index(drop=True)

    summary_rows = []
    candidate_rows = []

    for name, r in results_compare.items():
        fs = r.get("final_selection", None)
        hist = fs.history if fs is not None else pd.DataFrame()

        summary_rows.append({
            "run": name,
            "n_breaks": len(r["final_breaks_m"]),
            "breaks_km": [round(b / 1000.0, 2) for b in r["final_breaks_m"]],
            "n_candidates": len(fs.candidates_used_m) if fs is not None else np.nan,
            "windows_used": r["window_selection"]["window_labels"],
            "resolved_windows_km": [round(w / 1000.0, 2) for w in r["resolved_windows_m"]],
            "final_sse": hist["total_sse"].iloc[-1] if len(hist) else np.nan,
            "total_time_s": r["timings"]["total"],
            "pelt_sweeps_s": r["timings"]["pelt_sweeps"],
            "final_selection_s": r["timings"].get("final_break_selection", np.nan),
        })

        candidate_rows.append({
            "run": name,
            "n_candidates": len(fs.candidates_used_m) if fs is not None else np.nan,
            "n_final_breaks": len(r["final_breaks_m"]),
            "candidate_to_break_ratio": (
                len(fs.candidates_used_m) / max(len(r["final_breaks_m"]), 1)
                if fs is not None else np.nan
            ),
        })

    summary_df = pd.DataFrame(summary_rows).sort_values("run").reset_index(drop=True)
    cluster_df = cluster_breaks_across_runs(results_compare, tol_m=tol_m)
    candidate_df = pd.DataFrame(candidate_rows).sort_values("run").reset_index(drop=True)



    if make_plot and len(results_compare) > 0:



        max_dist_km = max(
            r["nodes_used"]["dist_m"].max() / 1000.0
            for r in results_compare.values()
        )

        fig, ax = plt.subplots(figsize=(16, 6))
        run_names = list(results_compare.keys())

        for i, name in enumerate(run_names):
            y = len(run_names) - i
            breaks_km = np.array(results_compare[name]["final_breaks_m"], dtype=float) / 1000.0

            ax.hlines(y, xmin=0, xmax=max_dist_km, color="0.88", linewidth=1)
            if len(breaks_km) > 0:
                ax.scatter(breaks_km, np.full_like(breaks_km, y), s=80)


        
        for _, row in cluster_df.iterrows():
            alpha_val = min(0.9, 0.12 + 0.05 * row["n_runs_supported"])
            ax.axvline(
                row["center_km"],
                color="tab:red",
                alpha=alpha_val,
                linewidth=2,
            )



        ax.set_xlim(0, max_dist_km)
        ax.set_yticks(range(1, len(run_names) + 1))
        ax.set_yticklabels(run_names[::-1])
        ax.set_xlabel("Distance downstream (km)")
        ax.set_ylabel("Run setup")
        ax.set_title("Final Breakpoints Across Alternative Setups")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        
        outdir = Path("test_figures")
        outdir.mkdir(exist_ok=True, parents=True)
        fname = outdir / f"PELT_config_{plot_title}"

        summary_df.to_csv(f'{fname}_summary_df.csv')
        cluster_df.to_csv(f'{fname}_cluster_df.csv')
        candidate_df.to_csv(f'{fname}_candidate_df.csv')


        fig.savefig(f'{fname}.png', dpi=200, bbox_inches="tight")
        plt.show()

    return summary_df, cluster_df, candidate_df

def plot_pelt_grid_results(
    results,
    core_min=None,
    medium_min=0.50,
    make_plot=True,
    plot_title="",
    save=True,
    outdir="test_figures",
):
    """
    Visualize and summarize grid-based final breakpoint selection.

    Expects:
        results["final_selection_grid"] from run_full_pipeline(...)

    Categories:
        core   : support_frac_grid >= core_min
        medium : medium_min <= support_frac_grid < core_min
        weak   : support_frac_grid < medium_min

    Returns
    -------
    summary_df : pd.DataFrame
        One row per selector-grid setting.
    consensus_df : pd.DataFrame
        One row per clustered breakpoint across grid settings.
    class_df : pd.DataFrame
        Consensus table with support class labels.
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from pathlib import Path

    grid = results.get("final_selection_grid", None)
    if grid is None:
        raise ValueError(
            "results['final_selection_grid'] is None. "
            "Enable break_selection_grid in run_full_pipeline first."
        )

    summary_df = grid.summary.copy()
    consensus_df = grid.consensus.copy()

    if core_min is None:
        core_min = float(grid.stable_support_frac_min)

    if summary_df.empty:
        raise ValueError("Grid summary is empty.")
    if consensus_df.empty:
        raise ValueError("Grid consensus is empty.")

    # Normalize distance column names
    if "cluster_center_km" not in consensus_df.columns:
        consensus_df["cluster_center_km"] = consensus_df["cluster_center_m"] / 1000.0

    # Label consensus support classes
    def classify_support(x):
        if x >= core_min:
            return "core"
        if x >= medium_min:
            return "medium"
        return "weak"

    class_df = consensus_df.copy()
    class_df["support_class"] = class_df["support_frac_grid"].apply(classify_support)

    # Sort settings in a stable, interpretable way
    sort_cols = [c for c in ["min_windows_supported", "min_support_frac_runs", "stop_rel_improvement"] if c in summary_df.columns]
    if sort_cols:
        summary_df = summary_df.sort_values(sort_cols).reset_index(drop=True)
    else:
        summary_df = summary_df.sort_values("setting").reset_index(drop=True)

    # Build per-setting breakpoint table from the actual grid results
    point_rows = []
    for setting in summary_df["setting"].tolist():
        sel = grid.individual_results.get(setting)
        if sel is None:
            continue
        for b in sel.breaks_m:
            point_rows.append({
                "setting": setting,
                "break_m": float(b),
                "break_km": float(b) / 1000.0,
            })
    points_df = pd.DataFrame(point_rows)

    # Add a compact per-cluster summary
    class_df["n_settings_total"] = len(summary_df)
    class_df["support_pct"] = 100.0 * class_df["support_frac_grid"]

    # Stable/core set according to the grid threshold
    stable_df = class_df[class_df["support_frac_grid"] >= core_min].copy()

    if make_plot:
        max_dist_km = max(
            float(r["nodes_used"]["dist_m"].max()) / 1000.0
            for r in [results]
        )

        fig, (ax1, ax2) = plt.subplots(
            2,
            1,
            figsize=(17, 9),
            sharex=True,
            gridspec_kw={"height_ratios": [3.2, 1.4]},
        )

        setting_names = summary_df["setting"].tolist()
        colors = {"core": "tab:red", "medium": "tab:orange", "weak": "0.65"}

        # Top panel: all selected breaks for each selector-grid setting
        for i, setting in enumerate(setting_names):
            y = len(setting_names) - i
            sub = points_df[points_df["setting"] == setting]

            ax1.hlines(y, xmin=0, xmax=max_dist_km, color="0.90", linewidth=1)
            if not sub.empty:
                ax1.scatter(
                    sub["break_km"],
                    np.full(len(sub), y),
                    s=55,
                    color="black",
                    alpha=0.80,
                    zorder=3,
                )

        # Consensus lines
        for _, row in class_df.sort_values("support_frac_grid").iterrows():
            cls = row["support_class"]
            lw = 1.2 + 2.2 * float(row["support_frac_grid"])
            alpha = {"core": 0.55, "medium": 0.35, "weak": 0.18}[cls]

            ax1.axvline(
                row["cluster_center_km"],
                color=colors[cls],
                linewidth=lw,
                alpha=alpha,
                zorder=1,
            )

        # Highlight stable/core breaks
        if not stable_df.empty:
            stable_x = stable_df["cluster_center_km"].to_numpy(dtype=float)
            ax1.scatter(
                stable_x,
                np.full_like(stable_x, len(setting_names) + 0.6),
                marker="*",
                s=180,
                color="darkred",
                edgecolor="white",
                linewidth=0.7,
                zorder=4,
                label=f"stable breaks (>= {core_min:.2f})",
            )
            ax1.legend(loc="upper right", frameon=True)

        ax1.set_xlim(0, max_dist_km)
        ax1.set_yticks(range(1, len(setting_names) + 1))
        ax1.set_yticklabels(setting_names[::-1])
        ax1.set_ylabel("Grid setting")
        ax1.set_title("Grid-Based Breakpoint Consensus" if plot_title == "" else f"Grid-Based Breakpoint Consensus: {plot_title}")
        ax1.grid(axis="x", alpha=0.25)

        # Bottom panel: support fraction by breakpoint location
        for cls in ["weak", "medium", "core"]:
            sub = class_df[class_df["support_class"] == cls]
            if sub.empty:
                continue

            ax2.vlines(
                sub["cluster_center_km"],
                0.0,
                sub["support_frac_grid"],
                color=colors[cls],
                linewidth=2,
                alpha=0.75,
            )
            ax2.scatter(
                sub["cluster_center_km"],
                sub["support_frac_grid"],
                color=colors[cls],
                s=70,
                label=cls if cls not in ax2.get_legend_handles_labels()[1] else None,
            )

        ax2.axhline(core_min, color="tab:red", linestyle="--", linewidth=1.2, alpha=0.8)
        ax2.axhline(medium_min, color="tab:orange", linestyle="--", linewidth=1.2, alpha=0.8)

        ax2.set_ylim(0, 1.05)
        ax2.set_ylabel("Grid support")
        ax2.set_xlabel("Distance downstream (km)")
        ax2.grid(axis="both", alpha=0.25)
        ax2.legend(loc="upper right", frameon=True)

        plt.tight_layout()

        if save:
            outdir = Path(outdir)
            outdir.mkdir(exist_ok=True, parents=True)

            suffix = f"_{plot_title}" if plot_title else ""
            stem = outdir / f"PELT_grid{suffix}"

            summary_df.to_csv(f"{stem}_summary_df.csv", index=False)
            consensus_df.to_csv(f"{stem}_consensus_df.csv", index=False)
            class_df.to_csv(f"{stem}_classified_df.csv", index=False)
            fig.savefig(f"{stem}.png", dpi=200, bbox_inches="tight")

        plt.show()

    return summary_df, consensus_df, class_df


def extract_pelt_grid_analysis_tables(
    results: Dict[str, object],
    reach_id: object | None = None,
    window_version: str | None = None,
    medium_min: float = 0.50,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build comparable analysis tables from one grid-enabled PELT result.

    Returns
    -------
    settings_df : pd.DataFrame
        One row per selector-grid setting.
    consensus_df : pd.DataFrame
        One row per consensus breakpoint cluster, with support class labels.
    run_summary_df : pd.DataFrame
        Single-row summary for the (reach, window_version) run.
    """
    grid = results.get("final_selection_grid", None)
    if grid is None:
        raise ValueError(
            "results['final_selection_grid'] is None. "
            "Enable break_selection_grid in run_full_pipeline first."
        )

    summary_df = grid.summary.copy()
    consensus_df = grid.consensus.copy()

    if summary_df.empty:
        raise ValueError("Grid summary is empty.")

    core_min = float(grid.stable_support_frac_min)
    stable_support_count = getattr(grid, "stable_support_count", None)
    consensus_method = str(getattr(grid, "consensus_method", "unknown"))
    merge_threshold_m = float(getattr(grid, "merge_threshold_m", np.nan))

    if "cluster_center_km" not in consensus_df.columns and "cluster_center_m" in consensus_df.columns:
        consensus_df["cluster_center_km"] = consensus_df["cluster_center_m"] / 1000.0

    def _classify_support(x: float) -> str:
        if float(x) >= core_min:
            return "core"
        if float(x) >= medium_min:
            return "medium"
        return "weak"

    if not consensus_df.empty:
        consensus_df = consensus_df.copy()
        consensus_df["support_class"] = consensus_df["support_frac_grid"].apply(_classify_support)
        consensus_df["support_pct"] = 100.0 * consensus_df["support_frac_grid"]
    else:
        consensus_df = pd.DataFrame(
            columns=[
                "cluster_id",
                "cluster_center_m",
                "cluster_center_km",
                "cluster_min_m",
                "cluster_max_m",
                "cluster_span_m",
                "n_members",
                "n_settings_supported",
                "support_frac_grid",
                "settings_supported",
                "members_m",
                "support_class",
                "support_pct",
            ]
        )

    window_info = dict(results.get("window_selection", {}))
    resolved_windows_m = tuple(float(w) for w in results.get("resolved_windows_m", ()))
    nominal_windows_m = tuple(
        float(w) for w in window_info.get("nominal_windows_m", results.get("resolved_windows_m_nominal", resolved_windows_m))
    )
    resolved_windows_km = [round(w / 1000.0, 6) for w in resolved_windows_m]
    nominal_windows_km = [round(w / 1000.0, 6) for w in nominal_windows_m]
    window_labels = list(window_info.get("window_labels", [window_label(w) for w in resolved_windows_m]))
    nominal_window_labels = list(
        window_info.get("nominal_window_labels", [window_label(w) for w in nominal_windows_m])
    )
    n_windows_total = int(len(window_labels))
    n_windows_nominal = int(window_info.get("n_windows_nominal", len(nominal_window_labels)))
    window_method = str(window_info.get("method", "unknown"))

    if window_version is None:
        if window_method == "raw":
            window_version = "raw"
        else:
            window_version = f"w{n_windows_total}"

    settings_df = summary_df.copy()
    if "status" not in settings_df.columns:
        settings_df["status"] = "ok"
    if "error_message" not in settings_df.columns:
        settings_df["error_message"] = ""
    settings_df["reach_id"] = reach_id
    settings_df["window_version"] = str(window_version)
    settings_df["window_method"] = window_method
    settings_df["n_windows_total"] = n_windows_total
    settings_df["n_windows_nominal"] = n_windows_nominal
    settings_df["window_labels"] = [window_labels] * len(settings_df)
    settings_df["nominal_window_labels"] = [nominal_window_labels] * len(settings_df)
    settings_df["resolved_windows_m"] = [list(resolved_windows_m)] * len(settings_df)
    settings_df["nominal_windows_m"] = [list(nominal_windows_m)] * len(settings_df)
    settings_df["resolved_windows_km"] = [resolved_windows_km] * len(settings_df)
    settings_df["nominal_windows_km"] = [nominal_windows_km] * len(settings_df)
    n_grid_settings_attempted_total = int(len(settings_df))
    n_grid_settings_valid_total = int((settings_df["status"].astype(str) == "ok").sum())
    settings_df["n_grid_settings_total"] = n_grid_settings_attempted_total
    settings_df["n_grid_settings_valid_total"] = n_grid_settings_valid_total
    settings_df["n_grid_settings_failed_total"] = n_grid_settings_attempted_total - n_grid_settings_valid_total
    settings_df["consensus_method"] = consensus_method
    settings_df["merge_threshold_m"] = merge_threshold_m
    settings_df["merge_threshold_km"] = merge_threshold_m / 1000.0 if np.isfinite(merge_threshold_m) else np.nan
    settings_df["stable_support_frac_min"] = core_min
    settings_df["stable_support_count"] = stable_support_count
    settings_df["medium_support_frac_min"] = float(medium_min)
    settings_df["min_support_frac_windows_effective"] = (
        settings_df["min_windows_supported"].astype(float) / max(n_windows_total, 1)
    )
    if "breaks_m" in settings_df.columns:
        settings_df["breaks_km"] = settings_df["breaks_m"].apply(
            lambda xs: [round(float(x) / 1000.0, 6) for x in xs]
        )

    consensus_df["reach_id"] = reach_id
    consensus_df["window_version"] = str(window_version)
    consensus_df["window_method"] = window_method
    consensus_df["n_windows_total"] = n_windows_total
    consensus_df["n_windows_nominal"] = n_windows_nominal
    consensus_df["window_labels"] = [window_labels] * len(consensus_df)
    consensus_df["nominal_window_labels"] = [nominal_window_labels] * len(consensus_df)
    consensus_df["resolved_windows_m"] = [list(resolved_windows_m)] * len(consensus_df)
    consensus_df["nominal_windows_m"] = [list(nominal_windows_m)] * len(consensus_df)
    consensus_df["resolved_windows_km"] = [resolved_windows_km] * len(consensus_df)
    consensus_df["nominal_windows_km"] = [nominal_windows_km] * len(consensus_df)
    consensus_df["n_grid_settings_total"] = n_grid_settings_attempted_total
    consensus_df["n_grid_settings_valid_total"] = n_grid_settings_valid_total
    consensus_df["n_grid_settings_failed_total"] = n_grid_settings_attempted_total - n_grid_settings_valid_total
    consensus_df["consensus_method"] = consensus_method
    consensus_df["merge_threshold_m"] = merge_threshold_m
    consensus_df["merge_threshold_km"] = merge_threshold_m / 1000.0 if np.isfinite(merge_threshold_m) else np.nan
    consensus_df["stable_support_frac_min"] = core_min
    consensus_df["stable_support_count"] = stable_support_count
    consensus_df["medium_support_frac_min"] = float(medium_min)

    core_df = consensus_df[consensus_df["support_class"] == "core"]
    medium_df = consensus_df[consensus_df["support_class"] == "medium"]
    weak_df = consensus_df[consensus_df["support_class"] == "weak"]

    penalties = sorted(
        {
            float(r.get("penalty"))
            for r in results.get("all_runs", [])
            if np.isfinite(float(r.get("penalty", np.nan)))
        }
    )

    run_summary = {
        "reach_id": reach_id,
        "window_version": str(window_version),
        "window_method": window_method,
        "n_windows_total": n_windows_total,
        "n_windows_nominal": n_windows_nominal,
        "window_labels": window_labels,
        "nominal_window_labels": nominal_window_labels,
        "resolved_windows_m": list(resolved_windows_m),
        "nominal_windows_m": list(nominal_windows_m),
        "resolved_windows_km": resolved_windows_km,
        "nominal_windows_km": nominal_windows_km,
        "n_penalties_total": int(len(penalties)),
        "penalties": penalties,
        "n_grid_settings_total": n_grid_settings_attempted_total,
        "n_grid_settings_valid_total": n_grid_settings_valid_total,
        "n_grid_settings_failed_total": n_grid_settings_attempted_total - n_grid_settings_valid_total,
        "grid_min_windows_supported_values": sorted(settings_df["min_windows_supported"].astype(int).unique().tolist()),
        "grid_min_support_frac_runs_values": sorted(settings_df["min_support_frac_runs"].astype(float).unique().tolist()),
        "grid_stop_rel_improvement_values": sorted(settings_df["stop_rel_improvement"].astype(float).unique().tolist()),
        "grid_min_support_frac_windows_effective_values": sorted(
            settings_df["min_support_frac_windows_effective"].astype(float).unique().tolist()
        ),
        "consensus_method": consensus_method,
        "merge_threshold_m": merge_threshold_m,
        "merge_threshold_km": merge_threshold_m / 1000.0 if np.isfinite(merge_threshold_m) else np.nan,
        "stable_support_frac_min": core_min,
        "stable_support_count": stable_support_count,
        "medium_support_frac_min": float(medium_min),
        "core_count": int(len(core_df)),
        "medium_count": int(len(medium_df)),
        "weak_count": int(len(weak_df)),
        "stable_breaks_m": [float(b) for b in grid.stable_breaks_m],
        "stable_breaks_km": [round(float(b) / 1000.0, 6) for b in grid.stable_breaks_m],
        "n_breaks_min": int(settings_df["n_breaks"].min()),
        "n_breaks_max": int(settings_df["n_breaks"].max()),
        "n_candidates_min": int(settings_df["n_candidates"].min()),
        "n_candidates_max": int(settings_df["n_candidates"].max()),
        "final_sse_min": float(settings_df["final_sse"].min()),
        "final_sse_max": float(settings_df["final_sse"].max()),
        "max_cluster_span_m": float(consensus_df["cluster_span_m"].max()) if len(consensus_df) else np.nan,
        "mean_core_span_m": float(core_df["cluster_span_m"].mean()) if len(core_df) else np.nan,
        "total_time_s": float(results.get("timings", {}).get("total", np.nan)),
        "pelt_sweeps_s": float(results.get("timings", {}).get("pelt_sweeps", np.nan)),
        "final_break_selection_s": float(results.get("timings", {}).get("final_break_selection", np.nan)),
        "final_break_selection_grid_s": float(results.get("timings", {}).get("final_break_selection_grid", np.nan)),
    }
    run_summary_df = pd.DataFrame([run_summary])

    return settings_df, consensus_df, run_summary_df
