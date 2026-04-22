from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    from shapely.geometry import LineString, MultiLineString, Point
except Exception:  # pragma: no cover - import-time guard for environments without shapely
    LineString = None
    MultiLineString = None
    Point = None

try:
    from scipy.ndimage import gaussian_filter1d
except Exception:  # pragma: no cover - scipy is optional
    gaussian_filter1d = None


@dataclass
class GeometryFeatureConfig:
    dist_col: str = "dist_m"
    width_col: str = "multi_width"

    # Geometry sampling. This is computational resolution, not the morphology scale.
    ds_geom_m: float = 10.0

    # Coordinate smoothing before curvature derivatives.
    # Use "piecewise_width_gaussian" for the current preferred test framework.
    smoothing_method: str = "piecewise_width_gaussian"  # "none", "fixed_gaussian", "piecewise_width_gaussian"
    fixed_sigma_m: float = 50.0
    sigma_width_frac: float = 0.25
    sigma_min_m: float = 20.0
    sigma_max_m: float = 150.0
    width_group_rel_tol: float = 0.20
    width_repr: str = "median"  # "median" or "mean"
    blend_m: Optional[float] = None

    # Window-level feature summaries.
    min_sinuosity_window_m: float = 0.0
    min_curvature_window_m: float = 100.0
    min_geom_samples: int = 10
    fill_edges: bool = True


def _require_cols(df: pd.DataFrame, cols: Sequence[str], name: str = "df") -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def _as_linestring(centerline):
    if LineString is None:
        raise ImportError("shapely is required for geometry feature computation.")
    if not isinstance(centerline, LineString):
        raise TypeError("centerline must be a shapely.geometry.LineString.")
    if centerline.is_empty or float(centerline.length) <= 0:
        raise ValueError("centerline must be a non-empty LineString with positive length.")
    return centerline


def reverse_linestring(centerline):
    ls = _as_linestring(centerline)
    return LineString(list(ls.coords)[::-1])


def summarize_centerline_geometries(
    centerlines,
    id_col: str = "main_path_id",
    geometry_col: str = "line",
    assert_all_linestring: bool = False,
) -> pd.DataFrame:
    """
    Summarize centerline geometry types and optionally fail on non-LineStrings.
    """
    rows = []
    if isinstance(centerlines, dict):
        iterator = centerlines.items()
    else:
        if id_col not in centerlines.columns:
            raise ValueError(f"centerlines is missing id column '{id_col}'.")
        if geometry_col not in centerlines.columns:
            raise ValueError(f"centerlines is missing geometry column '{geometry_col}'.")
        iterator = (
            (row[id_col], row[geometry_col])
            for _, row in centerlines.iterrows()
        )

    for key, geom in iterator:
        geom_type = None if geom is None else getattr(geom, "geom_type", type(geom).__name__)
        rows.append(
            {
                id_col: key,
                "geometry_type": geom_type,
                "is_linestring": bool(LineString is not None and isinstance(geom, LineString)),
                "is_multilinestring": bool(MultiLineString is not None and isinstance(geom, MultiLineString)),
                "is_empty": bool(geom is None or getattr(geom, "is_empty", False)),
                "length_m": float(getattr(geom, "length", np.nan)) if geom is not None else np.nan,
            }
        )

    summary = pd.DataFrame(rows)
    if assert_all_linestring and not summary.empty:
        bad = summary[~summary["is_linestring"] | summary["is_empty"]].copy()
        if not bad.empty:
            sample = bad[id_col].head(10).tolist()
            raise ValueError(
                f"Expected all centerlines to be non-empty LineStrings; "
                f"found {len(bad)} invalid geometries. Sample {id_col}: {sample}"
            )
    return summary


def _node_point_from_row(row, geometry_col: str = "geometry"):
    if Point is None:
        raise ImportError("shapely is required for geometry QA.")
    if geometry_col in row.index:
        geom = row[geometry_col]
        if geom is not None and not getattr(geom, "is_empty", True) and hasattr(geom, "x") and hasattr(geom, "y"):
            return geom
    if {"x", "y"}.issubset(row.index):
        x = row["x"]
        y = row["y"]
        if np.isfinite(x) and np.isfinite(y):
            return Point(float(x), float(y))
    return None


def orient_centerline_to_node_dist(
    centerline,
    nodes_df: pd.DataFrame,
    dist_col: str = "dist_m",
    node_geometry_col: str = "geometry",
    reverse_if_needed: bool = True,
    tolerance_m: float = 1e-6,
):
    """
    Orient a LineString so distance 0 is nearest the lowest node distance.

    Returns:
      oriented_centerline, qa_dict
    """
    ls = _as_linestring(centerline)
    if dist_col not in nodes_df.columns:
        raise ValueError(f"nodes_df is missing distance column '{dist_col}'.")

    nodes = nodes_df.sort_values(dist_col).reset_index(drop=True)
    qa = {
        "orientation_checked": False,
        "orientation_status": "not_checked",
        "reversed": False,
        "forward_endpoint_error_m": np.nan,
        "reversed_endpoint_error_m": np.nan,
        "orientation_margin_m": np.nan,
        "first_node_dist_m": float(nodes[dist_col].iloc[0]) if len(nodes) else np.nan,
        "last_node_dist_m": float(nodes[dist_col].iloc[-1]) if len(nodes) else np.nan,
    }

    if nodes.empty:
        qa["orientation_status"] = "no_nodes"
        return ls, qa

    first_node = _node_point_from_row(nodes.iloc[0], geometry_col=node_geometry_col)
    last_node = _node_point_from_row(nodes.iloc[-1], geometry_col=node_geometry_col)
    if first_node is None or last_node is None:
        qa["orientation_status"] = "missing_node_geometry"
        return ls, qa

    line_start = Point(ls.coords[0])
    line_end = Point(ls.coords[-1])
    forward_error = float(line_start.distance(first_node) + line_end.distance(last_node))
    reversed_error = float(line_start.distance(last_node) + line_end.distance(first_node))
    margin = forward_error - reversed_error

    qa.update(
        {
            "orientation_checked": True,
            "forward_endpoint_error_m": forward_error,
            "reversed_endpoint_error_m": reversed_error,
            "orientation_margin_m": float(margin),
        }
    )

    if reversed_error + float(tolerance_m) < forward_error:
        qa["orientation_status"] = "reversed_to_match_node_dist"
        qa["reversed"] = bool(reverse_if_needed)
        return (reverse_linestring(ls) if reverse_if_needed else ls), qa

    if abs(margin) <= float(tolerance_m):
        qa["orientation_status"] = "ambiguous"
    else:
        qa["orientation_status"] = "matches_node_dist"
    return ls, qa


def summarize_geometry_feature_nan_rates(
    features_by_window: Dict[str, pd.DataFrame],
    feature_cols: Sequence[str] = ("sinu", "curv_int"),
) -> pd.DataFrame:
    rows = []
    for window_label, fdf in features_by_window.items():
        for col in feature_cols:
            if col not in fdf.columns:
                continue
            values = fdf[col].to_numpy(dtype=float)
            rows.append(
                {
                    "window_label": window_label,
                    "feature": col,
                    "n": int(len(values)),
                    "n_finite": int(np.isfinite(values).sum()),
                    "n_missing": int((~np.isfinite(values)).sum()),
                    "missing_frac": float(np.mean(~np.isfinite(values))) if len(values) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _fill_nearest(values: np.ndarray) -> np.ndarray:
    s = pd.Series(values, dtype=float)
    return s.ffill().bfill().to_numpy(dtype=float)


def _gaussian_kernel_1d(sigma_pts: float) -> np.ndarray:
    if sigma_pts <= 0:
        return np.array([1.0])
    radius = int(np.ceil(3.0 * sigma_pts))
    x = np.arange(-radius, radius + 1, dtype=float)
    k = np.exp(-(x * x) / (2.0 * sigma_pts * sigma_pts))
    k /= k.sum()
    return k


def _smooth_xy_gaussian(xy: np.ndarray, sigma_m: float, step_m: float) -> np.ndarray:
    if sigma_m <= 0 or len(xy) < 3:
        return xy.copy()

    sigma_pts = float(sigma_m) / max(float(step_m), 1e-9)
    if gaussian_filter1d is not None:
        xs = gaussian_filter1d(xy[:, 0], sigma_pts, mode="nearest")
        ys = gaussian_filter1d(xy[:, 1], sigma_pts, mode="nearest")
    else:
        k = _gaussian_kernel_1d(sigma_pts)
        pad = len(k) // 2
        x = np.pad(xy[:, 0], (pad, pad), mode="edge")
        y = np.pad(xy[:, 1], (pad, pad), mode="edge")
        xs = np.convolve(x, k, mode="valid")
        ys = np.convolve(y, k, mode="valid")

    out = np.column_stack([xs, ys])
    out[0] = xy[0]
    out[-1] = xy[-1]
    return out


def sample_centerline_equal_spacing(centerline, step_m: float = 10.0) -> pd.DataFrame:
    """
    Sample a LineString at equal chainage spacing.

    Returns a DataFrame with dist_m, x, y.
    """
    ls = _as_linestring(centerline)
    step_m = float(step_m)
    if step_m <= 0:
        raise ValueError("step_m must be positive.")

    length_m = float(ls.length)
    dists = np.arange(0.0, length_m + 0.5 * step_m, step_m)
    if dists[-1] < length_m:
        dists = np.append(dists, length_m)
    dists[-1] = length_m

    xs = np.empty(len(dists), dtype=float)
    ys = np.empty(len(dists), dtype=float)
    for i, d in enumerate(dists):
        p = ls.interpolate(float(d))
        xs[i] = p.x
        ys[i] = p.y

    return pd.DataFrame({"dist_m": dists, "x": xs, "y": ys})


def _node_width_arrays(
    nodes_df: pd.DataFrame,
    cfg: GeometryFeatureConfig,
    ) -> Tuple[np.ndarray, np.ndarray]:
    _require_cols(nodes_df, [cfg.dist_col, cfg.width_col], name="nodes_df")
    d = nodes_df[[cfg.dist_col, cfg.width_col]].copy()
    d = d.replace([np.inf, -np.inf], np.nan).dropna()
    d = d[d[cfg.width_col] > 0].sort_values(cfg.dist_col)
    if d.empty:
        raise ValueError(f"No finite positive widths found in '{cfg.width_col}'.")
    return (
        d[cfg.dist_col].to_numpy(dtype=float),
        d[cfg.width_col].to_numpy(dtype=float),
    )


def interpolate_node_width_to_geom(
    geom_dist_m: np.ndarray,
    nodes_df: pd.DataFrame,
    cfg: GeometryFeatureConfig,
    ) -> np.ndarray:
    node_dist, node_width = _node_width_arrays(nodes_df, cfg)
    if len(node_dist) == 1:
        return np.full(len(geom_dist_m), float(node_width[0]), dtype=float)
    return np.interp(
        np.asarray(geom_dist_m, dtype=float),
        node_dist,
        node_width,
        left=float(node_width[0]),
        right=float(node_width[-1]),
    )


def _group_width_segments(
    nodes_df: pd.DataFrame,
    line_length_m: float,
    cfg: GeometryFeatureConfig,
    ) -> pd.DataFrame:
    node_dist, node_width = _node_width_arrays(nodes_df, cfg)
    order = np.argsort(node_dist)
    node_dist = node_dist[order]
    node_width = node_width[order]

    groups = []
    start_i = 0
    running = float(node_width[0])
    count = 1
    rel_tol = float(cfg.width_group_rel_tol)

    for i in range(1, len(node_width)):
        rel_dev = abs(float(node_width[i]) - running) / max(abs(running), 1e-9)
        if rel_dev <= rel_tol:
            count += 1
            running += (float(node_width[i]) - running) / count
            continue

        groups.append((start_i, i - 1))
        start_i = i
        running = float(node_width[i])
        count = 1
    groups.append((start_i, len(node_width) - 1))

    rows = []
    for group_id, (i0, i1) in enumerate(groups):
        start_m = 0.0 if group_id == 0 else 0.5 * (node_dist[i0 - 1] + node_dist[i0])
        end_m = (
            float(line_length_m)
            if group_id == len(groups) - 1
            else 0.5 * (node_dist[i1] + node_dist[i1 + 1])
        )
        widths = node_width[i0 : i1 + 1]
        if cfg.width_repr == "mean":
            width_repr = float(np.mean(widths))
        elif cfg.width_repr == "median":
            width_repr = float(np.median(widths))
        else:
            raise ValueError("GeometryFeatureConfig.width_repr must be 'median' or 'mean'.")

        sigma_m = float(cfg.sigma_width_frac) * width_repr
        sigma_m = max(float(cfg.sigma_min_m), sigma_m)
        sigma_m = min(float(cfg.sigma_max_m), sigma_m)

        rows.append(
            {
                "segment_id": int(group_id),
                "start_m": float(max(0.0, start_m)),
                "end_m": float(min(float(line_length_m), end_m)),
                "width_m": width_repr,
                "sigma_m": sigma_m,
            }
        )

    return pd.DataFrame(rows)


def smooth_geom_xy(
    geom_df: pd.DataFrame,
    nodes_df: pd.DataFrame,
    cfg: GeometryFeatureConfig = GeometryFeatureConfig(),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Smooth sampled centerline coordinates for curvature calculation.

    Returns:
      smoothed_geom_df, smoothing_segments_df
    """
    _require_cols(geom_df, ["dist_m", "x", "y"], name="geom_df")
    geom = geom_df.sort_values("dist_m").reset_index(drop=True).copy()
    xy = geom[["x", "y"]].to_numpy(dtype=float)
    step_m = float(cfg.ds_geom_m)
    method = str(cfg.smoothing_method)

    if method == "none":
        sm_xy = xy.copy()
        segments = pd.DataFrame(
            [{"segment_id": 0, "start_m": float(geom["dist_m"].min()), "end_m": float(geom["dist_m"].max()), "sigma_m": 0.0}]
        )
    elif method == "fixed_gaussian":
        sm_xy = _smooth_xy_gaussian(xy, sigma_m=float(cfg.fixed_sigma_m), step_m=step_m)
        segments = pd.DataFrame(
            [{
                "segment_id": 0,
                "start_m": float(geom["dist_m"].min()),
                "end_m": float(geom["dist_m"].max()),
                "sigma_m": float(cfg.fixed_sigma_m),
            }]
        )
    elif method == "piecewise_width_gaussian":
        line_length_m = float(geom["dist_m"].max())
        segments = _group_width_segments(nodes_df, line_length_m=line_length_m, cfg=cfg)
        blend_m = float(cfg.blend_m) if cfg.blend_m is not None else 5.0 * step_m

        acc = np.zeros_like(xy)
        w_sum = np.zeros(len(xy), dtype=float)
        dist = geom["dist_m"].to_numpy(dtype=float)

        for _, seg in segments.iterrows():
            start = float(seg["start_m"])
            end = float(seg["end_m"])
            sigma_m = float(seg["sigma_m"])
            if end <= start:
                continue
            blend = min(blend_m, 0.5 * (end - start)) if blend_m > 0 else 0.0
            ext_start = max(float(dist[0]), start - blend)
            ext_end = min(float(dist[-1]), end + blend)
            idx = np.where((dist >= ext_start) & (dist <= ext_end))[0]
            if len(idx) < 3:
                continue

            xy_seg = xy[idx]
            xy_sm = _smooth_xy_gaussian(xy_seg, sigma_m=sigma_m, step_m=step_m)

            d = dist[idx]
            weights = np.ones_like(d, dtype=float)
            if blend > 0:
                left = d < start
                right = d > end
                weights[left] = (d[left] - (start - blend)) / blend
                weights[right] = ((end + blend) - d[right]) / blend
                weights = np.clip(weights, 0.0, 1.0)

            acc[idx] += xy_sm * weights[:, None]
            w_sum[idx] += weights

        sm_xy = xy.copy()
        valid = w_sum > 0
        sm_xy[valid] = acc[valid] / w_sum[valid, None]
        sm_xy[0] = xy[0]
        sm_xy[-1] = xy[-1]
    else:
        raise ValueError(
            "GeometryFeatureConfig.smoothing_method must be one of "
            "'none', 'fixed_gaussian', or 'piecewise_width_gaussian'."
        )

    out = geom.copy()
    out["x_smooth"] = sm_xy[:, 0]
    out["y_smooth"] = sm_xy[:, 1]
    return out, segments.reset_index(drop=True)


def signed_curvature_from_smoothed_xy(
    x: np.ndarray,
    y: np.ndarray,
    ds_m: float = 10.0,
    eps: float = 1e-12,
    ) -> np.ndarray:
    """
    Compute signed curvature using finite differences on uniformly spaced coordinates.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 5:
        return np.full(n, np.nan, dtype=float)

    ds_m = float(ds_m)
    dx = np.empty(n, dtype=float)
    dy = np.empty(n, dtype=float)
    ddx = np.empty(n, dtype=float)
    ddy = np.empty(n, dtype=float)

    dx[1:-1] = (x[2:] - x[:-2]) / (2.0 * ds_m)
    dy[1:-1] = (y[2:] - y[:-2]) / (2.0 * ds_m)
    dx[0] = (x[1] - x[0]) / ds_m
    dy[0] = (y[1] - y[0]) / ds_m
    dx[-1] = (x[-1] - x[-2]) / ds_m
    dy[-1] = (y[-1] - y[-2]) / ds_m

    ddx[1:-1] = (x[2:] - 2.0 * x[1:-1] + x[:-2]) / (ds_m * ds_m)
    ddy[1:-1] = (y[2:] - 2.0 * y[1:-1] + y[:-2]) / (ds_m * ds_m)
    ddx[0] = ddx[1]
    ddy[0] = ddy[1]
    ddx[-1] = ddx[-2]
    ddy[-1] = ddy[-2]

    denom = (dx * dx + dy * dy) ** 1.5
    denom = np.maximum(denom, float(eps))
    return (dx * ddy - dy * ddx) / denom


def build_geometry_base(
    centerline,
    nodes_df: pd.DataFrame,
    cfg: GeometryFeatureConfig = GeometryFeatureConfig(),
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build 10 m original/smoothed geometry and signed curvature once per reach.
    """
    geom = sample_centerline_equal_spacing(centerline, step_m=cfg.ds_geom_m)
    geom, smoothing_segments = smooth_geom_xy(geom, nodes_df=nodes_df, cfg=cfg)
    geom["kappa"] = signed_curvature_from_smoothed_xy(
        geom["x_smooth"].to_numpy(dtype=float),
        geom["y_smooth"].to_numpy(dtype=float),
        ds_m=cfg.ds_geom_m,
    )
    geom["width_interp_m"] = interpolate_node_width_to_geom(
        geom["dist_m"].to_numpy(dtype=float),
        nodes_df=nodes_df,
        cfg=cfg,
    )
    return geom, smoothing_segments


def _window_bounds(center_m: float, window_m: float, max_dist_m: float) -> Tuple[float, float]:
    half = 0.5 * float(window_m)
    lo = max(0.0, float(center_m) - half)
    hi = min(float(max_dist_m), float(center_m) + half)
    return lo, hi


def _polyline_sinuosity_in_window(
    geom_df: pd.DataFrame,
    lo_m: float,
    hi_m: float,
    min_samples: int,
    ) -> float:
    sub = geom_df[(geom_df["dist_m"] >= lo_m) & (geom_df["dist_m"] <= hi_m)].copy()
    if len(sub) < int(min_samples):
        return np.nan

    x = sub["x"].to_numpy(dtype=float)
    y = sub["y"].to_numpy(dtype=float)
    dx = np.diff(x)
    dy = np.diff(y)
    along = float(np.sum(np.sqrt(dx * dx + dy * dy)))
    direct = float(np.sqrt((x[-1] - x[0]) ** 2 + (y[-1] - y[0]) ** 2))
    if along <= 0.0 or direct <= 0.0:
        return np.nan
    return along / direct


def _median_width_in_window(
    nodes_df: pd.DataFrame,
    center_m: float,
    lo_m: float,
    hi_m: float,
    cfg: GeometryFeatureConfig,
    ) -> float:
    dist = nodes_df[cfg.dist_col].to_numpy(dtype=float)
    width = nodes_df[cfg.width_col].to_numpy(dtype=float)
    m = (dist >= lo_m) & (dist <= hi_m) & np.isfinite(width) & (width > 0)
    if np.any(m):
        return float(np.median(width[m]))

    node_dist, node_width = _node_width_arrays(nodes_df, cfg)
    if len(node_dist) == 1:
        return float(node_width[0])
    return float(np.interp(float(center_m), node_dist, node_width, left=node_width[0], right=node_width[-1]))


def compute_geometry_features_for_window(
    centerline,
    nodes_df: pd.DataFrame,
    window_m: float,
    cfg: GeometryFeatureConfig = GeometryFeatureConfig(),
    geom_base: Optional[pd.DataFrame] = None,
    return_diagnostics: bool = False,
    ) -> pd.DataFrame | Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Compute node-level sinuosity and dimensionless curvature for one window.

    curv_int = RMS(kappa_10m inside active window) * median(width inside same window)
    sinu = centerline_length / direct_endpoint_distance inside active window
    """
    _require_cols(nodes_df, [cfg.dist_col, cfg.width_col], name="nodes_df")
    nodes = nodes_df.sort_values(cfg.dist_col).reset_index(drop=True).copy()

    if geom_base is None:
        geom, smoothing_segments = build_geometry_base(centerline, nodes, cfg=cfg)
    else:
        geom = geom_base.copy()
        smoothing_segments = pd.DataFrame()

    max_dist_m = float(geom["dist_m"].max())
    node_dist = nodes[cfg.dist_col].to_numpy(dtype=float)
    window_m = float(window_m)
    sinu_window_m = max(window_m, float(cfg.min_sinuosity_window_m))
    curv_window_m = max(window_m, float(cfg.min_curvature_window_m))

    sinu = np.full(len(nodes), np.nan, dtype=float)
    kappa_rms = np.full(len(nodes), np.nan, dtype=float)
    width_med = np.full(len(nodes), np.nan, dtype=float)
    curv_int = np.full(len(nodes), np.nan, dtype=float)

    geom_dist = geom["dist_m"].to_numpy(dtype=float)
    kappa = geom["kappa"].to_numpy(dtype=float)
    min_samples = int(cfg.min_geom_samples)

    for i, d in enumerate(node_dist):
        lo_s, hi_s = _window_bounds(d, sinu_window_m, max_dist_m)
        sinu[i] = _polyline_sinuosity_in_window(geom, lo_s, hi_s, min_samples=min_samples)

        lo_c, hi_c = _window_bounds(d, curv_window_m, max_dist_m)
        m = (geom_dist >= lo_c) & (geom_dist <= hi_c) & np.isfinite(kappa)
        if int(np.sum(m)) >= min_samples:
            vals = kappa[m]
            kappa_rms[i] = float(np.sqrt(np.mean(vals * vals)))
            width_med[i] = _median_width_in_window(nodes, d, lo_c, hi_c, cfg)
            curv_int[i] = kappa_rms[i] * width_med[i]

    if cfg.fill_edges:
        sinu = _fill_nearest(sinu)
        kappa_rms = _fill_nearest(kappa_rms)
        width_med = _fill_nearest(width_med)
        curv_int = _fill_nearest(curv_int)

    out = pd.DataFrame(
        {
            cfg.dist_col: node_dist,
            "window_m": window_m,
            "sinu": sinu,
            "curv_int": curv_int,
            "kappa_rms": kappa_rms,
            "width_med_window_m": width_med,
            "sinu_window_m_effective": float(sinu_window_m),
            "curv_window_m_effective": float(curv_window_m),
        }
    )

    if not return_diagnostics:
        return out

    diagnostics = {
        "geom_base": geom,
        "smoothing_segments": smoothing_segments,
        "config": cfg,
    }
    return out, diagnostics


def compute_geometry_features_for_windows(
    centerline,
    nodes_df: pd.DataFrame,
    windows_m: Iterable[float],
    cfg: GeometryFeatureConfig = GeometryFeatureConfig(),
    return_diagnostics: bool = False,
    ) -> Dict[str, pd.DataFrame] | Tuple[Dict[str, pd.DataFrame], Dict[str, object]]:
    """
    Compute geometry features for multiple PELT windows using one shared curvature base.
    """
    nodes = nodes_df.sort_values(cfg.dist_col).reset_index(drop=True).copy()
    geom, smoothing_segments = build_geometry_base(centerline, nodes, cfg=cfg)

    out: Dict[str, pd.DataFrame] = {}
    for window_m in windows_m:
        key = f"W{float(window_m) / 1000.0:g}km"
        out[key] = compute_geometry_features_for_window(
            centerline=centerline,
            nodes_df=nodes,
            window_m=float(window_m),
            cfg=cfg,
            geom_base=geom,
            return_diagnostics=False,
        )

    if not return_diagnostics:
        return out

    diagnostics = {
        "geom_base": geom,
        "smoothing_segments": smoothing_segments,
        "config": cfg,
    }
    return out, diagnostics
