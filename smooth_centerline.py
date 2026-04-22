import numpy as np
from shapely.geometry import LineString, Point
import pandas as pd
import time

try:
    from scipy.spatial import cKDTree
    _HAS_CKDTREE = True
except Exception:
    _HAS_CKDTREE = False

try:
    from scipy.ndimage import gaussian_filter1d
    _HAS_NDIMAGE = True
except Exception:
    _HAS_NDIMAGE = False

try:
    from shapely import points as _shp_points
    from shapely import distance as _shp_distance
    _HAS_SHAPELY_V2 = True
except Exception:
    _HAS_SHAPELY_V2 = False

def resample_linestring_equal(ls: LineString, step: float) -> LineString: 
    step = float(step) 
    L = float(ls.length) 
    if L <= 0 or step <= 0: 
        return ls 
    xy = np.asarray(ls.coords, float)
    if len(xy) < 2:
        return ls
    deltas = np.diff(xy, axis=0)
    seglen = np.hypot(deltas[:, 0], deltas[:, 1])
    if not np.any(seglen > 0):
        return ls
    cum = np.concatenate([[0.0], np.cumsum(seglen)])
    # Remove duplicate distances caused by zero-length segments
    unique_cum, unique_idx = np.unique(cum, return_index=True)
    xy_u = xy[unique_idx]
    L = float(unique_cum[-1])
    if L <= 0:
        return ls
    n = int(np.floor(L / step)) 
    dists = np.linspace(0, L, n + 2) # include endpoints 
    xs = np.interp(dists, unique_cum, xy_u[:, 0])
    ys = np.interp(dists, unique_cum, xy_u[:, 1])
    pts = np.column_stack([xs, ys])
    return LineString(pts.tolist())

def gaussian_kernel_1d(sigma_pts: float) -> np.ndarray: 
    if sigma_pts <= 0: 
        return np.array([1.0]) 
    radius = int(np.ceil(3 * sigma_pts)) 
    x = np.arange(-radius, radius + 1) 
    k = np.exp(-(x**2) / (2 * sigma_pts**2)) 
    k /= k.sum() 
    return k

def smooth_linestring_gaussian(ls: LineString, sigma_m: float, step_m: float) -> LineString: 
    if sigma_m <= 0: 
        return ls 
    xy = np.asarray(ls.coords, float) 
    if len(xy) < 3: 
        return ls
    sigma_pts = float(sigma_m) / float(step_m) 
    if _HAS_NDIMAGE:
        xs = gaussian_filter1d(xy[:, 0], sigma_pts, mode="nearest")
        ys = gaussian_filter1d(xy[:, 1], sigma_pts, mode="nearest")
    else:
        k = gaussian_kernel_1d(sigma_pts) 
        pad = len(k) // 2 
        x = np.pad(xy[:, 0], (pad, pad), mode="edge") 
        y = np.pad(xy[:, 1], (pad, pad), mode="edge") 
        xs = np.convolve(x, k, mode="valid") 
        ys = np.convolve(y, k, mode="valid") 
    out = np.column_stack([xs, ys]) 
    out[0] = xy[0] 
    out[-1] = xy[-1] 
    return LineString(out.tolist())

def turning_energy(ls: LineString) -> float: 
    """Curvature proxy: sum(|turn angles|)/length.""" 
    xy = np.asarray(ls.coords, float) 
    if len(xy) < 3 or ls.length <= 0: 
        return 0.0 
    v1 = xy[1:-1] - xy[:-2] 
    v2 = xy[2:] - xy[1:-1] 
    n1 = np.linalg.norm(v1, axis=1) 
    n2 = np.linalg.norm(v2, axis=1) 
    m = (n1 > 1e-12) & (n2 > 1e-12) 
    v1 = v1[m] / n1[m, None] 
    v2 = v2[m] / n2[m, None] 
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0] 
    dot = np.sum(v1 * v2, axis=1) 
    ang = np.abs(np.arctan2(cross, dot)) 
    return float(np.sum(ang) / ls.length)




def width_profile_from_breakpoints(breaks_m, widths_m):
    """
    Piecewise-constant width profile along chainage.
    breaks_m: e.g. [0, 50_000, 100_000]
    widths_m: e.g. [40, 100] (applies to [0-50km), [50-100km])
    """
    breaks_m = np.asarray(breaks_m, float)
    widths_m = np.asarray(widths_m, float)
    assert len(breaks_m) == len(widths_m) + 1
    assert np.all(np.diff(breaks_m) > 0)

    def w(d):
        d = np.asarray(d, float)
        idx = np.searchsorted(breaks_m[1:], d, side="right")
        idx = np.clip(idx, 0, len(widths_m) - 1)
        return widths_m[idx]
    return w

def distances_with_budget(original: LineString, smoothed: LineString, *, n=600, budget_fn=None):
    """
    Sample distances from points on original to smoothed, and compare to a (possibly varying) budget.
    Returns:
      dists: (n,) point-to-line distances
      budgets: (n,) allowed distances
    """
    L = float(original.length)
    ds = np.linspace(0.0, L, int(n))
    pts = [original.interpolate(d) for d in ds]
    dists = np.array([smoothed.distance(p) for p in pts], float)

    if budget_fn is None:
        budgets = np.full_like(dists, np.inf, dtype=float)
    else:
        budgets = np.asarray(budget_fn(ds), float)
        budgets = np.maximum(budgets, 1e-9)

    return dists, budgets

def simplest_still_similar_turning(
    ls: LineString,
    *,
    # sampling
    step_m=10.0,
    step_m_search=None,
    step_m_final=None,

    # similarity budget control
    width_fn=None,                # function width(chainage_m) -> meters
    budget_abs_m=None,            # if set, overrides width_fn scaling
    budget_width_frac=0.30,       # budget = frac * width(chainage)
    budget_stat="p95",            # "max" or "p95" (or "mean")

    # smoothing search
    sigma_min=None,
    sigma_max=None,
    iters=10,

    # optional "must actually simplify"
    min_turn_drop_frac=0.15,      # require >=15% turning reduction vs original

    # sampling for budget evaluation
    n_samples=600,
    n_samples_search=None,
    n_samples_final=None,

    # distance computation
    distance_mode="kdtree",       # "kdtree", "vectorized", "loop"
    distance_simplify=True,
    distance_simplify_tol=None,
    print_timing=False,
    ):
    """
    Find the most smoothed (largest sigma) line such that:
      - deviation stays within (possibly varying) budget
      - turning energy is reduced by at least min_turn_drop_frac (optional)
    """

    timing = {}
    counts = {}
    t_total = time.perf_counter()

    def _tick(key, dt):
        timing[key] = timing.get(key, 0.0) + float(dt)

    def _count(key, inc=1):
        counts[key] = counts.get(key, 0) + int(inc)

    def _report():
        if not print_timing:
            return
        _tick("total", time.perf_counter() - t_total)
        lines = [
            "timing_summary:",
            f"  step_m_search={step_m_search} step_m_final={step_m_final}",
            f"  n_samples_search={n_samples_search} n_samples_final={n_samples_final}",
            f"  distance_mode_search={distance_mode_search} distance_mode_final={distance_mode_final}",
        ]
        for k in sorted(counts.keys()):
            lines.append(f"  {k}={counts[k]}")
        for k in sorted(timing.keys()):
            lines.append(f"  {k}={timing[k]:.4f}s")
        print("\n".join(lines))

    def _done(out):
        _report()
        return out

    if step_m_final is None:
        step_m_final = step_m
    if step_m_search is None:
        step_m_search = step_m_final
    step_m_final = float(step_m_final)
    step_m_search = float(step_m_search)

    t_rs = time.perf_counter()
    ls_eq_final = resample_linestring_equal(ls, step=step_m_final)
    _tick("resample_final", time.perf_counter() - t_rs)

    if step_m_search == step_m_final:
        ls_eq_search = ls_eq_final
    else:
        t_rs = time.perf_counter()
        ls_eq_search = resample_linestring_equal(ls, step=step_m_search)
        _tick("resample_search", time.perf_counter() - t_rs)

    L_final = float(ls_eq_final.length)
    L_search = float(ls_eq_search.length)

    if sigma_min is None:
        sigma_min = 2.0 * float(step_m_final)
    if sigma_max is None:
        sigma_max = max(L_final / 2.0, 2000.0)

    # Build budget function
    if budget_abs_m is not None:
        def budget_fn(ds):
            return np.full_like(ds, float(budget_abs_m), dtype=float)
    else:
        if width_fn is None:
            # fallback: constant budget based on length scale (only if user provides no width info)
            const = L_final / 400.0
            def budget_fn(ds):
                return np.full_like(ds, float(const), dtype=float)
        else:
            def budget_fn(ds):
                return float(budget_width_frac) * np.asarray(width_fn(ds), float)

    t0_final = float(turning_energy(ls_eq_final))
    t0_search = t0_final if ls_eq_search is ls_eq_final else float(turning_energy(ls_eq_search))

    if distance_simplify_tol is None:
        distance_simplify_tol = float(step_m_final)
    if distance_mode == "kdtree":
        # kdtree distance is to vertices; keep line dense for accuracy
        distance_simplify = False

    if n_samples_search is None:
        n_samples_search = min(200, int(n_samples))
    if n_samples_final is None:
        n_samples_final = int(n_samples)
    n_samples_search = int(max(10, n_samples_search))
    n_samples_final = int(max(10, n_samples_final))

    def _prepare_sm_for_distance(sm):
        if not distance_simplify:
            return sm
        tol = float(distance_simplify_tol)
        if tol <= 0:
            return sm
        sm_dist = sm.simplify(tol, preserve_topology=False)
        if not isinstance(sm_dist, LineString) or len(sm_dist.coords) < 2:
            return sm
        return sm_dist

    def _select_distance_mode():
        if distance_mode == "kdtree":
            if _HAS_CKDTREE:
                return "kdtree"
            return "vectorized" if _HAS_SHAPELY_V2 else "loop"
        if distance_mode == "vectorized":
            return "vectorized" if _HAS_SHAPELY_V2 else "loop"
        return "loop"

    def make_evaluator(ls_base, L_base, t0_base, n_samples_eval, label):
        t_prep = time.perf_counter()
        mode = _select_distance_mode()
        ds = np.linspace(0.0, L_base, int(n_samples_eval))
        sample_coords = np.array([ls_base.interpolate(d).coords[0] for d in ds], dtype=float)
        budgets = budget_fn(ds)
        budgets = np.asarray(budgets, float)
        if budgets.ndim == 0:
            budgets = np.full_like(ds, float(budgets), dtype=float)
        budgets = np.maximum(budgets, 1e-9)
        budget_nonfinite_global = not np.all(np.isfinite(budgets))

        if mode == "loop":
            pts = [Point(xy) for xy in sample_coords]
            pts_geom = None
        elif mode == "vectorized":
            pts = None
            pts_geom = _shp_points(sample_coords[:, 0], sample_coords[:, 1])
        else:
            pts = None
            pts_geom = None
        _tick(f"prep_{label}", time.perf_counter() - t_prep)

        def _calc_dists(sm):
            t1 = time.perf_counter()
            sm_dist = _prepare_sm_for_distance(sm)
            if mode == "kdtree":
                coords = np.asarray(sm_dist.coords, float)
                if len(coords) == 0:
                    d = np.full(len(sample_coords), np.inf, dtype=float)
                    _tick(f"dist_{label}_{mode}", time.perf_counter() - t1)
                    return d
                if len(coords) == 1:
                    d = np.linalg.norm(sample_coords - coords[0], axis=1)
                    _tick(f"dist_{label}_{mode}", time.perf_counter() - t1)
                    return d
                tree = cKDTree(coords)
                dists, _ = tree.query(sample_coords, k=1)
                _tick(f"dist_{label}_{mode}", time.perf_counter() - t1)
                return dists
            if mode == "vectorized":
                d = np.asarray(_shp_distance(sm_dist, pts_geom), float)
                _tick(f"dist_{label}_{mode}", time.perf_counter() - t1)
                return d
            d = np.array([sm_dist.distance(p) for p in pts], float)
            _tick(f"dist_{label}_{mode}", time.perf_counter() - t1)
            return d

        def is_feasible(sm):
            _count(f"calls_{label}")
            dists = _calc_dists(sm)

            finite = np.isfinite(dists) & np.isfinite(budgets)
            budget_nonfinite = budget_nonfinite_global or not np.all(finite)
            if not np.any(finite):
                t2 = time.perf_counter()
                tm = float(turning_energy(sm))
                _tick(f"turn_{label}", time.perf_counter() - t2)
                turn_drop = (t0_base - tm) / max(t0_base, 1e-12)
                ok_turn = (turn_drop >= float(min_turn_drop_frac)) if min_turn_drop_frac is not None else True
                return (
                    False,
                    np.nan,
                    np.nan,
                    np.nan,
                    tm,
                    turn_drop,
                    False,
                    ok_turn,
                    budget_nonfinite,
                )

            d = dists[finite]
            b = budgets[finite]

            if budget_stat == "max":
                excess_stat = float(np.max(d - b))
                ok_dist = excess_stat <= 0.0
                dist_stat = float(np.max(d))
                budget_stat_value = float(np.max(b))
            elif budget_stat == "mean":
                excess_stat = float(np.mean(d - b))
                ok_dist = excess_stat <= 0.0
                dist_stat = float(np.mean(d))
                budget_stat_value = float(np.mean(b))
            else:  # default "p95"
                excess_stat = float(np.percentile(d - b, 95))
                ok_dist = excess_stat <= 0.0
                dist_stat = float(np.percentile(d, 95))
                budget_stat_value = float(np.percentile(b, 95))

            if budget_nonfinite:
                ok_dist = False

            t2 = time.perf_counter()
            tm = float(turning_energy(sm))
            _tick(f"turn_{label}", time.perf_counter() - t2)
            turn_drop = (t0_base - tm) / max(t0_base, 1e-12)
            ok_turn = (turn_drop >= float(min_turn_drop_frac)) if min_turn_drop_frac is not None else True

            return (
                ok_dist and ok_turn,
                dist_stat,
                budget_stat_value,
                excess_stat,
                tm,
                turn_drop,
                ok_dist,
                ok_turn,
                budget_nonfinite,
            )

        return is_feasible, mode

    is_feasible_search, distance_mode_search = make_evaluator(
        ls_eq_search, L_search, t0_search, n_samples_search, "search"
    )
    same_eval = (
        n_samples_final == n_samples_search and
        ls_eq_final is ls_eq_search and
        t0_final == t0_search
    )
    if same_eval:
        is_feasible_final = is_feasible_search
        distance_mode_final = distance_mode_search
    else:
        is_feasible_final, distance_mode_final = make_evaluator(
            ls_eq_final, L_final, t0_final, n_samples_final, "final"
        )

    def smooth_at_search(sigma):
        t_sm = time.perf_counter()
        sm = smooth_linestring_gaussian(ls_eq_search, sigma_m=float(sigma), step_m=step_m_search)
        _tick("smooth_search", time.perf_counter() - t_sm)
        _count("smooth_calls_search")
        return sm

    def smooth_at_final(sigma):
        t_sm = time.perf_counter()
        sm = smooth_linestring_gaussian(ls_eq_final, sigma_m=float(sigma), step_m=step_m_final)
        _tick("smooth_final", time.perf_counter() - t_sm)
        _count("smooth_calls_final")
        return sm

    # Check feasibility at sigma_min (search sampling)
    sm_lo = smooth_at_search(sigma_min)
    ok_lo, dist_lo, budget_lo, excess_lo, t_lo, drop_lo, ok_dist_lo, ok_turn_lo, budget_nonfinite_lo = is_feasible_search(sm_lo)
    if budget_nonfinite_lo:
        reasons = ["budget_nonfinite"]
        if not ok_dist_lo:
            reasons.append("distance")
        if not ok_turn_lo:
            reasons.append("turning")
        return _done({
            "line": ls_eq_final,
            "sigma": 0.0,
            "step_m": float(step_m_final),
            "note": "Even minimal smoothing violated constraints; returning resampled original.",
            "fail_reason": "+".join(reasons),
            "turning_original": t0_final,
            "turning_min": t_lo,
            "turn_drop_frac": drop_lo,
            "dist_stat": dist_lo,
            "budget_stat": budget_stat,
            "budget_stat_value": budget_lo,
            "dist_minus_budget_stat": excess_lo,
            "distance_mode": distance_mode_search,
        })

    # If distance fails in search, confirm with final samples before returning.
    if not ok_dist_lo:
        (
            ok_lo_f,
            dist_lo_f,
            budget_lo_f,
            excess_lo_f,
            t_lo_f,
            drop_lo_f,
            ok_dist_lo_f,
            ok_turn_lo_f,
            budget_nonfinite_lo_f,
        ) = is_feasible_final(smooth_at_final(sigma_min))
        if budget_nonfinite_lo_f:
            reasons = ["budget_nonfinite"]
            if not ok_dist_lo_f:
                reasons.append("distance")
            if not ok_turn_lo_f:
                reasons.append("turning")
            return _done({
                "line": ls_eq_final,
                "sigma": 0.0,
                "step_m": float(step_m_final),
                "note": "Even minimal smoothing violated constraints; returning resampled original.",
                "fail_reason": "+".join(reasons),
                "turning_original": t0_final,
                "turning_min": t_lo_f,
                "turn_drop_frac": drop_lo_f,
                "dist_stat": dist_lo_f,
                "budget_stat": budget_stat,
                "budget_stat_value": budget_lo_f,
                "dist_minus_budget_stat": excess_lo_f,
                "distance_mode": distance_mode_final,
            })
        if not ok_dist_lo_f:
            reasons = ["distance"]
            if not ok_turn_lo_f:
                reasons.append("turning")
            return _done({
                "line": ls_eq_final,
                "sigma": 0.0,
                "step_m": float(step_m_final),
                "note": "Even minimal smoothing violated constraints; returning resampled original.",
                "fail_reason": "+".join(reasons),
                "turning_original": t0_final,
                "turning_min": t_lo_f,
                "turn_drop_frac": drop_lo_f,
                "dist_stat": dist_lo_f,
                "budget_stat": budget_stat,
                "budget_stat_value": budget_lo_f,
                "dist_minus_budget_stat": excess_lo_f,
                "distance_mode": distance_mode_final,
            })

        dist_lo = dist_lo_f
        budget_lo = budget_lo_f
        excess_lo = excess_lo_f
        t_lo = t_lo_f
        drop_lo = drop_lo_f
        ok_dist_lo = ok_dist_lo_f
        ok_turn_lo = ok_turn_lo_f
        budget_nonfinite_lo = budget_nonfinite_lo_f

    # Distance-only search for the largest feasible sigma
    sm_hi = smooth_at_search(sigma_max)
    ok_hi, dist_hi, budget_hi, excess_hi, t_hi, drop_hi, ok_dist_hi, ok_turn_hi, budget_nonfinite_hi = is_feasible_search(sm_hi)

    if ok_dist_hi:
        # Expand above initial sigma_max until distance fails, then bracket and search.
        growth = 2.0
        max_growth_iters = 8
        sigma_lo = float(sigma_max)
        sm_lo_up = sm_hi
        dist_lo_up = dist_hi
        budget_lo_up = budget_hi
        excess_lo_up = excess_hi
        t_lo_up = t_hi
        drop_lo_up = drop_hi
        sigma_hi = None

        for _ in range(int(max_growth_iters)):
            sigma_try = sigma_lo * growth
            if sigma_try <= 0:
                break
            sm_try = smooth_at_search(sigma_try)
            ok, dist_stat, budget_stat_value, excess_stat, tm, drop, ok_dist, ok_turn, budget_nonfinite = is_feasible_search(sm_try)
            if ok_dist:
                sigma_lo = sigma_try
                sm_lo_up = sm_try
                dist_lo_up = dist_stat
                budget_lo_up = budget_stat_value
                excess_lo_up = excess_stat
                t_lo_up = tm
                drop_lo_up = drop
            else:
                sigma_hi = sigma_try
                break

        if sigma_hi is None:
            sm_best, sigma_best, dist_best, budget_best, excess_best, t_best, drop_best = (
                sm_lo_up, float(sigma_lo), dist_lo_up, budget_lo_up, excess_lo_up, t_lo_up, drop_lo_up
            )
        else:
            lo, hi = float(sigma_lo), float(sigma_hi)
            best = (sm_lo_up, lo, dist_lo_up, budget_lo_up, excess_lo_up, t_lo_up, drop_lo_up)
            for _ in range(int(iters)):
                mid = np.sqrt(lo * hi)
                sm = smooth_at_search(mid)
                ok, dist_stat, budget_stat_value, excess_stat, tm, drop, ok_dist, ok_turn, budget_nonfinite = is_feasible_search(sm)
                if ok_dist:
                    best = (sm, mid, dist_stat, budget_stat_value, excess_stat, tm, drop)
                    lo = mid
                else:
                    hi = mid
            sm_best, sigma_best, dist_best, budget_best, excess_best, t_best, drop_best = best
    else:
        lo, hi = float(sigma_min), float(sigma_max)
        best = (sm_lo, lo, dist_lo, budget_lo, excess_lo, t_lo, drop_lo)
        for _ in range(int(iters)):
            mid = np.sqrt(lo * hi)
            sm = smooth_at_search(mid)
            ok, dist_stat, budget_stat_value, excess_stat, tm, drop, ok_dist, ok_turn, budget_nonfinite = is_feasible_search(sm)
            if ok_dist:
                best = (sm, mid, dist_stat, budget_stat_value, excess_stat, tm, drop)
                lo = mid
            else:
                hi = mid
        sm_best, sigma_best, dist_best, budget_best, excess_best, t_best, drop_best = best

    # Validate the candidate with final samples (coarse-to-fine)
    sm_best_final = smooth_at_final(sigma_best)
    (
        ok_best_f,
        dist_best_f,
        budget_best_f,
        excess_best_f,
        t_best_f,
        drop_best_f,
        ok_dist_best_f,
        ok_turn_best_f,
        budget_nonfinite_best_f,
    ) = is_feasible_final(sm_best_final)

    if budget_nonfinite_best_f or not ok_dist_best_f:
        # Refine with final samples within [sigma_min, sigma_best]
        lo, hi = float(sigma_min), float(sigma_best)
        sm_lo_final = smooth_at_final(sigma_min)
        (
            ok_lo_f,
            dist_lo_f,
            budget_lo_f,
            excess_lo_f,
            t_lo_f,
            drop_lo_f,
            ok_dist_lo_f,
            ok_turn_lo_f,
            budget_nonfinite_lo_f,
        ) = is_feasible_final(sm_lo_final)
        best_final = (sm_lo_final, lo, dist_lo_f, budget_lo_f, excess_lo_f, t_lo_f, drop_lo_f)
        iters_final = min(6, int(iters))
        for _ in range(int(iters_final)):
            mid = np.sqrt(lo * hi)
            sm = smooth_at_final(mid)
            ok, dist_stat, budget_stat_value, excess_stat, tm, drop, ok_dist, ok_turn, budget_nonfinite = is_feasible_final(sm)
            if ok_dist:
                best_final = (sm, mid, dist_stat, budget_stat_value, excess_stat, tm, drop)
                lo = mid
            else:
                hi = mid
        sm_best, sigma_best, dist_best, budget_best, excess_best, t_best, drop_best = best_final
        (
            ok_best_f,
            dist_best_f,
            budget_best_f,
            excess_best_f,
            t_best_f,
            drop_best_f,
            ok_dist_best_f,
            ok_turn_best_f,
            budget_nonfinite_best_f,
        ) = is_feasible_final(sm_best)
        if budget_nonfinite_best_f or not ok_dist_best_f:
            reasons = ["distance"]
            if budget_nonfinite_best_f:
                reasons.append("budget_nonfinite")
            return _done({
                "line": ls_eq_final,
                "sigma": 0.0,
                "step_m": float(step_m_final),
                "note": "Even minimal smoothing violated constraints; returning resampled original.",
                "fail_reason": "+".join(reasons),
                "turning_original": t0_final,
                "turning_min": t_best_f,
                "turn_drop_frac": drop_best_f,
                "dist_stat": dist_best_f,
                "budget_stat": budget_stat,
                "budget_stat_value": budget_best_f,
                "dist_minus_budget_stat": excess_best_f,
                "distance_mode": distance_mode_final,
            })
    else:
        sm_best = sm_best_final

    # Apply turning requirement at the distance-max sigma.
    if min_turn_drop_frac is not None and not ok_turn_best_f:
        return _done({
            "line": ls_eq_final,
            "sigma": 0.0,
            "step_m": float(step_m_final),
            "note": "Distance budget satisfied but turning constraint not met; returning resampled original.",
            "fail_reason": "turning",
            "turning_original": t0_final,
            "turning_min": t_best_f,
            "turn_drop_frac": drop_best_f,
            "dist_stat": dist_best_f,
            "budget_stat": budget_stat,
            "budget_stat_value": budget_best_f,
            "dist_minus_budget_stat": excess_best_f,
            "sigma_dist_max": float(sigma_best),
            "distance_mode": distance_mode_final,
            "n_samples_search": int(n_samples_search),
            "n_samples_final": int(n_samples_final),
        })

    out = {
        "line": sm_best,
        "sigma": float(sigma_best),
        "step_m": float(step_m_final),
        "note": "Largest sigma satisfying distance + turning constraints.",
        "turning_original": t0_final,
        "turning_final": t_best_f,
        "turn_drop_frac": drop_best_f,
        "dist_stat": dist_best_f,
        "budget_stat": budget_stat,
        "budget_stat_value": budget_best_f,
        "dist_minus_budget_stat": excess_best_f,
        "distance_mode": distance_mode_final,
        "n_samples_search": int(n_samples_search),
        "n_samples_final": int(n_samples_final),
    }
    return _done(out)

def width_breaks_relative(
    df: pd.DataFrame,
    width_col="width",
    dist_col="dist_out",
    len_col="reach_len",
    rel_tol=0.2,          # 20% deviation allowed
    width_repr="mean",    # "mean", "median", "first"
):
    # 1) sort upstream -> downstream
    d = df.sort_values(dist_col, ascending=False).copy()

    # 2) cumulative length
    d["total_len"] = d[len_col].cumsum()

    # 3) rolling relative-mean grouping
    w = d[width_col].to_numpy(dtype=float)
    groups = np.zeros(len(d), dtype=int)

    g = 0
    mean = w[0]
    cnt = 1
    groups[0] = g

    for i in range(1, len(w)):
        # relative deviation condition
        if mean != 0 and abs(w[i] - mean) / abs(mean) <= rel_tol:
            groups[i] = g
            cnt += 1
            mean = mean + (w[i] - mean) / cnt
        else:
            g += 1
            groups[i] = g
            mean = w[i]
            cnt = 1

    d["group"] = groups

    # 4) break = max total_len per group
    agg = {"total_len": "max"}
    if width_repr == "mean":
        agg[width_col] = "mean"
    elif width_repr == "median":
        agg[width_col] = "median"
    elif width_repr == "first":
        agg[width_col] = "first"
    else:
        raise ValueError("width_repr must be one of: 'mean', 'median', 'first'")

    out = (
        d.groupby("group", as_index=False)
         .agg(agg)
         .rename(columns={"total_len": "breaks", width_col: "width"})
         [["width", "breaks"]]
    )

    return out, d


def smooth_mainpath(
    l,
    df,
    mpi,
    print_output=False,
    *,
    step_m=10.0,
    step_m_search=None,
    step_m_final=None,
    budget_width_frac=0.30,
    budget_stat="p95",
    min_turn_drop_frac=0.15,
    sigma_min=None,
    sigma_max=None,
    iters=10,
    n_samples=600,
    n_samples_search=None,
    n_samples_final=None,
    budget_abs_m=None,
    distance_mode="kdtree",
    distance_simplify=True,
    distance_simplify_tol=None,
    print_timing=False,
):
    t_mp = time.perf_counter()
    df_mpi = df[(df['main_path_id'] == mpi) & (df['is_mainstem_edge'] == True)]
    df_mpi = df_mpi[df_mpi['width'] > 30]

    breaks_df, df_with_groups = width_breaks_relative(df_mpi)
        # widths apply piecewise along chainage (meters)
    breaks_m = breaks_df['breaks'].to_list()
    breaks_m.insert(0,0)
    print('Breaks M')
    print(breaks_m)
    print(breaks_df['width'])

    wfn = width_profile_from_breakpoints(
        breaks_m=breaks_m,
        widths_m=breaks_df['width'].to_list()
    )
    if print_timing:
        print(f"mainpath_prep={time.perf_counter() - t_mp:.4f}s")

    out = simplest_still_similar_turning(
        l,
        step_m=step_m,
        step_m_search=step_m_search,
        step_m_final=step_m_final,
        width_fn=wfn,
        budget_width_frac=budget_width_frac,   # allowed deviation = frac * width(chainage)
        budget_stat=budget_stat,               # tolerate a few local outliers
        min_turn_drop_frac=min_turn_drop_frac, # must actually reduce wiggles (set None to disable)
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        iters=iters,
        n_samples=n_samples,
        n_samples_search=n_samples_search,
        n_samples_final=n_samples_final,
        budget_abs_m=budget_abs_m,
        distance_mode=distance_mode,
        distance_simplify=distance_simplify,
        distance_simplify_tol=distance_simplify_tol,
        print_timing=print_timing,
    )
    best_line = out["line"]
    if print_output == True:
        print(out)
    return best_line
