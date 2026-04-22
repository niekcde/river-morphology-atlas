from __future__ import annotations

import ast
import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import PELT


WINDOW_KEY_TO_VERSION = {
    0: "raw",
    2: "w2",
    3: "w3",
    4: "w4",
    5: "w5",
}


def infer_window_version_from_run_key(run_key: object) -> str:
    run_key = str(run_key)
    if "_" not in run_key:
        return "unknown"
    right = run_key.rsplit("_", 1)[1]
    if right.isdigit():
        return WINDOW_KEY_TO_VERSION.get(int(right), f"w{right}")
    return "unknown"


def weighted_median(values: Sequence[float], weights: Sequence[float]) -> float:
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cdf = np.cumsum(weights) / np.sum(weights)
    return float(values[np.searchsorted(cdf, 0.5, side="left")])


def collect_break_rows(results_dict: Dict[str, object]) -> pd.DataFrame:
    rows = []
    for run_key, result in results_dict.items():
        grid = result.get("final_selection_grid", None)
        if grid is None:
            continue

        individual_results = dict(grid.individual_results)
        n_settings_total = int(len(individual_results))
        window_version = infer_window_version_from_run_key(run_key)

        for setting, sel in individual_results.items():
            for b in sel.breaks_m:
                rows.append(
                    {
                        "run_key": str(run_key),
                        "window_version": window_version,
                        "setting": str(setting),
                        "break_m": round(float(b), 6),
                        "n_settings_total": n_settings_total,
                    }
                )

    return pd.DataFrame(rows)


def collapse_exact_positions(break_rows: pd.DataFrame) -> pd.DataFrame:
    if break_rows.empty:
        return pd.DataFrame(
            columns=[
                "run_key",
                "window_version",
                "break_m",
                "n_settings_total",
                "n_settings_here",
                "support_frac_here",
                "settings_here",
                "gap_prev_km",
                "gap_next_km",
            ]
        )

    out = (
        break_rows.groupby(["run_key", "window_version", "n_settings_total", "break_m"], as_index=False)
        .agg(
            n_settings_here=("setting", "nunique"),
            settings_here=("setting", lambda s: tuple(sorted(set(s)))),
        )
        .sort_values(["run_key", "break_m"])
        .reset_index(drop=True)
    )
    out["support_frac_here"] = out["n_settings_here"] / out["n_settings_total"]
    out["gap_prev_km"] = out.groupby("run_key")["break_m"].diff() / 1000.0
    out["gap_next_km"] = (
        out.groupby("run_key")["break_m"].shift(-1) - out["break_m"]
    ) / 1000.0
    return out


def build_support_aware_gap_table(
    unique_breaks: pd.DataFrame,
    dominant_frac_min: float = 0.15,
    satellite_frac_max: float = 0.05,
    supported_frac_min: float = 0.10,
) -> pd.DataFrame:
    rows = []
    for run_key, g in unique_breaks.groupby("run_key", sort=False):
        g = g.sort_values("break_m").reset_index(drop=True)
        if len(g) < 2:
            continue

        window_version = str(g["window_version"].iloc[0])
        n_settings_total = int(g["n_settings_total"].iloc[0])

        for i in range(len(g) - 1):
            left_frac = float(g.loc[i, "support_frac_here"])
            right_frac = float(g.loc[i + 1, "support_frac_here"])
            gap_km = float((g.loc[i + 1, "break_m"] - g.loc[i, "break_m"]) / 1000.0)

            fmin = min(left_frac, right_frac)
            fmax = max(left_frac, right_frac)

            if fmax >= dominant_frac_min and fmin <= satellite_frac_max:
                pair_type = "likely_same"
            elif fmin >= supported_frac_min:
                pair_type = "risky"
            else:
                pair_type = "intermediate"

            rows.append(
                {
                    "run_key": str(run_key),
                    "window_version": window_version,
                    "n_settings_total": n_settings_total,
                    "left_break_km": float(g.loc[i, "break_m"] / 1000.0),
                    "right_break_km": float(g.loc[i + 1, "break_m"] / 1000.0),
                    "gap_km": gap_km,
                    "left_support_frac": left_frac,
                    "right_support_frac": right_frac,
                    "pair_type": pair_type,
                }
            )

    return pd.DataFrame(rows)


def compute_global_merge_threshold(
    pair_df: pd.DataFrame,
    local_upper_km: float = 20.0,
    families: Sequence[str] = ("w2", "w3", "w4", "w5"),
    min_likely_pairs: int = 2,
    round_to_km: float = 0.5,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    local_df = pair_df[
        (pair_df["gap_km"] <= float(local_upper_km))
        & (pair_df["window_version"].isin(tuple(families)))
    ].copy()

    rows = []
    for family, g in local_df.groupby("window_version", sort=False):
        likely = g[g["pair_type"] == "likely_same"].copy()
        risky = g[g["pair_type"] == "risky"].copy()
        if len(likely) < int(min_likely_pairs):
            continue

        rows.append(
            {
                "window_version": family,
                "n_likely_same_pairs": int(len(likely)),
                "n_risky_pairs": int(len(risky)),
                "q90_same_km": float(likely["gap_km"].quantile(0.90)),
                "q95_same_km": float(likely["gap_km"].quantile(0.95)),
                "max_same_km": float(likely["gap_km"].max()),
            }
        )

    fam_df = pd.DataFrame(rows).sort_values("window_version").reset_index(drop=True)
    if fam_df.empty:
        raise ValueError("No families passed min_likely_pairs for the requested local_upper_km.")

    weights = fam_df["n_likely_same_pairs"].to_numpy(dtype=float)
    global_q90_mean = float(np.average(fam_df["q90_same_km"], weights=weights))
    global_q95_mean = float(np.average(fam_df["q95_same_km"], weights=weights))
    global_q90_med = weighted_median(fam_df["q90_same_km"], weights)
    global_q95_med = weighted_median(fam_df["q95_same_km"], weights)

    global_q90 = 0.5 * (global_q90_mean + global_q90_med)
    global_q95 = 0.5 * (global_q95_mean + global_q95_med)
    midpoint_km = 0.5 * (global_q90 + global_q95)
    rounded_threshold_km = round(midpoint_km / float(round_to_km)) * float(round_to_km)

    summary = pd.Series(
        {
            "local_upper_km": float(local_upper_km),
            "families_used": list(fam_df["window_version"]),
            "min_likely_pairs": int(min_likely_pairs),
            "global_q90_same_km": float(global_q90),
            "global_q95_same_km": float(global_q95),
            "midpoint_km": float(midpoint_km),
            "rounded_threshold_km": float(rounded_threshold_km),
        }
    )

    return fam_df, summary, local_df


def sweep_local_upper_thresholds(
    pair_df: pd.DataFrame,
    local_upper_values_km: Iterable[float],
    families: Sequence[str] = ("w2", "w3", "w4", "w5"),
    min_likely_pairs: int = 2,
    round_to_km: float = 0.5,
) -> pd.DataFrame:
    rows = []
    for upper in local_upper_values_km:
        try:
            _, summary, _ = compute_global_merge_threshold(
                pair_df=pair_df,
                local_upper_km=float(upper),
                families=families,
                min_likely_pairs=min_likely_pairs,
                round_to_km=round_to_km,
            )
            rows.append(dict(summary))
        except ValueError:
            rows.append(
                {
                    "local_upper_km": float(upper),
                    "families_used": [],
                    "min_likely_pairs": int(min_likely_pairs),
                    "global_q90_same_km": np.nan,
                    "global_q95_same_km": np.nan,
                    "midpoint_km": np.nan,
                    "rounded_threshold_km": np.nan,
                }
            )
    return pd.DataFrame(rows).sort_values("local_upper_km").reset_index(drop=True)


def _normalize_families(x: object) -> Tuple[str, ...]:
    if isinstance(x, str):
        try:
            x = ast.literal_eval(x)
        except Exception:
            x = [x]
    return tuple(sorted(str(v) for v in x))


def find_first_stable_plateau(
    sweep_df: pd.DataFrame,
    target_families: Sequence[str] = ("w2", "w3", "w4", "w5"),
    min_consecutive: int = 3,
    tol_km: float = 0.0,
    require_all_target_families: bool = False,
    min_plateau_families: int = 3,
) -> Tuple[Optional[pd.Series], pd.DataFrame]:
    df = sweep_df.copy()
    df["families_used_norm"] = df["families_used"].apply(_normalize_families)
    target_families = tuple(sorted(str(v) for v in target_families))
    df["has_all_families"] = df["families_used_norm"].apply(
        lambda fams: all(f in fams for f in target_families)
    )
    df["n_families_used"] = df["families_used_norm"].apply(len)
    df = df.sort_values("local_upper_km").reset_index(drop=True)

    for i in range(len(df) - int(min_consecutive) + 1):
        sub = df.iloc[i : i + int(min_consecutive)].copy()
        if require_all_target_families:
            if not sub["has_all_families"].all():
                continue
        else:
            if not (sub["n_families_used"] >= int(min_plateau_families)).all():
                continue
            if len(set(sub["families_used_norm"])) != 1:
                continue

        thr = sub["rounded_threshold_km"].to_numpy(dtype=float)
        if np.nanmax(thr) - np.nanmin(thr) <= float(tol_km):
            families_used = list(sub["families_used_norm"].iloc[0])
            summary = pd.Series(
                {
                    "plateau_start_local_upper_km": float(sub["local_upper_km"].iloc[0]),
                    "plateau_end_local_upper_km": float(sub["local_upper_km"].iloc[-1]),
                    "n_points_in_plateau_check": int(len(sub)),
                    "target_families": list(target_families),
                    "families_used": families_used,
                    "require_all_target_families": bool(require_all_target_families),
                    "min_plateau_families": int(min_plateau_families),
                    "rounded_threshold_km": float(sub["rounded_threshold_km"].iloc[0]),
                    "midpoint_km_first_row": float(sub["midpoint_km"].iloc[0]),
                    "tol_km": float(tol_km),
                }
            )
            return summary, sub

    return None, pd.DataFrame()


def calibrate_consensus_from_results_dict(
    results_dict: Dict[str, object],
    local_upper_values_km: Sequence[float] = (15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120),
    target_families: Sequence[str] = ("w2", "w3", "w4", "w5"),
    min_likely_pairs: int = 2,
    min_consecutive: int = 3,
    tol_km: float = 0.0,
    round_to_km: float = 0.5,
    require_all_target_families: bool = False,
    min_plateau_families: int = 3,
) -> Dict[str, object]:
    break_rows = collect_break_rows(results_dict)
    unique_breaks = collapse_exact_positions(break_rows)
    pair_df = build_support_aware_gap_table(unique_breaks)
    sweep_df = sweep_local_upper_thresholds(
        pair_df=pair_df,
        local_upper_values_km=local_upper_values_km,
        families=target_families,
        min_likely_pairs=min_likely_pairs,
        round_to_km=round_to_km,
    )
    plateau_summary, plateau_rows = find_first_stable_plateau(
        sweep_df=sweep_df,
        target_families=target_families,
        min_consecutive=min_consecutive,
        tol_km=tol_km,
        require_all_target_families=require_all_target_families,
        min_plateau_families=min_plateau_families,
    )
    if plateau_summary is None:
        raise ValueError("No stable eligible-family plateau found in the local_upper_km sweep.")

    chosen_families = tuple(plateau_summary.get("families_used", target_families))

    family_thresholds, chosen_summary, local_df = compute_global_merge_threshold(
        pair_df=pair_df,
        local_upper_km=float(plateau_summary["plateau_start_local_upper_km"]),
        families=chosen_families,
        min_likely_pairs=min_likely_pairs,
        round_to_km=round_to_km,
    )
    consensus_cfg = PELT.ConsensusConfig(
        method="complete_linkage",
        merge_threshold_m=float(chosen_summary["rounded_threshold_km"]) * 1000.0,
        calibration_label="full_grid_first_stable_plateau",
    )

    return {
        "break_rows": break_rows,
        "unique_breaks": unique_breaks,
        "pair_df": pair_df,
        "local_pair_df": local_df,
        "sweep_df": sweep_df,
        "plateau_summary": plateau_summary,
        "plateau_rows": plateau_rows,
        "family_thresholds": family_thresholds,
        "chosen_summary": chosen_summary,
        "consensus_cfg": consensus_cfg,
    }


def save_calibration_artifacts(
    calibration_outputs: Dict[str, object],
    outdir: str | Path,
) -> Dict[str, Path]:
    outdir = Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    sweep_path = outdir / "PELT_consensus_sweep.csv"
    family_path = outdir / "PELT_consensus_family_thresholds.csv"
    pair_path = outdir / "PELT_consensus_local_pairs.csv"
    config_path = outdir / "PELT_consensus_config.json"

    calibration_outputs["sweep_df"].to_csv(sweep_path, index=False)
    calibration_outputs["family_thresholds"].to_csv(family_path, index=False)
    calibration_outputs["local_pair_df"].to_csv(pair_path, index=False)

    payload = {
        "consensus_cfg": asdict(calibration_outputs["consensus_cfg"]),
        "plateau_summary": dict(calibration_outputs["plateau_summary"]),
        "chosen_summary": dict(calibration_outputs["chosen_summary"]),
    }
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return {
        "sweep_path": sweep_path,
        "family_path": family_path,
        "pair_path": pair_path,
        "config_path": config_path,
    }


def load_consensus_config(path: str | Path) -> PELT.ConsensusConfig:
    path = Path(path)
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return PELT.ConsensusConfig(**payload["consensus_cfg"])
