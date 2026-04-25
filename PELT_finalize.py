from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
import pickle

import numpy as np
import pandas as pd

import PELT
import PELT_geometry_features as pgf
import incorporate_multichannel_segments as ims
from open_SWOT_files import open_SWOT_files

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None


DEFAULT_OUTDIR = Path("test_figures_final")
DEFAULT_RESULTS_PICKLE = "PELT_final_results_dict.pkl"
DEFAULT_SWOT_NODE_DIR = "/Volumes/PhD/SWOT/RiverSP_D_parq/node/"
DEFAULT_SWOT_REGION = "SA"
WINDOW_VERSION_TO_KEY = {"raw": 0, "w2": 2, "w3": 3, "w4": 4, "w5": 5}
WINDOW_VERSION_TO_N_WINDOWS = {"raw": 1, "w2": 2, "w3": 3, "w4": 4, "w5": 5}


def _feature_cols_need_geometry(pelt_feature_cols):
    return bool(set(PELT.normalize_feature_cols(pelt_feature_cols)) & {"sinu", "curv_int"})


def _feature_cols_need_wse(pelt_feature_cols):
    return "slope" in set(PELT.normalize_feature_cols(pelt_feature_cols))


def _lookup_centerline(centerlines, mip, centerline_id_col="main_path_id", centerline_geometry_col="line"):
    if centerlines is None:
        return None
    if isinstance(centerlines, dict):
        if mip not in centerlines:
            raise KeyError(f"Missing centerline for main path id {mip}.")
        return centerlines[mip]

    if centerline_id_col not in centerlines.columns:
        raise ValueError(f"centerlines is missing id column '{centerline_id_col}'.")
    if centerline_geometry_col not in centerlines.columns:
        raise ValueError(f"centerlines is missing geometry column '{centerline_geometry_col}'.")

    matches = centerlines.loc[centerlines[centerline_id_col] == mip]
    if matches.empty:
        raise KeyError(f"Missing centerline for {centerline_id_col}={mip}.")
    if len(matches) > 1:
        raise ValueError(f"Expected one centerline for {centerline_id_col}={mip}; found {len(matches)}.")
    return matches.iloc[0][centerline_geometry_col]


def _prepare_centerline_for_features(
    centerlines,
    mip,
    nodes_df,
    pelt_feature_cols,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    check_orientation=True,
):
    if not _feature_cols_need_geometry(pelt_feature_cols):
        return None, {}
    centerline = _lookup_centerline(
        centerlines,
        mip,
        centerline_id_col=centerline_id_col,
        centerline_geometry_col=centerline_geometry_col,
    )
    if centerline is None:
        raise ValueError(
            "Geometry feature columns were requested, but no centerlines lookup was provided."
        )

    geometry_summary = pgf.summarize_centerline_geometries(
        {mip: centerline},
        id_col=centerline_id_col,
        assert_all_linestring=True,
    )
    qa = {
        "centerline_id": mip,
        "centerline_geometry_type": str(geometry_summary["geometry_type"].iloc[0]),
        "centerline_length_m": float(geometry_summary["length_m"].iloc[0]),
        "centerline_is_multilinestring": bool(geometry_summary["is_multilinestring"].iloc[0]),
    }

    if check_orientation:
        centerline, orientation_qa = pgf.orient_centerline_to_node_dist(
            centerline,
            nodes_df,
            dist_col="dist_m",
            node_geometry_col=node_geometry_col,
            reverse_if_needed=True,
        )
        qa.update(orientation_qa)

    return centerline, qa


def _attach_geometry_qa(result, centerline_qa):
    if centerline_qa:
        result["geometry_qa"] = centerline_qa
    if result.get("geometry_features_by_window") is not None:
        result["geometry_feature_nan_rates"] = pgf.summarize_geometry_feature_nan_rates(
            result["features_by_window"],
            feature_cols=("sinu", "curv_int"),
        )
    return result


@dataclass(frozen=True)
class ExplicitSelectorSetting:
    min_support_frac_runs: float
    min_windows_supported: int
    stop_rel_improvement: float
    label: Optional[str] = None


def _build_window_runs():
    return {
        0: {
            "window_version": "raw",
            "window_selection_method": "raw",
            "window_selection": PELT.WindowSelectionConfig(method="raw"),
            "break_min_windows_supported": 1,
        },
        2: {
            "window_version": "w2",
            "window_selection_method": "width_quantile_log",
            "window_selection": PELT.WindowSelectionConfig(
                method="width_quantile_log",
                n_windows=2,
                width_col="multi_width",
                low_quantile=0.10,
                high_quantile=0.90,
                min_width_multiplier=36.0,
                max_width_multiplier=36.0,
                max_window_fraction_of_length=0.15,
            ),
            "break_min_windows_supported": 2,
        },
        3: {
            "window_version": "w3",
            "window_selection_method": "width_quantile_log",
            "window_selection": PELT.WindowSelectionConfig(
                method="width_quantile_log",
                n_windows=3,
                width_col="multi_width",
                low_quantile=0.10,
                high_quantile=0.90,
                min_width_multiplier=36.0,
                max_width_multiplier=36.0,
                max_window_fraction_of_length=0.15,
            ),
            "break_min_windows_supported": 2,
        },
        4: {
            "window_version": "w4",
            "window_selection_method": "width_quantile_log",
            "window_selection": PELT.WindowSelectionConfig(
                method="width_quantile_log",
                n_windows=4,
                width_col="multi_width",
                low_quantile=0.10,
                high_quantile=0.90,
                min_width_multiplier=36.0,
                max_width_multiplier=36.0,
                max_window_fraction_of_length=0.15,
            ),
            "break_min_windows_supported": 2,
        },
        5: {
            "window_version": "w5",
            "window_selection_method": "width_quantile_log",
            "window_selection": PELT.WindowSelectionConfig(
                method="width_quantile_log",
                n_windows=5,
                width_col="multi_width",
                low_quantile=0.10,
                high_quantile=0.90,
                min_width_multiplier=36.0,
                max_width_multiplier=36.0,
                max_window_fraction_of_length=0.15,
            ),
            "break_min_windows_supported": 2,
        },
    }


DEFAULT_STAGE1_W2_SELECTOR_SETTINGS: Tuple[ExplicitSelectorSetting, ...] = (
    ExplicitSelectorSetting(0.10, 1, 0.050),
    ExplicitSelectorSetting(0.05, 1, 0.050),
    ExplicitSelectorSetting(0.15, 1, 0.050),
    ExplicitSelectorSetting(0.10, 1, 0.045),
    ExplicitSelectorSetting(0.10, 1, 0.060),
    )


def get_stage1_w2_selector_settings() -> Tuple[ExplicitSelectorSetting, ...]:
    return DEFAULT_STAGE1_W2_SELECTOR_SETTINGS


def default_final_stable_support_count(n_selector_settings: int) -> int:
    """
    Default final support rule for retained selector settings.

    Require all settings when there are only two; otherwise allow one retained
    setting to disagree. This maps 2->2, 3->2, 4->3, 5->4, 6->5.
    """
    n = int(n_selector_settings)
    if n < 1:
        raise ValueError("n_selector_settings must be at least 1.")
    if n == 1:
        return 1
    return max(2, n - 1)


def derive_finalization_inputs_from_analysis_tables(
        analysis_tables: Dict[str, pd.DataFrame],
        stable_support_count: Optional[int] = None,
        consensus_cfg: PELT.ConsensusConfig = PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    )-> Dict[str, object]:
    winning_rep_df = analysis_tables.get("winning_representative_df", pd.DataFrame())
    recommended_grid_settings_df = analysis_tables.get("recommended_grid_settings_df", pd.DataFrame())

    if winning_rep_df.empty:
        raise ValueError("analysis_tables['winning_representative_df'] is empty.")
    if recommended_grid_settings_df.empty:
        raise ValueError("analysis_tables['recommended_grid_settings_df'] is empty.")

    winning_rep = winning_rep_df.iloc[0].copy()
    winning_family = str(winning_rep["window_version"])
    if winning_family not in WINDOW_VERSION_TO_KEY:
        raise ValueError(
            f"Unsupported winning_family '{winning_family}'. "
            f"Expected one of {sorted(WINDOW_VERSION_TO_KEY)}."
        )

    window_key = int(WINDOW_VERSION_TO_KEY[winning_family])
    n_windows_total = int(WINDOW_VERSION_TO_N_WINDOWS[winning_family])

    selector_settings = tuple(
        ExplicitSelectorSetting(
            min_support_frac_runs=float(row["min_support_frac_runs"]),
            min_windows_supported=int(
                round(float(row["min_support_frac_windows_effective"]) * n_windows_total)
            ),
            stop_rel_improvement=float(row["stop_rel_improvement"]),
        )
        for _, row in recommended_grid_settings_df.iterrows()
    )

    stable_support_count_effective = (
        default_final_stable_support_count(len(selector_settings))
        if stable_support_count is None
        else int(stable_support_count)
    )

    return {
        "winning_family": winning_family,
        "window_key": window_key,
        "n_windows_total": n_windows_total,
        "selector_settings": selector_settings,
        "selector_settings_df": recommended_grid_settings_df.copy(),
        "winning_representative": winning_rep,
        "stable_support_count": stable_support_count_effective,
        "consensus_cfg": consensus_cfg,
    }


def _format_float_label(x: float) -> str:
    s = f"{float(x):.3f}".rstrip("0").rstrip(".")
    return s.replace(".", "p")


def _selector_setting_label(setting: ExplicitSelectorSetting) -> str:
    if setting.label:
        return setting.label
    return (
        f"mw{int(setting.min_windows_supported)}_"
        f"runs{_format_float_label(setting.min_support_frac_runs)}_"
        f"stop{_format_float_label(setting.stop_rel_improvement)}"
    )


def _prepare_reach_nodes(
    df,
    dfN,
    mip,
    pelt_feature_cols=("width_s", "nch_s"),
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
    ):
    D, DN = ims.run_code(df, mip, dfN)

    dist_out_col = "hydro_dist_out" if "hydro_dist_out" in DN.columns else "dist_out"
    node_length_col = "node_length" if "node_length" in DN.columns else "node_len"
    DN = DN.sort_values(dist_out_col, ascending=False).copy()
    DN["dist_m"] = DN[node_length_col].cumsum()

    if _feature_cols_need_wse(pelt_feature_cols):
        nodeWSE = open_SWOT_files(
            D[["reach_id"]].copy(),
            swot_node_dir,
            swot_region,
        )
        DN = DN.drop(columns=["wse"], errors="ignore")
        DN = DN.merge(nodeWSE, how="left", on="node_id")
    return DN


def apply_explicit_selector_grid(
    results: Dict[str, object],
    selector_settings: Sequence[ExplicitSelectorSetting],
    windows: Optional[Sequence[str]] = None,
    feature_cols: Optional[Sequence[str]] = None,
    candidate_source: str = "stability",
    candidate_freq_min: Optional[float] = None,
    min_support_frac_windows: float = 0.0,
    stability_tolerance_m: Optional[float] = None,
    min_spacing_m: float = 10_000.0,
    min_reach_len_m: Optional[float] = None,
    max_breaks: int = 30,
    stop_abs_improvement: float = 0.0,
    window_weights: Optional[Dict[str, float]] = None,
    consensus_cfg: PELT.ConsensusConfig = PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    stable_support_frac_min: Optional[float] = None,
    stable_support_count: Optional[int] = None,
    attach_to_results: bool = True,
    verbose: bool = False,
):
    if stable_support_frac_min is not None and stable_support_count is not None:
        raise ValueError("Specify either stable_support_frac_min or stable_support_count, not both.")
    if not selector_settings:
        raise ValueError("selector_settings must be non-empty.")
    if str(results.get("run_status", "ok")) != "ok":
        if not attach_to_results:
            raise ValueError("Skipped base results can only be returned with attach_to_results=True.")
        updated_results = dict(results)
        updated_results["final_selection_grid"] = None
        updated_results["stable_breaks_m"] = list(results.get("stable_breaks_m", []))
        updated_results["stable_segments"] = list(results.get("stable_segments", []))
        updated_results["final_selection_grid_meta"] = {
            "status": str(results.get("run_status", "skipped")),
            "reason": str(results.get("run_status_reason", "")),
            "detail": str(results.get("run_status_detail", "")),
            "selector_settings": [s.__dict__ for s in selector_settings],
        }
        return updated_results
    if stable_support_frac_min is None and stable_support_count is None:
        stable_support_count = default_final_stable_support_count(len(selector_settings))

    feature_cols_used = tuple(feature_cols) if feature_cols is not None else tuple(
        results.get("final_selection_feature_cols", PELT.FEATURE_COLS_ALL)
    )
    n_windows_effective = max(int(len(results.get("standardized_by_window", {}))), 1)

    individual_results: Dict[str, PELT.BreakSelectionResult] = {}
    summary_rows: List[Dict[str, object]] = []
    break_rows: List[Dict[str, object]] = []

    for setting in selector_settings:
        label = _selector_setting_label(setting)
        min_windows_supported_effective = min(int(setting.min_windows_supported), n_windows_effective)
        try:
            sel = PELT.select_segment_breaks_from_results(
                results=results,
                windows=windows,
                feature_cols=feature_cols_used,
                candidate_source=candidate_source,
                candidate_freq_min=candidate_freq_min,
                min_support_frac_runs=float(setting.min_support_frac_runs),
                min_windows_supported=min_windows_supported_effective,
                min_support_frac_windows=min_support_frac_windows,
                stability_tolerance_m=stability_tolerance_m,
                min_spacing_m=min_spacing_m,
                min_reach_len_m=min_reach_len_m,
                max_breaks=max_breaks,
                stop_rel_improvement=float(setting.stop_rel_improvement),
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
                        "min_windows_supported": int(setting.min_windows_supported),
                        "min_windows_supported_effective": int(min_windows_supported_effective),
                        "min_support_frac_runs": float(setting.min_support_frac_runs),
                        "stop_rel_improvement": float(setting.stop_rel_improvement),
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
        final_sse = float(sel.history["total_sse"].iloc[-1]) if len(sel.history) else np.nan
        summary_rows.append(
            {
                "setting": label,
                "min_windows_supported": int(setting.min_windows_supported),
                "min_windows_supported_effective": int(min_windows_supported_effective),
                "min_support_frac_runs": float(setting.min_support_frac_runs),
                "stop_rel_improvement": float(setting.stop_rel_improvement),
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
                    "min_windows_supported": int(setting.min_windows_supported),
                    "min_windows_supported_effective": int(min_windows_supported_effective),
                    "min_support_frac_runs": float(setting.min_support_frac_runs),
                    "stop_rel_improvement": float(setting.stop_rel_improvement),
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

    n_settings_valid = int((summary_df["status"].astype(str) == "ok").sum()) if not summary_df.empty else 0
    consensus_df, stable_breaks_m, stable_segments, effective_support_frac, stable_support_count_effective = (
        PELT.build_grid_consensus_from_break_rows(
            break_rows=break_rows,
            ref_dist=ref_dist,
            n_settings_valid=n_settings_valid,
            consensus_cfg=consensus_cfg,
            stable_support_frac_min=stable_support_frac_min,
            stable_support_count=stable_support_count,
        )
    )

    if verbose:
        print(
            f"Explicit selector grid produced {len(stable_breaks_m)} stable breaks "
            f"with {consensus_cfg.method} at {consensus_cfg.merge_threshold_m / 1000.0:.2f} km, "
            f"threshold count={stable_support_count_effective}, frac={effective_support_frac:.3f}."
        )

    grid_result = PELT.BreakSelectionGridResult(
        individual_results=individual_results,
        summary=summary_df,
        consensus=consensus_df,
        stable_breaks_m=[float(b) for b in stable_breaks_m],
        stable_segments=stable_segments,
        stable_support_frac_min=effective_support_frac,
        stable_support_count=stable_support_count_effective,
        consensus_method=str(consensus_cfg.method),
        merge_threshold_m=float(consensus_cfg.merge_threshold_m),
        n_settings_valid=int(n_settings_valid),
    )

    if not attach_to_results:
        return grid_result

    updated_results = dict(results)
    updated_results["final_selection_grid"] = grid_result
    updated_results["stable_breaks_m"] = grid_result.stable_breaks_m
    updated_results["stable_segments"] = grid_result.stable_segments
    updated_results["final_selection_grid_meta"] = {
        "selector_settings": [s.__dict__ for s in selector_settings],
        "stable_support_count": stable_support_count_effective,
        "stable_support_frac_min_effective": effective_support_frac,
        "n_settings_valid": n_settings_valid,
        "n_windows_effective": n_windows_effective,
        "consensus_method": str(consensus_cfg.method),
        "merge_threshold_m": float(consensus_cfg.merge_threshold_m),
    }
    return updated_results


def _run_base_reach_pipeline(
    df,
    dfN,
    mip,
    window_key=2,
    penalties=(2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
    pelt_feature_cols=("width_s", "nch_s"),
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
    centerlines=None,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    geometry_feature_cfg=None,
    check_centerline_orientation=True,
    print_timings=False,
):
    window_runs = _build_window_runs()
    if window_key not in window_runs:
        raise ValueError(f"Unknown window_key {window_key}. Valid options are {list(window_runs)}.")

    cfg = window_runs[window_key]
    reach_nodes_df = _prepare_reach_nodes(
        df,
        dfN,
        mip,
        pelt_feature_cols=pelt_feature_cols,
        swot_node_dir=swot_node_dir,
        swot_region=swot_region,
    )
    centerline, centerline_qa = _prepare_centerline_for_features(
        centerlines=centerlines,
        mip=mip,
        nodes_df=reach_nodes_df,
        pelt_feature_cols=pelt_feature_cols,
        centerline_id_col=centerline_id_col,
        centerline_geometry_col=centerline_geometry_col,
        node_geometry_col=node_geometry_col,
        check_orientation=check_centerline_orientation,
    )

    result = PELT.run_full_pipeline(
        nodes_df=reach_nodes_df,
        feat_cfg=PELT.FeatureConfig(
            dist_col="dist_m",
            wse_col="wse",
            width_col="multi_width",
            nch_col="multi_n_chan",
            nch_summary="mean",
        ),
        pelt_cfg=PELT.PeltConfig(
            feature_cols=tuple(pelt_feature_cols),
            jump=5,
        ),
        pipe_cfg=PELT.PipelineConfig(
            penalties=tuple(penalties),
            window_selection_method=cfg["window_selection_method"],
            window_selection=cfg["window_selection"],
            break_selection=PELT.BreakSelectionConfig(
                enabled=False,
                min_support_frac_runs=0.20,
                min_windows_supported=cfg["break_min_windows_supported"],
                stop_rel_improvement=0.02,
            ),
            break_selection_grid=PELT.BreakSelectionGridConfig(enabled=False),
            print_timings=print_timings,
        ),
        centerline=centerline,
        geometry_feature_cfg=geometry_feature_cfg,
    )
    return _attach_geometry_qa(result, centerline_qa)


def _finalize_base_result(
    base_results,
    mip,
    window_key=2,
    pelt_feature_cols=("width_s", "nch_s"),
    selector_settings: Sequence[ExplicitSelectorSetting] = DEFAULT_STAGE1_W2_SELECTOR_SETTINGS,
    stable_support_count: Optional[int] = None,
    stable_support_frac_min: Optional[float] = None,
    consensus_cfg: PELT.ConsensusConfig = PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    make_plot=False,
    save_exports=False,
    outdir=DEFAULT_OUTDIR,
):
    window_runs = _build_window_runs()
    if window_key not in window_runs:
        raise ValueError(f"Unknown window_key {window_key}. Valid options are {list(window_runs)}.")
    cfg = window_runs[window_key]

    final_results = apply_explicit_selector_grid(
        results=base_results,
        selector_settings=selector_settings,
        feature_cols=tuple(pelt_feature_cols),
        stable_support_count=stable_support_count,
        stable_support_frac_min=stable_support_frac_min,
        consensus_cfg=consensus_cfg,
        attach_to_results=True,
    )

    if str(final_results.get("run_status", "ok")) != "ok":
        return final_results

    if make_plot:
        outdir = Path(outdir)
        if save_exports:
            outdir.mkdir(exist_ok=True, parents=True)
        PELT.plot_pelt_grid_results(
            final_results,
            core_min=float(final_results["final_selection_grid"].stable_support_frac_min),
            medium_min=0.50,
            make_plot=True,
            plot_title=f"{mip}_{cfg['window_version']}_final",
            save=save_exports,
            outdir=outdir,
        )

    return final_results


def build_base_results_batch(
    df,
    dfN,
    mips,
    window_key=2,
    penalties=(2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
    pelt_feature_cols=("width_s", "nch_s"),
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
    centerlines=None,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    geometry_feature_cfg=None,
    check_centerline_orientation=True,
    show_progress=False,
    print_timings=False,
):
    window_runs = _build_window_runs()
    if window_key not in window_runs:
        raise ValueError(f"Unknown window_key {window_key}. Valid options are {list(window_runs)}.")
    window_version = window_runs[window_key]["window_version"]

    results_dict = {}
    mip_iter = mips
    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(mips, desc=f"PELT base {window_version}", unit="reach")
        mip_iter = progress_bar

    for mip in mip_iter:
        if progress_bar is not None:
            progress_bar.set_postfix_str(f"reach={mip}")
        run_key = f"{mip}_{window_key}"
        results_dict[run_key] = _run_base_reach_pipeline(
            df=df,
            dfN=dfN,
            mip=mip,
            window_key=window_key,
            penalties=penalties,
            pelt_feature_cols=pelt_feature_cols,
            swot_node_dir=swot_node_dir,
            swot_region=swot_region,
            centerlines=centerlines,
            centerline_id_col=centerline_id_col,
            centerline_geometry_col=centerline_geometry_col,
            node_geometry_col=node_geometry_col,
            geometry_feature_cfg=geometry_feature_cfg,
            check_centerline_orientation=check_centerline_orientation,
            print_timings=print_timings,
        )

    return {
        "results_dict": results_dict,
        "window_version": window_version,
        "window_key": window_key,
    }


def run_final_reach_pipeline(
    df,
    dfN,
    mip,
    window_key=2,
    penalties=(2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
    pelt_feature_cols=("width_s", "nch_s"),
    selector_settings: Sequence[ExplicitSelectorSetting] = DEFAULT_STAGE1_W2_SELECTOR_SETTINGS,
    stable_support_count: Optional[int] = None,
    stable_support_frac_min: Optional[float] = None,
    consensus_cfg: PELT.ConsensusConfig = PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
    centerlines=None,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    geometry_feature_cfg=None,
    check_centerline_orientation=True,
    make_plot=False,
    save_exports=False,
    outdir=DEFAULT_OUTDIR,
    print_timings=False,
):
    base_results = _run_base_reach_pipeline(
        df=df,
        dfN=dfN,
        mip=mip,
        window_key=window_key,
        penalties=penalties,
        pelt_feature_cols=pelt_feature_cols,
        swot_node_dir=swot_node_dir,
        swot_region=swot_region,
        centerlines=centerlines,
        centerline_id_col=centerline_id_col,
        centerline_geometry_col=centerline_geometry_col,
        node_geometry_col=node_geometry_col,
        geometry_feature_cfg=geometry_feature_cfg,
        check_centerline_orientation=check_centerline_orientation,
        print_timings=print_timings,
    )

    return _finalize_base_result(
        base_results=base_results,
        mip=mip,
        window_key=window_key,
        pelt_feature_cols=pelt_feature_cols,
        selector_settings=selector_settings,
        stable_support_count=stable_support_count,
        stable_support_frac_min=stable_support_frac_min,
        consensus_cfg=consensus_cfg,
        make_plot=make_plot,
        save_exports=save_exports,
        outdir=outdir,
    )


def run_final_batch(
    df,
    dfN,
    mips,
    window_key=2,
    penalties=(2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
    pelt_feature_cols=("width_s", "nch_s"),
    selector_settings: Sequence[ExplicitSelectorSetting] = DEFAULT_STAGE1_W2_SELECTOR_SETTINGS,
    stable_support_count: Optional[int] = None,
    stable_support_frac_min: Optional[float] = None,
    consensus_cfg: PELT.ConsensusConfig = PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
    centerlines=None,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    geometry_feature_cfg=None,
    check_centerline_orientation=True,
    make_plots=False,
    save_exports=True,
    show_progress=False,
    print_timings=False,
    outdir=DEFAULT_OUTDIR,
    base_results_dict=None,
):
    window_runs = _build_window_runs()
    if window_key not in window_runs:
        raise ValueError(f"Unknown window_key {window_key}. Valid options are {list(window_runs)}.")
    window_version = window_runs[window_key]["window_version"]
    if stable_support_frac_min is not None and stable_support_count is not None:
        raise ValueError("Specify either stable_support_frac_min or stable_support_count, not both.")
    stable_support_count_effective = stable_support_count
    if stable_support_frac_min is None and stable_support_count_effective is None:
        stable_support_count_effective = default_final_stable_support_count(len(selector_settings))

    outdir = Path(outdir)
    if save_exports:
        outdir.mkdir(exist_ok=True, parents=True)

    results_dict = {}
    settings_tables = []
    consensus_tables = []
    run_summary_tables = []
    skipped_run_rows = []
    geometry_qa_rows = []
    geometry_nan_tables = []

    if base_results_dict is None:
        base_run_outputs = build_base_results_batch(
            df=df,
            dfN=dfN,
            mips=mips,
            window_key=window_key,
            penalties=penalties,
            pelt_feature_cols=pelt_feature_cols,
            swot_node_dir=swot_node_dir,
            swot_region=swot_region,
            centerlines=centerlines,
            centerline_id_col=centerline_id_col,
            centerline_geometry_col=centerline_geometry_col,
            node_geometry_col=node_geometry_col,
            geometry_feature_cfg=geometry_feature_cfg,
            check_centerline_orientation=check_centerline_orientation,
            show_progress=show_progress,
            print_timings=print_timings,
        )
        base_results_dict = base_run_outputs["results_dict"]

    mip_iter = mips
    progress_bar = None
    if show_progress and tqdm is not None and base_results_dict is not None:
        progress_bar = tqdm(mips, desc=f"Finalize {window_version}", unit="reach")
        mip_iter = progress_bar

    for mip in mip_iter:
        if progress_bar is not None:
            progress_bar.set_postfix_str(f"reach={mip}")
        run_key = f"{mip}_{window_key}"
        if run_key not in base_results_dict:
            raise KeyError(f"Missing base result for run_key '{run_key}'.")

        result = _finalize_base_result(
            base_results=base_results_dict[run_key],
            mip=mip,
            window_key=window_key,
            pelt_feature_cols=pelt_feature_cols,
            selector_settings=selector_settings,
            stable_support_count=stable_support_count_effective,
            stable_support_frac_min=stable_support_frac_min,
            consensus_cfg=consensus_cfg,
            make_plot=make_plots,
            save_exports=save_exports,
            outdir=outdir,
        )

        results_dict[run_key] = result
        if str(result.get("run_status", "ok")) != "ok":
            skipped_run_rows.append(
                {
                    "run_key": run_key,
                    "reach_id": mip,
                    "window_version": window_version,
                    "status": str(result.get("run_status", "skipped")),
                    "reason": str(result.get("run_status_reason", "")),
                    "detail": str(result.get("run_status_detail", "")),
                    "nominal_windows_m": list(result.get("resolved_windows_m_nominal", ())),
                    "feasible_windows_m": list(result.get("resolved_windows_m", ())),
                }
            )
            continue
        if result.get("geometry_qa"):
            geometry_qa_rows.append({"run_key": run_key, **dict(result["geometry_qa"])})
        nan_rates = result.get("geometry_feature_nan_rates")
        if nan_rates is not None and not nan_rates.empty:
            tmp = nan_rates.copy()
            tmp["run_key"] = run_key
            geometry_nan_tables.append(tmp)

        settings_df, consensus_df, run_summary_df = PELT.extract_pelt_grid_analysis_tables(
            result,
            reach_id=mip,
            window_version=window_version,
            medium_min=0.50,
        )

        settings_df["run_key"] = run_key
        consensus_df["run_key"] = run_key
        run_summary_df["run_key"] = run_key

        settings_tables.append(settings_df)
        consensus_tables.append(consensus_df)
        run_summary_tables.append(run_summary_df)

    grid_settings_master_df = (
        pd.concat(settings_tables, ignore_index=True)
        if settings_tables
        else pd.DataFrame()
    )
    grid_consensus_master_df = (
        pd.concat(consensus_tables, ignore_index=True)
        if consensus_tables
        else pd.DataFrame()
    )
    grid_run_summary_master_df = (
        pd.concat(run_summary_tables, ignore_index=True)
        if run_summary_tables
        else pd.DataFrame()
    )
    skipped_runs_df = pd.DataFrame(skipped_run_rows)
    geometry_qa_master_df = pd.DataFrame(geometry_qa_rows)
    geometry_feature_nan_rates_master_df = (
        pd.concat(geometry_nan_tables, ignore_index=True)
        if geometry_nan_tables
        else pd.DataFrame()
    )

    results_pickle_path = None
    if save_exports:
        grid_settings_master_df.to_csv(outdir / "PELT_final_settings_master.csv", index=False)
        grid_consensus_master_df.to_csv(outdir / "PELT_final_consensus_master.csv", index=False)
        grid_run_summary_master_df.to_csv(outdir / "PELT_final_run_summary_master.csv", index=False)
        if not skipped_runs_df.empty:
            skipped_runs_df.to_csv(outdir / "PELT_final_skipped_runs.csv", index=False)
        if not geometry_qa_master_df.empty:
            geometry_qa_master_df.to_csv(outdir / "PELT_final_geometry_qa_master.csv", index=False)
        if not geometry_feature_nan_rates_master_df.empty:
            geometry_feature_nan_rates_master_df.to_csv(
                outdir / "PELT_final_geometry_feature_nan_rates_master.csv",
                index=False,
            )

        results_pickle_path = outdir / DEFAULT_RESULTS_PICKLE
        with open(results_pickle_path, "wb") as f:
            pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    return {
        "results_dict": results_dict,
        "grid_settings_master_df": grid_settings_master_df,
        "grid_consensus_master_df": grid_consensus_master_df,
        "grid_run_summary_master_df": grid_run_summary_master_df,
        "skipped_runs_df": skipped_runs_df,
        "geometry_qa_master_df": geometry_qa_master_df,
        "geometry_feature_nan_rates_master_df": geometry_feature_nan_rates_master_df,
        "results_pickle_path": results_pickle_path,
        "window_version": window_version,
        "selector_settings": [s.__dict__ for s in selector_settings],
        "stable_support_count": stable_support_count_effective,
        "stable_support_frac_min": stable_support_frac_min,
        "consensus_method": str(consensus_cfg.method),
        "merge_threshold_m": float(consensus_cfg.merge_threshold_m),
        "pelt_feature_cols": tuple(pelt_feature_cols),
    }


def _infer_reach_id_from_run_key(run_key):
    run_key = str(run_key)
    if "_" in run_key:
        left, _ = run_key.rsplit("_", 1)
    else:
        left = run_key
    try:
        return int(left)
    except Exception:
        return left


def summarize_final_batch(final_outputs):
    results_dict = final_outputs["results_dict"]
    window_version = final_outputs.get("window_version", None)

    reach_rows = []
    for run_key, result in results_dict.items():
        grid = result.get("final_selection_grid", None)
        run_status = str(result.get("run_status", "ok"))
        if grid is None:
            reach_rows.append(
                {
                    "run_key": run_key,
                    "reach_id": _infer_reach_id_from_run_key(run_key),
                    "window_version": window_version,
                    "run_status": run_status,
                    "consensus_method": np.nan,
                    "merge_threshold_m": np.nan,
                    "merge_threshold_km": np.nan,
                    "stable_support_count": np.nan,
                    "stable_support_frac_effective": np.nan,
                    "n_settings_valid": np.nan,
                    "n_stable_breaks": 0,
                    "has_stable_breaks": 0,
                    "stable_breaks_m": [],
                    "stable_breaks_km": [],
                    "n_consensus_clusters": 0,
                    "n_core_clusters": 0,
                    "mean_core_span_km": np.nan,
                    "max_core_span_km": np.nan,
                    "mean_core_support": np.nan,
                }
            )
            continue

        meta = dict(result.get("final_selection_grid_meta", {}))
        support_frac_effective = float(
            meta.get("stable_support_frac_min_effective", getattr(grid, "stable_support_frac_min", np.nan))
        )
        support_count_effective = meta.get("stable_support_count", final_outputs.get("stable_support_count", np.nan))
        n_settings_valid = meta.get("n_settings_valid", np.nan)
        consensus_method = str(meta.get("consensus_method", getattr(grid, "consensus_method", "unknown")))
        merge_threshold_m = float(meta.get("merge_threshold_m", getattr(grid, "merge_threshold_m", np.nan)))

        consensus_df = grid.consensus.copy()
        if not consensus_df.empty:
            core_df = consensus_df[consensus_df["support_frac_grid"] >= support_frac_effective].copy()
            mean_core_span_km = (
                float(core_df["cluster_span_m"].mean()) / 1000.0 if len(core_df) else np.nan
            )
            max_core_span_km = (
                float(core_df["cluster_span_m"].max()) / 1000.0 if len(core_df) else np.nan
            )
            mean_core_support = float(core_df["support_frac_grid"].mean()) if len(core_df) else np.nan
            n_consensus_clusters = int(len(consensus_df))
            n_core_clusters = int(len(core_df))
        else:
            mean_core_span_km = np.nan
            max_core_span_km = np.nan
            mean_core_support = np.nan
            n_consensus_clusters = 0
            n_core_clusters = 0

        stable_breaks_m = list(result.get("stable_breaks_m", []))
        reach_rows.append(
            {
                "run_key": run_key,
                "reach_id": _infer_reach_id_from_run_key(run_key),
                "window_version": window_version,
                "run_status": run_status,
                "consensus_method": consensus_method,
                "merge_threshold_m": merge_threshold_m,
                "merge_threshold_km": merge_threshold_m / 1000.0 if np.isfinite(merge_threshold_m) else np.nan,
                "stable_support_count": support_count_effective,
                "stable_support_frac_effective": support_frac_effective,
                "n_settings_valid": n_settings_valid,
                "n_stable_breaks": int(len(stable_breaks_m)),
                "has_stable_breaks": int(len(stable_breaks_m) > 0),
                "stable_breaks_m": [float(b) for b in stable_breaks_m],
                "stable_breaks_km": [float(b) / 1000.0 for b in stable_breaks_m],
                "n_consensus_clusters": n_consensus_clusters,
                "n_core_clusters": n_core_clusters,
                "mean_core_span_km": mean_core_span_km,
                "max_core_span_km": max_core_span_km,
                "mean_core_support": mean_core_support,
            }
        )

    reach_summary_df = pd.DataFrame(reach_rows).sort_values("reach_id").reset_index(drop=True)
    if reach_summary_df.empty:
        raise ValueError("No finalized reach results available to summarize.")

    finalized_rows = reach_summary_df[reach_summary_df["run_status"] == "ok"].copy()
    support_count_value = (
        finalized_rows["stable_support_count"].iloc[0]
        if not finalized_rows.empty
        else np.nan
    )
    overall_summary_df = pd.DataFrame(
        [
            {
                "window_version": window_version,
                "consensus_method": (
                    str(finalized_rows["consensus_method"].iloc[0])
                    if not finalized_rows.empty
                    else np.nan
                ),
                "merge_threshold_m": (
                    float(finalized_rows["merge_threshold_m"].iloc[0])
                    if not finalized_rows.empty
                    else np.nan
                ),
                "merge_threshold_km": (
                    float(finalized_rows["merge_threshold_km"].iloc[0])
                    if not finalized_rows.empty
                    else np.nan
                ),
                "stable_support_count": support_count_value,
                "stable_support_frac_effective": float(finalized_rows["stable_support_frac_effective"].mean()),
                "reaches": int(len(reach_summary_df)),
                "reaches_finalized": int(len(finalized_rows)),
                "reaches_skipped": int((reach_summary_df["run_status"] != "ok").sum()),
                "reaches_with_stable_breaks": int(reach_summary_df["has_stable_breaks"].sum()),
                "frac_reaches_with_stable_breaks": float(reach_summary_df["has_stable_breaks"].mean()),
                "total_stable_breaks": int(reach_summary_df["n_stable_breaks"].sum()),
                "mean_n_stable_breaks": float(reach_summary_df["n_stable_breaks"].mean()),
                "median_n_stable_breaks": float(reach_summary_df["n_stable_breaks"].median()),
                "mean_n_consensus_clusters": float(finalized_rows["n_consensus_clusters"].mean()),
                "mean_n_core_clusters": float(finalized_rows["n_core_clusters"].mean()),
                "mean_core_span_km": float(finalized_rows["mean_core_span_km"].mean()),
                "mean_max_core_span_km": float(finalized_rows["max_core_span_km"].mean()),
                "mean_core_support": float(finalized_rows["mean_core_support"].mean()),
                "mean_n_settings_valid": float(finalized_rows["n_settings_valid"].mean()),
            }
        ]
    )

    return {
        "reach_summary_df": reach_summary_df,
        "overall_summary_df": overall_summary_df,
    }


def run_final_support_sweep(
    df,
    dfN,
    mips,
    support_counts=(3, 4, 5),
    window_key=2,
    penalties=(2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
    pelt_feature_cols=("width_s", "nch_s"),
    selector_settings: Sequence[ExplicitSelectorSetting] = DEFAULT_STAGE1_W2_SELECTOR_SETTINGS,
    consensus_cfg: PELT.ConsensusConfig = PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
    centerlines=None,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    geometry_feature_cfg=None,
    check_centerline_orientation=True,
    make_plots=False,
    save_exports=True,
    show_progress=False,
    print_timings=False,
    outdir=DEFAULT_OUTDIR,
):
    support_counts = tuple(int(v) for v in support_counts)
    if not support_counts:
        raise ValueError("support_counts must be non-empty.")

    outdir = Path(outdir)
    if save_exports:
        outdir.mkdir(exist_ok=True, parents=True)

    base_run_outputs = build_base_results_batch(
        df=df,
        dfN=dfN,
        mips=mips,
        window_key=window_key,
        penalties=penalties,
        pelt_feature_cols=pelt_feature_cols,
        swot_node_dir=swot_node_dir,
        swot_region=swot_region,
        centerlines=centerlines,
        centerline_id_col=centerline_id_col,
        centerline_geometry_col=centerline_geometry_col,
        node_geometry_col=node_geometry_col,
        geometry_feature_cfg=geometry_feature_cfg,
        check_centerline_orientation=check_centerline_orientation,
        show_progress=show_progress,
        print_timings=print_timings,
    )
    base_results_dict = base_run_outputs["results_dict"]

    outer_iter = support_counts
    outer_progress = None
    if show_progress and tqdm is not None:
        outer_progress = tqdm(support_counts, desc="Apply support count", unit="support")
        outer_iter = outer_progress

    outputs_by_support = {}
    reach_summary_tables = []
    overall_summary_tables = []

    for stable_support_count in outer_iter:
        if outer_progress is not None:
            outer_progress.set_postfix_str(f"support={stable_support_count}")

        batch_outdir = outdir / f"support_count_{int(stable_support_count)}"
        final_outputs = run_final_batch(
            df=df,
            dfN=dfN,
            mips=mips,
            window_key=window_key,
            penalties=penalties,
            pelt_feature_cols=pelt_feature_cols,
            selector_settings=selector_settings,
            stable_support_count=int(stable_support_count),
            stable_support_frac_min=None,
            consensus_cfg=consensus_cfg,
            swot_node_dir=swot_node_dir,
            swot_region=swot_region,
            centerlines=centerlines,
            centerline_id_col=centerline_id_col,
            centerline_geometry_col=centerline_geometry_col,
            node_geometry_col=node_geometry_col,
            geometry_feature_cfg=geometry_feature_cfg,
            check_centerline_orientation=check_centerline_orientation,
            make_plots=make_plots,
            save_exports=save_exports,
            show_progress=False,
            print_timings=print_timings,
            outdir=batch_outdir,
            base_results_dict=base_results_dict,
        )
        outputs_by_support[int(stable_support_count)] = final_outputs

        summary = summarize_final_batch(final_outputs)
        reach_summary_df = summary["reach_summary_df"].copy()
        overall_summary_df = summary["overall_summary_df"].copy()
        reach_summary_tables.append(reach_summary_df)
        overall_summary_tables.append(overall_summary_df)

    reach_summary_comparison_df = pd.concat(reach_summary_tables, ignore_index=True).sort_values(
        ["stable_support_count", "reach_id"]
    ).reset_index(drop=True)
    overall_support_summary_df = pd.concat(overall_summary_tables, ignore_index=True).sort_values(
        "stable_support_count"
    ).reset_index(drop=True)

    if save_exports:
        reach_summary_comparison_df.to_csv(outdir / "PELT_final_support_sweep_reach_summary.csv", index=False)
        overall_support_summary_df.to_csv(outdir / "PELT_final_support_sweep_summary.csv", index=False)

    return {
        "base_results_dict": base_results_dict,
        "outputs_by_support": outputs_by_support,
        "reach_summary_comparison_df": reach_summary_comparison_df,
        "overall_support_summary_df": overall_support_summary_df,
    }
