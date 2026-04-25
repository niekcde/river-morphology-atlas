import numpy as np
import pandas as pd
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import PELT
import PELT_consensus_calibration as pcc
import PELT_geometry_features as pgf
import incorporate_multichannel_segments as ims
from open_SWOT_files import open_SWOT_files

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

DEFAULT_OUTDIR = Path("test_figures")
DEFAULT_RESULTS_PICKLE = "PELT_results_dict.pkl"
DEFAULT_SWOT_NODE_DIR = "/Volumes/PhD/SWOT/RiverSP_D_parq/node/"
DEFAULT_SWOT_REGION = "SA"


def _feature_cols_need_geometry(pelt_feature_cols):
    return bool(set(PELT.normalize_feature_cols(pelt_feature_cols)) & {"sinu", "curv_int"})


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


def _build_window_runs():
    return {
        0: {
            "window_version": "raw",
            "window_selection_method": "raw",
            "window_selection": PELT.WindowSelectionConfig(method="raw"),
            "break_min_windows_supported": 1,
            "grid_min_windows_supported_values": (1,),
        },
        2: {
            "window_version": "w2",
            "window_selection_method": "width_quantile_log",
            "window_selection": PELT.WindowSelectionConfig(
                method="width_quantile_log",
                n_windows=2,
                width_col="multi_width",
                low_quantile=0.25,
                high_quantile=0.75,
                min_width_multiplier=36.0,
                max_width_multiplier=36.0,
                max_window_fraction_of_length=0.15,
            ),
            "break_min_windows_supported": 2,
            "grid_min_windows_supported_values": (1,),
        },
        3: {
            "window_version": "w3",
            "window_selection_method": "width_quantile_log",
            "window_selection": PELT.WindowSelectionConfig(
                method="width_quantile_log",
                n_windows=3,
                width_col="multi_width",
                low_quantile=0.25,
                high_quantile=0.75,
                min_width_multiplier=36.0,
                max_width_multiplier=36.0,
                max_window_fraction_of_length=0.15,
            ),
            "break_min_windows_supported": 2,
            "grid_min_windows_supported_values": (1, 2),
        },
        4: {
            "window_version": "w4",
            "window_selection_method": "width_quantile_log",
            "window_selection": PELT.WindowSelectionConfig(
                method="width_quantile_log",
                n_windows=4,
                width_col="multi_width",
                low_quantile=0.25,
                high_quantile=0.75,
                min_width_multiplier=36.0,
                max_width_multiplier=36.0,
                max_window_fraction_of_length=0.15,
            ),
            "break_min_windows_supported": 2,
            "grid_min_windows_supported_values": (1, 2, 3),
        },
        5: {
            "window_version": "w5",
            "window_selection_method": "width_quantile_log",
            "window_selection": PELT.WindowSelectionConfig(
                method="width_quantile_log",
                n_windows=5,
                width_col="multi_width",
                low_quantile=0.25,
                high_quantile=0.75,
                min_width_multiplier=36.0,
                max_width_multiplier=36.0,
                max_window_fraction_of_length=0.15,
            ),
            "break_min_windows_supported": 2,
            "grid_min_windows_supported_values": (1, 2, 3),
        },
    }


def _select_window_keys(input_windows, window_runs):
    selected_window_keys = [w for w in window_runs if w in set(input_windows)]
    if not selected_window_keys:
        raise ValueError(
            f"No valid input_windows selected. Got {list(input_windows)}. "
            f"Valid options are {list(window_runs)}."
        )
    return selected_window_keys


def _prepare_reach_nodes(
    df,
    dfN,
    mip,
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
):
    D, DN = ims.run_code(df, mip, dfN)

    nodeWSE = open_SWOT_files(
        D[["reach_id"]].copy(),
        swot_node_dir,
        swot_region,
    )

    DN = DN.sort_values("hydro_dist_out", ascending=False).copy()
    DN["dist_m"] = DN["node_length"].cumsum()

    DN = DN.drop(columns=["wse"], axis=1)
    DN = DN.merge(nodeWSE, how="left", on="node_id")
    return DN


def _run_single_window_job(
    nodes_df,
    mip,
    window_key,
    PELT_penalties,
    min_support_frac_runs_values,
    stop_rel_improvement_values,
    consensus_cfg,
    pelt_feature_cols,
    make_plots,
    print_timings,
    save_exports,
    outdir,
    centerline=None,
    geometry_feature_cfg=None,
    centerline_qa=None,
    ):
    window_runs = _build_window_runs()
    cfg = window_runs[window_key]

    result = PELT.run_full_pipeline(
        nodes_df=nodes_df,
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
            penalties=PELT_penalties,
            window_selection_method=cfg["window_selection_method"],
            window_selection=cfg["window_selection"],
            break_selection=PELT.BreakSelectionConfig(
                min_support_frac_runs=0.20,
                min_windows_supported=cfg["break_min_windows_supported"],
                stop_rel_improvement=0.02,
            ),
            break_selection_grid=PELT.BreakSelectionGridConfig(
                enabled=True,
                min_support_frac_runs_values=min_support_frac_runs_values,
                min_windows_supported_values=cfg["grid_min_windows_supported_values"],
                stop_rel_improvement_values=stop_rel_improvement_values,
                consensus=consensus_cfg,
                stable_support_frac_min=PELT.DEFAULT_TUNING_STABLE_SUPPORT_FRAC_MIN,
            ),
            print_timings=print_timings,
        ),
        centerline=centerline,
        geometry_feature_cfg=geometry_feature_cfg,
    )
    result = _attach_geometry_qa(result, centerline_qa)

    if make_plots:
        PELT.plot_pelt_grid_results(
            result,
            plot_title=f"{mip}_{window_key}",
            core_min=PELT.DEFAULT_TUNING_STABLE_SUPPORT_FRAC_MIN,
            medium_min=0.50,
            save=save_exports,
            outdir=outdir,
        )

    settings_df, consensus_df, run_summary_df = PELT.extract_pelt_grid_analysis_tables(
        result,
        reach_id=mip,
        window_version=cfg["window_version"],
        medium_min=0.50,
    )

    return {
        "run_key": f"{mip}_{window_key}",
        "result": result,
        "settings_df": settings_df,
        "consensus_df": consensus_df,
        "run_summary_df": run_summary_df,
    }


def _append_job_output(job_output, results_dict, settings_tables, consensus_tables, run_summary_tables):
    results_dict[job_output["run_key"]] = job_output["result"]
    settings_tables.append(job_output["settings_df"])
    consensus_tables.append(job_output["consensus_df"])
    run_summary_tables.append(job_output["run_summary_df"])


def _finalize_grid_outputs(
    results_dict,
    settings_tables,
    consensus_tables,
    run_summary_tables,
    save_exports,
    outdir,
    consensus_cfg,
    pelt_feature_cols=("width_s", "nch_s"),
):
    grid_settings_master_df = pd.concat(settings_tables, ignore_index=True)
    grid_consensus_master_df = pd.concat(consensus_tables, ignore_index=True)
    grid_run_summary_master_df = pd.concat(run_summary_tables, ignore_index=True)
    geometry_qa_rows = []
    geometry_nan_tables = []
    for run_key, result in results_dict.items():
        if result.get("geometry_qa"):
            geometry_qa_rows.append({"run_key": run_key, **dict(result["geometry_qa"])})
        nan_rates = result.get("geometry_feature_nan_rates")
        if nan_rates is not None and not nan_rates.empty:
            tmp = nan_rates.copy()
            tmp["run_key"] = run_key
            geometry_nan_tables.append(tmp)
    geometry_qa_master_df = pd.DataFrame(geometry_qa_rows)
    geometry_feature_nan_rates_master_df = (
        pd.concat(geometry_nan_tables, ignore_index=True)
        if geometry_nan_tables
        else pd.DataFrame()
    )

    results_pickle_path = None
    if save_exports:
        grid_settings_master_df.to_csv(outdir / "PELT_grid_settings_master.csv", index=False)
        grid_consensus_master_df.to_csv(outdir / "PELT_grid_consensus_master.csv", index=False)
        grid_run_summary_master_df.to_csv(outdir / "PELT_grid_run_summary_master.csv", index=False)
        if not geometry_qa_master_df.empty:
            geometry_qa_master_df.to_csv(outdir / "PELT_geometry_qa_master.csv", index=False)
        if not geometry_feature_nan_rates_master_df.empty:
            geometry_feature_nan_rates_master_df.to_csv(
                outdir / "PELT_geometry_feature_nan_rates_master.csv",
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
        "geometry_qa_master_df": geometry_qa_master_df,
        "geometry_feature_nan_rates_master_df": geometry_feature_nan_rates_master_df,
        "results_pickle_path": results_pickle_path,
        "consensus_cfg": consensus_cfg,
        "consensus_method": str(consensus_cfg.method),
        "merge_threshold_m": float(consensus_cfg.merge_threshold_m),
        "pelt_feature_cols": tuple(pelt_feature_cols),
    }


def _infer_grid_output_outdir(grid_outputs, fallback=DEFAULT_OUTDIR):
    results_pickle_path = grid_outputs.get("results_pickle_path")
    if results_pickle_path is not None:
        return Path(results_pickle_path).parent
    return Path(fallback)


def _grid_break_rows_from_individual_results(grid_result):
    rows = []
    for setting, sel in grid_result.individual_results.items():
        for b in sel.breaks_m:
            rows.append(
                {
                    "setting": str(setting),
                    "break_m": float(b),
                }
            )
    return rows


def _reference_dist_for_grid_consensus(result):
    std_by_window = result["standardized_by_window"]
    windows_used = sorted(list(std_by_window.keys()))
    ref_df = std_by_window[windows_used[0]]
    dist_col = "dist_m" if "dist_m" in ref_df.columns else result.get("dist_col_used", "dist_m")
    return ref_df[dist_col].to_numpy(dtype=float)


def _reapply_consensus_to_result(
    result,
    consensus_cfg,
    stable_support_frac_min=None,
):
    grid = result.get("final_selection_grid", None)
    if grid is None:
        raise ValueError("Result is missing final_selection_grid; cannot reapply consensus.")

    if stable_support_frac_min is None:
        stable_support_frac_min = float(grid.stable_support_frac_min)

    break_rows = _grid_break_rows_from_individual_results(grid)
    ref_dist = _reference_dist_for_grid_consensus(result)

    consensus_df, stable_breaks_m, stable_segments, effective_support_frac, effective_support_count = (
        PELT.build_grid_consensus_from_break_rows(
            break_rows=break_rows,
            ref_dist=ref_dist,
            n_settings_valid=int(grid.n_settings_valid),
            consensus_cfg=consensus_cfg,
            stable_support_frac_min=float(stable_support_frac_min),
            stable_support_count=None,
        )
    )

    updated_grid = PELT.BreakSelectionGridResult(
        individual_results=grid.individual_results,
        summary=grid.summary.copy(),
        consensus=consensus_df,
        stable_breaks_m=[float(b) for b in stable_breaks_m],
        stable_segments=stable_segments,
        stable_support_frac_min=float(effective_support_frac),
        stable_support_count=effective_support_count,
        consensus_method=str(consensus_cfg.method),
        merge_threshold_m=float(consensus_cfg.merge_threshold_m),
        n_settings_valid=int(grid.n_settings_valid),
    )

    updated_result = dict(result)
    updated_result["final_selection_grid"] = updated_grid
    updated_result["stable_breaks_m"] = updated_grid.stable_breaks_m
    updated_result["stable_segments"] = updated_grid.stable_segments
    updated_result["final_selection_grid_meta"] = {
        **dict(result.get("final_selection_grid_meta", {})),
        "stable_support_count": effective_support_count,
        "stable_support_frac_min_effective": effective_support_frac,
        "n_settings_valid": int(grid.n_settings_valid),
        "consensus_method": str(consensus_cfg.method),
        "merge_threshold_m": float(consensus_cfg.merge_threshold_m),
    }
    return updated_result


def reapply_consensus_to_grid_outputs(
    grid_outputs,
    consensus_cfg,
    save_exports=True,
    outdir=None,
    stable_support_frac_min=None,
):
    """
    Recluster already-computed selector-grid breakpoints with a new consensus
    configuration. This avoids rerunning the expensive PELT grid search.
    """
    outdir = Path(outdir) if outdir is not None else _infer_grid_output_outdir(grid_outputs)
    if save_exports:
        outdir.mkdir(exist_ok=True, parents=True)

    updated_results_dict = {}
    settings_tables = []
    consensus_tables = []
    run_summary_tables = []

    for run_key, result in grid_outputs["results_dict"].items():
        updated_result = _reapply_consensus_to_result(
            result=result,
            consensus_cfg=consensus_cfg,
            stable_support_frac_min=stable_support_frac_min,
        )
        updated_results_dict[run_key] = updated_result

        reach_id, window_version = infer_run_meta(run_key, updated_result)
        settings_df, consensus_df, run_summary_df = PELT.extract_pelt_grid_analysis_tables(
            updated_result,
            reach_id=reach_id,
            window_version=window_version,
            medium_min=0.50,
        )
        settings_tables.append(settings_df)
        consensus_tables.append(consensus_df)
        run_summary_tables.append(run_summary_df)

    updated_outputs = _finalize_grid_outputs(
        results_dict=updated_results_dict,
        settings_tables=settings_tables,
        consensus_tables=consensus_tables,
        run_summary_tables=run_summary_tables,
        save_exports=save_exports,
        outdir=outdir,
        consensus_cfg=consensus_cfg,
        pelt_feature_cols=grid_outputs.get("pelt_feature_cols", ("width_s", "nch_s")),
    )

    for key, value in grid_outputs.items():
        if key not in updated_outputs:
            updated_outputs[key] = value

    return updated_outputs


def calibrate_and_reapply_consensus_to_grid_outputs(
    grid_outputs,
    save_exports=True,
    outdir=None,
    calibration_outdir=None,
    local_upper_values_km=(15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120),
    target_families=("w2", "w3", "w4", "w5"),
    min_likely_pairs=2,
    min_consecutive=3,
    tol_km=0.0,
    round_to_km=0.5,
    require_all_target_families=False,
    min_plateau_families=3,
    stable_support_frac_min=None,
):
    """
    Estimate the consensus merge threshold from the full grid output, then
    reapply that calibrated threshold to the existing selector-grid results.
    """
    outdir = Path(outdir) if outdir is not None else _infer_grid_output_outdir(grid_outputs)
    calibration_outdir = Path(calibration_outdir) if calibration_outdir is not None else outdir

    calibration_outputs = pcc.calibrate_consensus_from_results_dict(
        results_dict=grid_outputs["results_dict"],
        local_upper_values_km=local_upper_values_km,
        target_families=target_families,
        min_likely_pairs=min_likely_pairs,
        min_consecutive=min_consecutive,
        tol_km=tol_km,
        round_to_km=round_to_km,
        require_all_target_families=require_all_target_families,
        min_plateau_families=min_plateau_families,
    )

    if save_exports:
        pcc.save_calibration_artifacts(
            calibration_outputs=calibration_outputs,
            outdir=calibration_outdir,
        )

    updated_outputs = reapply_consensus_to_grid_outputs(
        grid_outputs=grid_outputs,
        consensus_cfg=calibration_outputs["consensus_cfg"],
        save_exports=save_exports,
        outdir=outdir,
        stable_support_frac_min=stable_support_frac_min,
    )
    updated_outputs["consensus_calibration"] = calibration_outputs
    updated_outputs["consensus_calibration_outdir"] = calibration_outdir
    return updated_outputs


def _PELT_grid_search_impl(
    df,
    dfN,
    mips,
    input_windows,
    PELT_penalties,
    min_support_frac_runs_values,
    stop_rel_improvement_values,
    pelt_feature_cols=("width_s", "nch_s"),
    make_plots=True,
    print_timings=True,
    save_exports=True,
    show_progress=False,
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
    outdir=DEFAULT_OUTDIR,
    consensus_cfg=PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    centerlines=None,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    geometry_feature_cfg=None,
    check_centerline_orientation=True,
    parallel_windows=False,
    max_workers=None,
):
    window_runs = _build_window_runs()
    results_dict = {}
    settings_tables = []
    consensus_tables = []
    run_summary_tables = []

    outdir = Path(outdir)
    if save_exports:
        outdir.mkdir(exist_ok=True, parents=True)

    selected_window_keys = _select_window_keys(input_windows, window_runs)

    mip_iter = mips
    progress_bar = None
    if show_progress and tqdm is not None:
        progress_bar = tqdm(mips, desc="PELT grid search", unit="reach")
        mip_iter = progress_bar

    executor = ProcessPoolExecutor(max_workers=max_workers) if parallel_windows else None
    try:
        for mip in mip_iter:
            if progress_bar is not None:
                progress_bar.set_postfix_str(f"reach={mip}")

            reach_nodes_df = _prepare_reach_nodes(
                df,
                dfN,
                mip,
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

            if executor is None:
                for window_key in selected_window_keys:
                    job_output = _run_single_window_job(
                        nodes_df=reach_nodes_df,
                        mip=mip,
                        window_key=window_key,
                        PELT_penalties=PELT_penalties,
                        min_support_frac_runs_values=min_support_frac_runs_values,
                        stop_rel_improvement_values=stop_rel_improvement_values,
                        consensus_cfg=consensus_cfg,
                        pelt_feature_cols=pelt_feature_cols,
                        make_plots=make_plots,
                        print_timings=print_timings,
                        save_exports=save_exports,
                        outdir=outdir,
                        centerline=centerline,
                        geometry_feature_cfg=geometry_feature_cfg,
                        centerline_qa=centerline_qa,
                    )
                    _append_job_output(
                        job_output,
                        results_dict,
                        settings_tables,
                        consensus_tables,
                        run_summary_tables,
                    )
            else:
                future_map = {
                    executor.submit(
                        _run_single_window_job,
                        reach_nodes_df,
                        mip,
                        window_key,
                        PELT_penalties,
                        min_support_frac_runs_values,
                        stop_rel_improvement_values,
                        consensus_cfg,
                        pelt_feature_cols,
                        make_plots,
                        print_timings,
                        save_exports,
                        outdir,
                        centerline,
                        geometry_feature_cfg,
                        centerline_qa,
                    ): window_key
                    for window_key in selected_window_keys
                }
                job_outputs_by_window = {}
                for future in as_completed(future_map):
                    window_key = future_map[future]
                    job_outputs_by_window[window_key] = future.result()

                for window_key in selected_window_keys:
                    _append_job_output(
                        job_outputs_by_window[window_key],
                        results_dict,
                        settings_tables,
                        consensus_tables,
                        run_summary_tables,
                    )
    finally:
        if executor is not None:
            executor.shutdown()

    return _finalize_grid_outputs(
        results_dict=results_dict,
        settings_tables=settings_tables,
        consensus_tables=consensus_tables,
        run_summary_tables=run_summary_tables,
        save_exports=save_exports,
        outdir=outdir,
        consensus_cfg=consensus_cfg,
        pelt_feature_cols=pelt_feature_cols,
    )


def PELT_grid_search(
    df,
    dfN,
    mips,
    input_windows,
    PELT_penalties,
    min_support_frac_runs_values,
    stop_rel_improvement_values,
    pelt_feature_cols=("width_s", "nch_s"),
    make_plots=True,
    print_timings=True,
    save_exports=True,
    show_progress=False,
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
    outdir=DEFAULT_OUTDIR,
    consensus_cfg=PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    centerlines=None,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    geometry_feature_cfg=None,
    check_centerline_orientation=True,
):
    return _PELT_grid_search_impl(
        df=df,
        dfN=dfN,
        mips=mips,
        input_windows=input_windows,
        PELT_penalties=PELT_penalties,
        min_support_frac_runs_values=min_support_frac_runs_values,
        stop_rel_improvement_values=stop_rel_improvement_values,
        pelt_feature_cols=pelt_feature_cols,
        make_plots=make_plots,
        print_timings=print_timings,
        save_exports=save_exports,
        show_progress=show_progress,
        swot_node_dir=swot_node_dir,
        swot_region=swot_region,
        outdir=outdir,
        consensus_cfg=consensus_cfg,
        centerlines=centerlines,
        centerline_id_col=centerline_id_col,
        centerline_geometry_col=centerline_geometry_col,
        node_geometry_col=node_geometry_col,
        geometry_feature_cfg=geometry_feature_cfg,
        check_centerline_orientation=check_centerline_orientation,
        parallel_windows=False,
    )


def PELT_grid_search_parallel(
    df,
    dfN,
    mips,
    input_windows,
    PELT_penalties,
    min_support_frac_runs_values,
    stop_rel_improvement_values,
    pelt_feature_cols=("width_s", "nch_s"),
    make_plots=True,
    print_timings=True,
    save_exports=True,
    show_progress=False,
    swot_node_dir=DEFAULT_SWOT_NODE_DIR,
    swot_region=DEFAULT_SWOT_REGION,
    outdir=DEFAULT_OUTDIR,
    consensus_cfg=PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    centerlines=None,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    geometry_feature_cfg=None,
    check_centerline_orientation=True,
    max_workers=None,
):
    return _PELT_grid_search_impl(
        df=df,
        dfN=dfN,
        mips=mips,
        input_windows=input_windows,
        PELT_penalties=PELT_penalties,
        min_support_frac_runs_values=min_support_frac_runs_values,
        stop_rel_improvement_values=stop_rel_improvement_values,
        pelt_feature_cols=pelt_feature_cols,
        make_plots=make_plots,
        print_timings=print_timings,
        save_exports=save_exports,
        show_progress=show_progress,
        swot_node_dir=swot_node_dir,
        swot_region=swot_region,
        outdir=outdir,
        consensus_cfg=consensus_cfg,
        centerlines=centerlines,
        centerline_id_col=centerline_id_col,
        centerline_geometry_col=centerline_geometry_col,
        node_geometry_col=node_geometry_col,
        geometry_feature_cfg=geometry_feature_cfg,
        check_centerline_orientation=check_centerline_orientation,
        parallel_windows=True,
        max_workers=max_workers,
    )



# ------------------------------------------------------------
# Helpers
# ------------------------------------------------------------
WINDOW_KEY_TO_VERSION = {0: "raw",2:'w2', 3: "w3", 4:'w4', 5: "w5"}


def infer_run_meta(run_key, result):
    run_key = str(run_key)

    reach_id = run_key
    window_version = None

    if "_" in run_key:
        left, right = run_key.rsplit("_", 1)
        reach_id = left
        if right.isdigit():
            wkey = int(right)
            window_version = WINDOW_KEY_TO_VERSION.get(wkey, f"w{wkey}")

    try:
        reach_id = int(reach_id)
    except Exception:
        pass

    if window_version is None:
        win_info = result.get("window_selection", {})
        method = str(win_info.get("method", "unknown"))
        n_windows = len(result.get("resolved_windows_m", ()))
        window_version = "raw" if method == "raw" else f"w{n_windows}"

    return reach_id, window_version


def segment_labels_from_breaks(dist_m, breaks_m):
    dist_m = np.asarray(dist_m, dtype=float)
    if breaks_m is None or len(breaks_m) == 0:
        return np.zeros(len(dist_m), dtype=int)
    breaks_m = np.sort(np.asarray(breaks_m, dtype=float))
    return np.searchsorted(breaks_m, dist_m, side="right")


def calinski_harabasz_manual(X, labels):
    X = np.asarray(X, dtype=float)
    labels = np.asarray(labels)

    if X.ndim != 2 or len(X) == 0:
        return np.nan

    unique_labels = np.unique(labels)
    n_samples = X.shape[0]
    n_clusters = len(unique_labels)

    if n_clusters < 2 or n_samples <= n_clusters:
        return np.nan

    overall_mean = X.mean(axis=0)
    between = 0.0
    within = 0.0

    for lab in unique_labels:
        Xi = X[labels == lab]
        ni = Xi.shape[0]
        if ni == 0:
            continue
        mean_i = Xi.mean(axis=0)
        between += ni * np.sum((mean_i - overall_mean) ** 2)
        within += np.sum((Xi - mean_i) ** 2)

    if within <= 0:
        return np.nan

    return (between / (n_clusters - 1)) / (within / (n_samples - n_clusters))


def compute_ch_for_setting(result, breaks_m):
    std_by_window = result["standardized_by_window"]
    dist_col = result["dist_col_used"]
    feature_cols = list(result["final_selection_feature_cols"])

    ch_rows = []
    for wlab, fdf in std_by_window.items():
        cols = [c for c in feature_cols if c in fdf.columns]
        if len(cols) == 0 or dist_col not in fdf.columns:
            continue

        dist_m = fdf[dist_col].to_numpy(dtype=float)
        labels = segment_labels_from_breaks(dist_m, breaks_m)

        valid = np.isfinite(fdf[cols].to_numpy(dtype=float)).all(axis=1)
        X = fdf.loc[valid, cols].to_numpy(dtype=float)
        y = labels[valid]

        ch = calinski_harabasz_manual(X, y)
        ch_rows.append({"window_label": wlab, "ch": ch})

    ch_df = pd.DataFrame(ch_rows)
    ch_mean = float(ch_df["ch"].mean()) if not ch_df.empty else np.nan
    return ch_mean, ch_df


def match_score(a, b, tol_m=5000.0):
    a = sorted(float(x) for x in a)
    b = sorted(float(x) for x in b)

    if len(a) == 0 and len(b) == 0:
        return 1.0

    used = set()
    matches = 0
    for x in a:
        candidates = [(abs(x - y), j) for j, y in enumerate(b) if j not in used and abs(x - y) <= tol_m]
        if candidates:
            _, j = min(candidates)
            used.add(j)
            matches += 1

    return (2.0 * matches) / max(len(a) + len(b), 1)


def nearest_cluster_span_stats(breaks_m, consensus_df, match_tol_m=5000.0):
    if consensus_df.empty or len(breaks_m) == 0:
        return {
            "matched_cluster_count": 0,
            "matched_core_count": 0,
            "matched_medium_count": 0,
            "matched_weak_count": 0,
            "mean_selected_span_km": 0.0,
            "max_selected_span_km": 0.0,
        }

    centers = consensus_df["cluster_center_m"].to_numpy(dtype=float)
    spans_km = consensus_df["cluster_span_m"].to_numpy(dtype=float) / 1000.0
    classes = consensus_df["support_class"].astype(str).to_numpy()

    matched_idx = []
    for b in breaks_m:
        j = int(np.argmin(np.abs(centers - float(b))))
        if abs(centers[j] - float(b)) <= match_tol_m:
            matched_idx.append(j)

    matched_idx = sorted(set(matched_idx))
    if len(matched_idx) == 0:
        return {
            "matched_cluster_count": 0,
            "matched_core_count": 0,
            "matched_medium_count": 0,
            "matched_weak_count": 0,
            "mean_selected_span_km": 0.0,
            "max_selected_span_km": 0.0,
        }

    matched_spans = spans_km[matched_idx]
    matched_classes = classes[matched_idx]

    return {
        "matched_cluster_count": len(matched_idx),
        "matched_core_count": int(np.sum(matched_classes == "core")),
        "matched_medium_count": int(np.sum(matched_classes == "medium")),
        "matched_weak_count": int(np.sum(matched_classes == "weak")),
        "mean_selected_span_km": float(np.mean(matched_spans)),
        "max_selected_span_km": float(np.max(matched_spans)),
    }


def add_plateau_instability(df):
    df = df.copy().sort_values(
        ["min_support_frac_runs", "min_support_frac_windows_effective", "stop_rel_improvement"]
    ).reset_index(drop=True)

    vals_runs = sorted(df["min_support_frac_runs"].unique())
    vals_wins = sorted(df["min_support_frac_windows_effective"].unique())
    vals_stop = sorted(df["stop_rel_improvement"].unique())

    lookup = {
        (r.min_support_frac_runs, r.min_support_frac_windows_effective, r.stop_rel_improvement): i
        for i, r in df.iterrows()
    }

    def neighbors(row):
        out = []
        for vals, col in [
            (vals_runs, "min_support_frac_runs"),
            (vals_wins, "min_support_frac_windows_effective"),
            (vals_stop, "stop_rel_improvement"),
        ]:
            cur = row[col]
            pos = vals.index(cur)
            for step in (-1, 1):
                j = pos + step
                if 0 <= j < len(vals):
                    key = (
                        row["min_support_frac_runs"] if col != "min_support_frac_runs" else vals[j],
                        row["min_support_frac_windows_effective"] if col != "min_support_frac_windows_effective" else vals[j],
                        row["stop_rel_improvement"] if col != "stop_rel_improvement" else vals[j],
                    )
                    if key in lookup:
                        out.append(lookup[key])
        return sorted(set(out))

    instabilities = []
    for _, row in df.iterrows():
        nei = neighbors(row)
        if not nei:
            instabilities.append(np.nan)
            continue

        dc = np.abs(df.loc[nei, "mean_centrality"] - row["mean_centrality"]).mean()
        ds = np.abs(df.loc[nei, "mean_selected_span_km"] - row["mean_selected_span_km"]).mean()
        dn = np.abs(df.loc[nei, "mean_n_breaks"] - row["mean_n_breaks"]).mean()
        de = np.abs(df.loc[nei, "mean_sse_rel"] - row["mean_sse_rel"]).mean()
        dh = np.abs(df.loc[nei, "mean_ch"] - row["mean_ch"]).mean()

        instabilities.append(float(dc + 0.20 * ds + 0.15 * dn + 0.15 * de + 0.10 * dh))

    df["plateau_instability"] = instabilities
    return df


def pct_rank(series, higher_is_better=True):
    series = series.astype(float)
    if higher_is_better:
        return series.rank(pct=True, ascending=True, method="average")
    return series.rank(pct=True, ascending=False, method="average")


def add_pareto_flag(df, maximize_cols, minimize_cols):
    df = df.copy().reset_index(drop=True)

    max_vals = df[maximize_cols].to_numpy(dtype=float) if maximize_cols else np.empty((len(df), 0))
    min_vals = df[minimize_cols].to_numpy(dtype=float) if minimize_cols else np.empty((len(df), 0))

    efficient = np.ones(len(df), dtype=bool)

    for i in range(len(df)):
        for j in range(len(df)):
            if i == j:
                continue

            better_or_equal = True
            strictly_better = False

            if maximize_cols:
                ge = np.all(max_vals[j] >= max_vals[i])
                gt = np.any(max_vals[j] > max_vals[i])
                better_or_equal &= ge
                strictly_better |= gt

            if minimize_cols:
                le = np.all(min_vals[j] <= min_vals[i])
                lt = np.any(min_vals[j] < min_vals[i])
                better_or_equal &= le
                strictly_better |= lt

            if better_or_equal and strictly_better:
                efficient[i] = False
                break

    df["is_pareto"] = efficient
    return df


def build_local_pareto_grid(family_df, rep_row, min_keep=4):
    family_df = family_df.copy()
    pareto_df = family_df[family_df["is_pareto"]].copy()

    vals_runs = sorted(family_df["min_support_frac_runs"].unique())
    vals_wins = sorted(family_df["min_support_frac_windows_effective"].unique())
    vals_stop = sorted(family_df["stop_rel_improvement"].unique())

    rep_run = float(rep_row["min_support_frac_runs"])
    rep_win = float(rep_row["min_support_frac_windows_effective"])
    rep_stop = float(rep_row["stop_rel_improvement"])

    def step_distance(row):
        return (
            abs(vals_runs.index(float(row["min_support_frac_runs"])) - vals_runs.index(rep_run)) +
            abs(vals_wins.index(float(row["min_support_frac_windows_effective"])) - vals_wins.index(rep_win)) +
            abs(vals_stop.index(float(row["stop_rel_improvement"])) - vals_stop.index(rep_stop))
        )

    pareto_df["step_distance"] = pareto_df.apply(step_distance, axis=1)

    local_df = pareto_df[pareto_df["step_distance"] <= 1].copy()

    if len(local_df) < min_keep:
        local_df = pareto_df.sort_values(
            ["step_distance", "mean_ch", "mean_centrality", "mean_plateau_instability", "mean_span_penalty_km", "mean_sse_rel"],
            ascending=[True, False, False, True, True, True],
        ).head(min_keep).copy()
    else:
        local_df = local_df.sort_values(
            ["step_distance", "mean_ch", "mean_centrality", "mean_plateau_instability", "mean_span_penalty_km", "mean_sse_rel"],
            ascending=[True, False, False, True, True, True],
        ).copy()

    return local_df.reset_index(drop=True)

def representative_rows_by_reach(settings_df, rep_row):
    mask = (
        (settings_df["window_version"] == rep_row["window_version"]) &
        np.isclose(settings_df["min_support_frac_runs"], float(rep_row["min_support_frac_runs"])) &
        np.isclose(
            settings_df["min_support_frac_windows_effective"],
            float(rep_row["min_support_frac_windows_effective"]),
        ) &
        np.isclose(settings_df["stop_rel_improvement"], float(rep_row["stop_rel_improvement"]))
    )
    cols = [
        "reach_id",
        "window_version",
        "ch_mean",
        "setting_centrality",
        "span_penalty_km",
        "sse_rel_to_best",
        "n_breaks",
        "run_key",
        "setting",
    ]
    return settings_df.loc[mask, cols].copy()


def choose_winning_family(
    settings_df,
    family_reps_df,
    mean_ch_rel_threshold=0.10,
    reach_ch_rel_tie=0.02,
    reach_win_frac_threshold=0.75,
    ):
    reps_ranked = family_reps_df.sort_values(
        ["mean_ch", "mean_centrality", "mean_plateau_instability", "mean_span_penalty_km", "mean_sse_rel"],
        ascending=[False, False, True, True, True],
    ).reset_index(drop=True)

    if len(reps_ranked) == 1:
        winner = reps_ranked.iloc[0]["window_version"]
        decision_df = pd.DataFrame([{
            "winner_family": winner,
            "runner_up_family": None,
            "decision_stage": "single_family",
            "mean_ch_rel_threshold": mean_ch_rel_threshold,
            "reach_ch_rel_tie": reach_ch_rel_tie,
            "reach_win_frac_threshold": reach_win_frac_threshold,
            "winner_mean_ch": reps_ranked.iloc[0]["mean_ch"],
            "runner_up_mean_ch": np.nan,
            "mean_ch_rel_diff": np.nan,
            "n_common_reaches": np.nan,
            "n_non_tied_reaches": np.nan,
            "winner_reach_win_frac": np.nan,
            "runner_up_reach_win_frac": np.nan,
            "n_reach_ties": np.nan,
        }])
        return winner, reps_ranked.iloc[0].copy(), decision_df, pd.DataFrame(), reps_ranked

    fam_a = reps_ranked.iloc[0].copy()
    fam_b = reps_ranked.iloc[1].copy()

    mean_ch_rel_diff = (float(fam_a["mean_ch"]) - float(fam_b["mean_ch"])) / max(abs(float(fam_b["mean_ch"])), 1e-12)

    reach_cmp = pd.DataFrame()
    decision_stage = None
    winner_family = None

    if mean_ch_rel_diff > mean_ch_rel_threshold:
        decision_stage = "mean_ch_gap"
        winner_family = fam_a["window_version"]
        winner_reach_win_frac = np.nan
        runner_up_reach_win_frac = np.nan
        n_common_reaches = np.nan
        n_both_valid_reaches = np.nan
        n_non_tied_reaches = np.nan
        n_reach_ties = np.nan
        n_auto_wins_a = np.nan
        n_auto_wins_b = np.nan
    else:
        a_rows = representative_rows_by_reach(settings_df, fam_a).rename(
            columns={
                "ch_mean": "ch_mean_a",
                "setting_centrality": "centrality_a",
                "span_penalty_km": "span_penalty_a",
                "sse_rel_to_best": "sse_rel_a",
                "n_breaks": "n_breaks_a",
                "run_key": "run_key_a",
                "setting": "setting_a",
            }
        )
        b_rows = representative_rows_by_reach(settings_df, fam_b).rename(
            columns={
                "ch_mean": "ch_mean_b",
                "setting_centrality": "centrality_b",
                "span_penalty_km": "span_penalty_b",
                "sse_rel_to_best": "sse_rel_b",
                "n_breaks": "n_breaks_b",
                "run_key": "run_key_b",
                "setting": "setting_b",
            }
        )

        reach_cmp = a_rows.merge(b_rows, on="reach_id", how="outer", indicator=True)
        reach_cmp["family_a"] = fam_a["window_version"]
        reach_cmp["family_b"] = fam_b["window_version"]
        both_valid = reach_cmp["_merge"] == "both"
        only_a = reach_cmp["_merge"] == "left_only"
        only_b = reach_cmp["_merge"] == "right_only"

        reach_cmp["ch_rel_diff_a_vs_b"] = np.nan
        reach_cmp.loc[both_valid, "ch_rel_diff_a_vs_b"] = (
            (reach_cmp.loc[both_valid, "ch_mean_a"] - reach_cmp.loc[both_valid, "ch_mean_b"]) /
            reach_cmp.loc[both_valid, "ch_mean_b"].abs().clip(lower=1e-12)
        )

        reach_cmp["reach_winner"] = "tie"
        reach_cmp.loc[only_a, "reach_winner"] = fam_a["window_version"]
        reach_cmp.loc[only_b, "reach_winner"] = fam_b["window_version"]
        reach_cmp.loc[
            both_valid & (reach_cmp["ch_rel_diff_a_vs_b"] > reach_ch_rel_tie),
            "reach_winner",
        ] = fam_a["window_version"]
        reach_cmp.loc[
            both_valid & (reach_cmp["ch_rel_diff_a_vs_b"] < -reach_ch_rel_tie),
            "reach_winner",
        ] = fam_b["window_version"]

        non_tied = reach_cmp[reach_cmp["reach_winner"] != "tie"].copy()
        n_common_reaches = int(len(reach_cmp))
        n_both_valid_reaches = int(both_valid.sum())
        n_non_tied_reaches = int(len(non_tied))
        n_reach_ties = int((reach_cmp["reach_winner"] == "tie").sum())
        n_auto_wins_a = int(only_a.sum())
        n_auto_wins_b = int(only_b.sum())

        if n_non_tied_reaches > 0:
            winner_reach_win_frac = float((non_tied["reach_winner"] == fam_a["window_version"]).mean())
            runner_up_reach_win_frac = float((non_tied["reach_winner"] == fam_b["window_version"]).mean())
        else:
            winner_reach_win_frac = np.nan
            runner_up_reach_win_frac = np.nan

        if n_non_tied_reaches > 0 and winner_reach_win_frac >= reach_win_frac_threshold:
            decision_stage = "paired_reach_ch"
            winner_family = fam_a["window_version"]
        elif n_non_tied_reaches > 0 and runner_up_reach_win_frac >= reach_win_frac_threshold:
            decision_stage = "paired_reach_ch"
            winner_family = fam_b["window_version"]
        else:
            decision_stage = "robustness_tiebreak"
            robustness_rank = pd.DataFrame([fam_a, fam_b]).sort_values(
                ["mean_centrality", "mean_plateau_instability", "mean_span_penalty_km", "mean_sse_rel"],
                ascending=[False, True, True, True],
            ).reset_index(drop=True)
            winner_family = robustness_rank.iloc[0]["window_version"]

    runner_up_family = fam_b["window_version"] if winner_family == fam_a["window_version"] else fam_a["window_version"]

    decision_df = pd.DataFrame([{
        "winner_family": winner_family,
        "runner_up_family": runner_up_family,
        "decision_stage": decision_stage,
        "mean_ch_rel_threshold": mean_ch_rel_threshold,
        "reach_ch_rel_tie": reach_ch_rel_tie,
        "reach_win_frac_threshold": reach_win_frac_threshold,
        "winner_mean_ch": float(fam_a["mean_ch"]) if winner_family == fam_a["window_version"] else float(fam_b["mean_ch"]),
        "runner_up_mean_ch": float(fam_b["mean_ch"]) if winner_family == fam_a["window_version"] else float(fam_a["mean_ch"]),
        "mean_ch_rel_diff": float(mean_ch_rel_diff),
        "n_common_reaches": n_common_reaches,
        "n_both_valid_reaches": n_both_valid_reaches,
        "n_non_tied_reaches": n_non_tied_reaches,
        "winner_reach_win_frac": winner_reach_win_frac,
        "runner_up_reach_win_frac": runner_up_reach_win_frac,
        "n_reach_ties": n_reach_ties,
        "n_auto_wins_winner": n_auto_wins_a if winner_family == fam_a["window_version"] else n_auto_wins_b,
        "n_auto_wins_runner_up": n_auto_wins_b if winner_family == fam_a["window_version"] else n_auto_wins_a,
    }])

    winning_rep = family_reps_df[family_reps_df["window_version"] == winner_family].iloc[0].copy()
    return winner_family, winning_rep, decision_df, reach_cmp, reps_ranked


def load_results_dict(results_path=None, outdir=DEFAULT_OUTDIR):
    if results_path is None:
        results_path = Path(outdir) / DEFAULT_RESULTS_PICKLE
    with open(results_path, "rb") as f:
        return pickle.load(f)


def run_tuning_analysis(
    results_dict=None,
    results_path=None,
    outdir=DEFAULT_OUTDIR,
    mean_ch_rel_threshold=0.10,
    reach_ch_rel_tie=0.02,
    reach_win_frac_threshold=0.75,
    threshold_grid=(0.50, 0.625, 0.75, 0.875),
    print_outputs=True,
):
    if results_dict is None:
        results_dict = load_results_dict(results_path=results_path, outdir=outdir)

    settings_tables = []
    consensus_tables = []
    run_summary_tables = []

    for run_key, result in results_dict.items():
        reach_id, window_version = infer_run_meta(run_key, result)

        settings_df, consensus_df, run_summary_df = PELT.extract_pelt_grid_analysis_tables(
            result,
            reach_id=reach_id,
            window_version=window_version,
            medium_min=0.50,
        )

        settings_df["run_key"] = run_key
        consensus_df["run_key"] = run_key
        run_summary_df["run_key"] = run_key

        settings_tables.append(settings_df)
        consensus_tables.append(consensus_df)
        run_summary_tables.append(run_summary_df)

    settings = pd.concat(settings_tables, ignore_index=True)
    consensus = pd.concat(consensus_tables, ignore_index=True)
    runs = pd.concat(run_summary_tables, ignore_index=True)
    settings_all = settings.copy()

    runs["break_count_range"] = runs["n_breaks_max"] - runs["n_breaks_min"]
    runs["candidate_range"] = runs["n_candidates_max"] - runs["n_candidates_min"]
    runs["max_cluster_span_km"] = runs["max_cluster_span_m"] / 1000.0
    runs["mean_core_span_km"] = runs["mean_core_span_m"] / 1000.0

    if "status" not in settings.columns:
        settings["status"] = "ok"
    settings = settings[settings["status"].astype(str) == "ok"].copy().reset_index(drop=True)
    if settings.empty:
        raise ValueError("No valid selector-grid settings available for tuning analysis.")

    settings["sse_rel_to_best"] = settings.groupby(
        ["reach_id", "window_version"]
    )["final_sse"].transform(lambda s: s / s.min())

    centrality_rows = []
    for (reach_id, window_version), g in settings.groupby(["reach_id", "window_version"]):
        idx = g.index.tolist()
        break_map = {i: g.loc[i, "breaks_m"] for i in idx}
        for i in idx:
            others = [j for j in idx if j != i]
            if not others:
                centrality = 1.0
            else:
                scores = [match_score(break_map[i], break_map[j], tol_m=5000.0) for j in others]
                centrality = float(np.mean(scores))
            centrality_rows.append({"_idx": i, "setting_centrality": centrality})

    centrality_df = pd.DataFrame(centrality_rows)
    settings = settings.merge(centrality_df, left_index=True, right_on="_idx").drop(columns="_idx")

    ch_rows = []
    for _, row in settings.iterrows():
        result = results_dict[row["run_key"]]
        setting_name = row["setting"]
        sel = result["final_selection_grid"].individual_results[setting_name]
        ch_mean, _ = compute_ch_for_setting(result, sel.breaks_m)

        ch_rows.append(
            {
                "run_key": row["run_key"],
                "setting": setting_name,
                "ch_mean": ch_mean,
            }
        )

    ch_df = pd.DataFrame(ch_rows)
    settings = settings.merge(ch_df, on=["run_key", "setting"], how="left")

    settings["ch_rel_to_best"] = settings.groupby(
        ["reach_id", "window_version"]
    )["ch_mean"].transform(lambda s: s / s.max() if np.isfinite(s).any() else np.nan)

    span_rows = []
    for _, row in settings.iterrows():
        cons_sub = consensus[(consensus["run_key"] == row["run_key"])].copy()

        span_stats = nearest_cluster_span_stats(
            row["breaks_m"],
            cons_sub,
            match_tol_m=5000.0,
        )

        span_rows.append(
            {
                "run_key": row["run_key"],
                "setting": row["setting"],
                **span_stats,
            }
        )

    span_df = pd.DataFrame(span_rows)
    settings = settings.merge(span_df, on=["run_key", "setting"], how="left")

    settings["span_penalty_km"] = (
        0.7 * settings["max_selected_span_km"].fillna(0.0)
        + 0.3 * settings["mean_selected_span_km"].fillna(0.0)
    )

    setting_stats = (
        settings.groupby(
            ["window_version", "min_support_frac_runs", "min_support_frac_windows_effective", "stop_rel_improvement"]
        )
        .agg(
            reaches=("reach_id", "nunique"),
            mean_n_breaks=("n_breaks", "mean"),
            sd_n_breaks=("n_breaks", "std"),
            mean_centrality=("setting_centrality", "mean"),
            sd_centrality=("setting_centrality", "std"),
            mean_sse_rel=("sse_rel_to_best", "mean"),
            mean_ch=("ch_mean", "mean"),
            mean_ch_rel=("ch_rel_to_best", "mean"),
            mean_selected_span_km=("mean_selected_span_km", "mean"),
            max_selected_span_km=("max_selected_span_km", "mean"),
            mean_span_penalty_km=("span_penalty_km", "mean"),
        )
        .reset_index()
    )

    setting_stats = pd.concat(
        [add_plateau_instability(g) for _, g in setting_stats.groupby("window_version", sort=False)],
        ignore_index=True,
    )

    family_setting_summary = pd.concat(
        [
            add_pareto_flag(
                g,
                maximize_cols=["mean_ch", "mean_centrality"],
                minimize_cols=["plateau_instability", "mean_span_penalty_km", "mean_sse_rel"],
            )
            for _, g in setting_stats.groupby("window_version", sort=False)
        ],
        ignore_index=True,
    )

    family_setting_summary = family_setting_summary.rename(
        columns={"plateau_instability": "mean_plateau_instability"}
    )

    family_representatives = (
        family_setting_summary[family_setting_summary["is_pareto"]]
        .sort_values(
            ["window_version", "mean_ch", "mean_centrality", "mean_plateau_instability", "mean_span_penalty_km", "mean_sse_rel"],
            ascending=[True, False, False, True, True, True],
        )
        .groupby("window_version", group_keys=False)
        .head(1)
        .reset_index(drop=True)
    )

    family_overview = (
        family_setting_summary.groupby("window_version")
        .agg(
            reaches=("reaches", "max"),
            n_settings=("window_version", "size"),
            n_pareto=("is_pareto", "sum"),
            mean_ch_all=("mean_ch", "mean"),
            mean_centrality_all=("mean_centrality", "mean"),
            mean_span_penalty_km_all=("mean_span_penalty_km", "mean"),
            mean_plateau_instability_all=("mean_plateau_instability", "mean"),
        )
        .join(
            family_representatives.set_index("window_version")[
                [
                    "min_support_frac_runs",
                    "min_support_frac_windows_effective",
                    "stop_rel_improvement",
                    "mean_ch",
                    "mean_centrality",
                    "mean_span_penalty_km",
                    "mean_plateau_instability",
                    "mean_sse_rel",
                    "mean_n_breaks",
                ]
            ].rename(
                columns={
                    "min_support_frac_runs": "rep_min_support_frac_runs",
                    "min_support_frac_windows_effective": "rep_min_support_frac_windows_effective",
                    "stop_rel_improvement": "rep_stop_rel_improvement",
                    "mean_ch": "rep_mean_ch",
                    "mean_centrality": "rep_mean_centrality",
                    "mean_span_penalty_km": "rep_mean_span_penalty_km",
                    "mean_plateau_instability": "rep_mean_plateau_instability",
                    "mean_sse_rel": "rep_mean_sse_rel",
                    "mean_n_breaks": "rep_mean_n_breaks",
                }
            ),
            how="left",
        )
        .reset_index()
    )

    winning_family, winning_representative, family_decision_df, reach_family_ch_comparison_df, family_rank = choose_winning_family(
        settings_df=settings,
        family_reps_df=family_representatives,
        mean_ch_rel_threshold=mean_ch_rel_threshold,
        reach_ch_rel_tie=reach_ch_rel_tie,
        reach_win_frac_threshold=reach_win_frac_threshold,
    )

    winning_family_front = family_setting_summary[
        family_setting_summary["window_version"] == winning_family
    ].copy()

    recommended_grid_settings = build_local_pareto_grid(
        winning_family_front,
        winning_representative,
        min_keep=4,
    )

    recommended_grid = pd.DataFrame(
        [
            {
                "window_version": winning_family,
                "anchor_min_support_frac_runs": float(winning_representative["min_support_frac_runs"]),
                "anchor_min_support_frac_windows_effective": float(winning_representative["min_support_frac_windows_effective"]),
                "anchor_stop_rel_improvement": float(winning_representative["stop_rel_improvement"]),
                "grid_min_support_frac_runs_values": tuple(sorted(recommended_grid_settings["min_support_frac_runs"].unique())),
                "grid_min_support_frac_windows_effective_values": tuple(sorted(recommended_grid_settings["min_support_frac_windows_effective"].unique())),
                "grid_stop_rel_improvement_values": tuple(sorted(recommended_grid_settings["stop_rel_improvement"].unique())),
                "n_grid_settings": int(len(recommended_grid_settings)),
            }
        ]
    )

    threshold_rows = []
    for run_key, result in results_dict.items():
        reach_id, window_version = infer_run_meta(run_key, result)
        cons = result["final_selection_grid"].consensus.copy()

        if cons.empty:
            for thr in threshold_grid:
                threshold_rows.append(
                    {
                        "reach_id": reach_id,
                        "window_version": window_version,
                        "run_key": run_key,
                        "core_threshold": thr,
                        "n_core_breaks": 0,
                        "mean_core_span_km": np.nan,
                        "max_core_span_km": np.nan,
                        "mean_core_support": np.nan,
                    }
                )
            continue

        cons["cluster_span_km"] = cons["cluster_span_m"] / 1000.0

        for thr in threshold_grid:
            sub = cons[cons["support_frac_grid"] >= thr].copy()
            threshold_rows.append(
                {
                    "reach_id": reach_id,
                    "window_version": window_version,
                    "run_key": run_key,
                    "core_threshold": thr,
                    "n_core_breaks": len(sub),
                    "mean_core_span_km": float(sub["cluster_span_km"].mean()) if len(sub) else np.nan,
                    "max_core_span_km": float(sub["cluster_span_km"].max()) if len(sub) else np.nan,
                    "mean_core_support": float(sub["support_frac_grid"].mean()) if len(sub) else np.nan,
                }
            )

    threshold_df = pd.DataFrame(threshold_rows)

    threshold_summary = (
        threshold_df.groupby(["window_version", "core_threshold"])
        .agg(
            reaches=("reach_id", "nunique"),
            mean_n_core_breaks=("n_core_breaks", "mean"),
            mean_core_span_km=("mean_core_span_km", "mean"),
            mean_max_core_span_km=("max_core_span_km", "mean"),
            mean_core_support=("mean_core_support", "mean"),
        )
        .reset_index()
    )

    threshold_pareto = pd.concat(
        [
            add_pareto_flag(
                g,
                maximize_cols=["mean_n_core_breaks", "mean_core_support"],
                minimize_cols=["mean_max_core_span_km"],
            )
            for _, g in threshold_summary.groupby("window_version", sort=False)
        ],
        ignore_index=True,
    )

    winning_threshold_candidates = (
        threshold_pareto[
            (threshold_pareto["window_version"] == winning_family)
            & (threshold_pareto["is_pareto"])
        ]
        .sort_values(
            ["mean_core_support", "mean_max_core_span_km", "mean_n_core_breaks"],
            ascending=[False, True, False],
        )
        .reset_index(drop=True)
    )

    if print_outputs:
        print("\nFamily overview")
        print(family_overview)

        print("\nFamily representatives")
        print(family_rank)

        print("\nFamily decision")
        print(family_decision_df)

        if not reach_family_ch_comparison_df.empty:
            print("\nReach-level CH comparison for top two families")
            print(
                reach_family_ch_comparison_df[
                    [
                        "reach_id",
                        "family_a",
                        "family_b",
                        "ch_mean_a",
                        "ch_mean_b",
                        "ch_rel_diff_a_vs_b",
                        "reach_winner",
                    ]
                ].sort_values("reach_id")
            )

        print("\nRecommended winning family")
        print(pd.DataFrame([winning_representative]))

        print("\nRecommended local Pareto grid")
        print(recommended_grid)

        print("\nSettings inside recommended local Pareto grid")
        print(recommended_grid_settings)

        print("\nThreshold summary")
        print(threshold_summary)

        print(f"\nPareto threshold candidates for winning family: {winning_family}")
        print(winning_threshold_candidates)

    analysis_tables = {
        "settings_eval_df": settings,
        "settings_all_df": settings_all,
        "consensus_eval_df": consensus,
        "runs_eval_df": runs,
        "setting_stats_df": setting_stats,
        "family_setting_summary_df": family_setting_summary,
        "family_overview_df": family_overview,
        "family_rank_df": family_rank,
        "family_decision_df": family_decision_df,
        "reach_family_ch_comparison_df": reach_family_ch_comparison_df,
        "winning_representative_df": pd.DataFrame([winning_representative]),
        "recommended_grid_df": recommended_grid,
        "recommended_grid_settings_df": recommended_grid_settings,
        "threshold_summary_df": threshold_summary,
        "threshold_pareto_df": threshold_pareto,
        "winning_threshold_candidates_df": winning_threshold_candidates,
    }
    return analysis_tables


def main():
    try:
        run_tuning_analysis(print_outputs=True)
    except FileNotFoundError as exc:
        raise SystemExit(
            f"Could not find saved tuning results at "
            f"{Path(DEFAULT_OUTDIR) / DEFAULT_RESULTS_PICKLE}. "
            f"Run PELT_grid_search(...) first or pass results_dict/results_path to run_tuning_analysis()."
        ) from exc


if __name__ == "__main__":
    main()
