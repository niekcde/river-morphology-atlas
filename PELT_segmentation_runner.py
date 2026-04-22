from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple
import pickle

import numpy as np
import pandas as pd

import PELT
import PELT_finalize as pf
import PELT_geometry_features as pgf
import PELT_tuning as pt
import reach_concatenation as rc

try:
    from shapely.geometry import LineString
    from shapely.ops import substring
except Exception:  # pragma: no cover - import-time guard for environments without shapely
    LineString = None
    substring = None


DEFAULT_PELT_PENALTIES = (2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
DEFAULT_MIN_SUPPORT_FRAC_RUNS_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25)
DEFAULT_STOP_REL_IMPROVEMENT_VALUES = (0.04, 0.045, 0.05, 0.06)
WIDTH_CHANNEL_WINDOWS = (0, 2, 3, 4, 5)
GEOMETRY_WINDOWS = (2, 3, 4, 5)


@dataclass(frozen=True)
class SegmentationStageConfig:
    name: str
    pelt_feature_cols: Tuple[str, ...]
    input_windows: Tuple[int, ...]


@dataclass(frozen=True)
class SegmentationSetupConfig:
    name: str
    stages: Tuple[SegmentationStageConfig, ...]


SEGMENTATION_SETUPS: Tuple[SegmentationSetupConfig, ...] = (
    SegmentationSetupConfig(
        name="01_width_channels",
        stages=(
            SegmentationStageConfig(
                name="width_channels",
                pelt_feature_cols=("width_s", "nch_s"),
                input_windows=WIDTH_CHANNEL_WINDOWS,
            ),
        ),
    ),
    SegmentationSetupConfig(
        name="02_sinuosity_curvature",
        stages=(
            SegmentationStageConfig(
                name="sinuosity_curvature",
                pelt_feature_cols=("sinu", "curv_int"),
                input_windows=GEOMETRY_WINDOWS,
            ),
        ),
    ),
    SegmentationSetupConfig(
        name="03_all_features",
        stages=(
            SegmentationStageConfig(
                name="all_features",
                pelt_feature_cols=("width_s", "nch_s", "sinu", "curv_int"),
                input_windows=GEOMETRY_WINDOWS,
            ),
        ),
    ),
    SegmentationSetupConfig(
        name="04_width_channels_then_geometry",
        stages=(
            SegmentationStageConfig(
                name="stage1_width_channels",
                pelt_feature_cols=("width_s", "nch_s"),
                input_windows=WIDTH_CHANNEL_WINDOWS,
            ),
            SegmentationStageConfig(
                name="stage2_sinuosity_curvature",
                pelt_feature_cols=("sinu", "curv_int"),
                input_windows=GEOMETRY_WINDOWS,
            ),
        ),
    ),
    SegmentationSetupConfig(
        name="05_geometry_then_width_channels",
        stages=(
            SegmentationStageConfig(
                name="stage1_sinuosity_curvature",
                pelt_feature_cols=("sinu", "curv_int"),
                input_windows=GEOMETRY_WINDOWS,
            ),
            SegmentationStageConfig(
                name="stage2_width_channels",
                pelt_feature_cols=("width_s", "nch_s"),
                input_windows=WIDTH_CHANNEL_WINDOWS,
            ),
        ),
    ),
)


def feature_cols_need_geometry(feature_cols: Sequence[str]) -> bool:
    return bool(set(PELT.normalize_feature_cols(feature_cols)) & {"sinu", "curv_int"})


def validate_stage_config(stage: SegmentationStageConfig) -> None:
    if feature_cols_need_geometry(stage.pelt_feature_cols) and 0 in set(stage.input_windows):
        raise ValueError(
            f"{stage.name}: raw window 0 cannot be used with geometry features "
            "('sinu', 'curv_int')."
        )


def build_centerlines_from_edges(
    df_edges,
    out_path: Optional[str | Path] = None,
    assert_all_linestring: bool = True,
    **merge_kwargs,
):
    """
    Build one centerline per main_path_id using reach_concatenation.merge_mainpaths.
    """
    centerlines = rc.merge_mainpaths(
        df_edges,
        return_diagnostics=True,
        **merge_kwargs,
    )
    qa_df = pgf.summarize_centerline_geometries(
        centerlines,
        id_col="main_path_id",
        geometry_col="line",
        assert_all_linestring=False,
    )
    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        qa_df.to_csv(out_path.with_suffix(".geometry_qa.csv"), index=False)
        try:
            import geopandas as gpd

            gdf = gpd.GeoDataFrame(centerlines, geometry="line", crs=getattr(df_edges, "crs", None))
            gdf.to_parquet(out_path)
        except Exception:
            centerlines.to_pickle(out_path)
    if assert_all_linestring:
        bad = qa_df[~qa_df["is_linestring"] | qa_df["is_empty"]].copy()
        if not bad.empty:
            sample = bad["main_path_id"].head(10).tolist()
            raise ValueError(
                f"Expected all merged centerlines to be non-empty LineStrings; "
                f"found {len(bad)} invalid geometries. Sample main_path_id: {sample}"
            )
    return centerlines, qa_df


def _target_families_from_input_windows(input_windows: Sequence[int]) -> Tuple[str, ...]:
    mapping = {2: "w2", 3: "w3", 4: "w4", 5: "w5"}
    return tuple(mapping[w] for w in input_windows if w in mapping)


def run_single_stage_setup(
    stage: SegmentationStageConfig,
    df,
    dfN,
    mips,
    outdir: str | Path,
    centerlines=None,
    penalties=DEFAULT_PELT_PENALTIES,
    min_support_frac_runs_values=DEFAULT_MIN_SUPPORT_FRAC_RUNS_VALUES,
    stop_rel_improvement_values=DEFAULT_STOP_REL_IMPROVEMENT_VALUES,
    swot_node_dir=pt.DEFAULT_SWOT_NODE_DIR,
    swot_region=pt.DEFAULT_SWOT_REGION,
    centerline_id_col="main_path_id",
    centerline_geometry_col="line",
    node_geometry_col="geometry",
    geometry_feature_cfg=None,
    check_centerline_orientation=True,
    calibrate_consensus: bool = True,
    show_progress: bool = True,
    parallel_tuning: bool = False,
    max_workers=None,
):
    validate_stage_config(stage)
    outdir = Path(outdir)
    tuning_outdir = outdir / "tuning"
    final_outdir = outdir / "final"
    tuning_outdir.mkdir(parents=True, exist_ok=True)
    final_outdir.mkdir(parents=True, exist_ok=True)

    grid_search = pt.PELT_grid_search_parallel if parallel_tuning else pt.PELT_grid_search
    grid_kwargs = {}
    if parallel_tuning:
        grid_kwargs["max_workers"] = max_workers

    grid_outputs = grid_search(
        df=df,
        dfN=dfN,
        mips=mips,
        input_windows=stage.input_windows,
        PELT_penalties=penalties,
        min_support_frac_runs_values=min_support_frac_runs_values,
        stop_rel_improvement_values=stop_rel_improvement_values,
        pelt_feature_cols=stage.pelt_feature_cols,
        consensus_cfg=PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
        centerlines=centerlines,
        centerline_id_col=centerline_id_col,
        centerline_geometry_col=centerline_geometry_col,
        node_geometry_col=node_geometry_col,
        geometry_feature_cfg=geometry_feature_cfg,
        check_centerline_orientation=check_centerline_orientation,
        make_plots=False,
        print_timings=False,
        save_exports=True,
        show_progress=show_progress,
        outdir=tuning_outdir,
        swot_node_dir=swot_node_dir,
        swot_region=swot_region,
        **grid_kwargs,
    )

    if calibrate_consensus:
        target_families = _target_families_from_input_windows(stage.input_windows)
        grid_outputs = pt.calibrate_and_reapply_consensus_to_grid_outputs(
            grid_outputs=grid_outputs,
            save_exports=True,
            outdir=tuning_outdir,
            calibration_outdir=tuning_outdir,
            target_families=target_families,
        )

    analysis_tables = pt.run_tuning_analysis(
        results_dict=grid_outputs["results_dict"],
        print_outputs=True,
    )
    with open(tuning_outdir / "PELT_analysis_tables.pkl", "wb") as f:
        pickle.dump(analysis_tables, f, protocol=pickle.HIGHEST_PROTOCOL)

    final_inputs = pf.derive_finalization_inputs_from_analysis_tables(
        analysis_tables,
        consensus_cfg=grid_outputs["consensus_cfg"],
    )

    final_outputs = pf.run_final_batch(
        df=df,
        dfN=dfN,
        mips=mips,
        window_key=final_inputs["window_key"],
        penalties=penalties,
        pelt_feature_cols=stage.pelt_feature_cols,
        selector_settings=final_inputs["selector_settings"],
        stable_support_count=final_inputs["stable_support_count"],
        consensus_cfg=final_inputs["consensus_cfg"],
        swot_node_dir=swot_node_dir,
        swot_region=swot_region,
        centerlines=centerlines,
        centerline_id_col=centerline_id_col,
        centerline_geometry_col=centerline_geometry_col,
        node_geometry_col=node_geometry_col,
        geometry_feature_cfg=geometry_feature_cfg,
        check_centerline_orientation=check_centerline_orientation,
        make_plots=False,
        save_exports=True,
        show_progress=show_progress,
        print_timings=False,
        outdir=final_outdir,
    )

    final_summary = pf.summarize_final_batch(final_outputs)
    with open(final_outdir / "PELT_final_summary.pkl", "wb") as f:
        pickle.dump(final_summary, f, protocol=pickle.HIGHEST_PROTOCOL)
    final_summary["reach_summary_df"].to_csv(final_outdir / "PELT_final_reach_summary.csv", index=False)
    final_summary["overall_summary_df"].to_csv(final_outdir / "PELT_final_overall_summary.csv", index=False)

    return {
        "stage": stage,
        "grid_outputs": grid_outputs,
        "analysis_tables": analysis_tables,
        "final_inputs": final_inputs,
        "final_outputs": final_outputs,
        "final_summary": final_summary,
    }


def _find_result_for_mip(final_outputs, mip):
    for run_key, result in final_outputs["results_dict"].items():
        if pf._infer_reach_id_from_run_key(run_key) == mip:
            return run_key, result
    raise KeyError(f"No stage-1 result found for mip {mip}.")


def _combined_segments_from_breaks(dist: np.ndarray, breaks_m: Sequence[float]):
    idx = np.searchsorted(dist, np.array(sorted(breaks_m), dtype=float), side="left")
    idx = [int(i) for i in idx if 0 < i < len(dist)]
    idx = sorted(set(idx))
    cuts = [0] + idx + [len(dist)]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def _window_run_config(window_key: int):
    window_runs = pf._build_window_runs()
    if window_key not in window_runs:
        raise ValueError(f"Unknown window_key {window_key}. Valid options are {list(window_runs)}.")
    return window_runs[window_key]


def _substring_centerline(centerline, start_m: float, end_m: float):
    if substring is None or LineString is None:
        raise ImportError("shapely is required for two-stage geometry segmentation.")
    lo = max(0.0, float(start_m))
    hi = min(float(end_m), float(centerline.length))
    if hi <= lo:
        return None
    out = substring(centerline, lo, hi)
    if not isinstance(out, LineString) or out.is_empty or float(out.length) <= 0.0:
        return None
    return out


def _run_stage_on_prepared_nodes(
    nodes_df,
    pelt_feature_cols,
    window_key,
    penalties,
    selector_settings,
    consensus_cfg,
    stable_support_count=None,
    centerline=None,
    geometry_feature_cfg=None,
):
    cfg = _window_run_config(window_key)
    base_result = PELT.run_full_pipeline(
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
            print_timings=False,
        ),
        centerline=centerline,
        geometry_feature_cfg=geometry_feature_cfg,
    )
    return pf.apply_explicit_selector_grid(
        results=base_result,
        selector_settings=selector_settings,
        feature_cols=tuple(pelt_feature_cols),
        stable_support_count=stable_support_count,
        consensus_cfg=consensus_cfg,
        attach_to_results=True,
    )


def run_two_stage_final_batch(
    stage1_outputs,
    stage2: SegmentationStageConfig,
    stage2_final_inputs,
    mips,
    outdir: str | Path,
    centerlines=None,
    penalties=DEFAULT_PELT_PENALTIES,
    geometry_feature_cfg=None,
    min_segment_nodes: int = 10,
    centerline_id_col: str = "main_path_id",
    centerline_geometry_col: str = "line",
    node_geometry_col: str = "geometry",
):
    validate_stage_config(stage2)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results_dict = {}
    summary_rows = []
    stage2_segment_rows = []
    need_geometry = feature_cols_need_geometry(stage2.pelt_feature_cols)

    for mip in mips:
        stage1_run_key, stage1_result = _find_result_for_mip(stage1_outputs, mip)
        nodes = stage1_result["nodes_used"].sort_values("dist_m").reset_index(drop=True)
        global_dist = nodes["dist_m"].to_numpy(dtype=float)
        stage1_breaks = [float(v) for v in stage1_result.get("stable_breaks_m", [])]
        stage1_segments = list(stage1_result.get("stable_segments", [(0, len(nodes))]))

        oriented_centerline = None
        if need_geometry:
            raw_centerline = pf._lookup_centerline(
                centerlines,
                mip,
                centerline_id_col=centerline_id_col,
                centerline_geometry_col=centerline_geometry_col,
            )
            oriented_centerline, _ = pgf.orient_centerline_to_node_dist(
                raw_centerline,
                nodes,
                dist_col="dist_m",
                node_geometry_col=node_geometry_col,
                reverse_if_needed=True,
            )

        stage2_breaks_global = []
        segment_results = []
        for segment_id, (a, b) in enumerate(stage1_segments):
            a = int(a)
            b = int(b)
            if b - a < int(min_segment_nodes):
                stage2_segment_rows.append(
                    {
                        "mip": mip,
                        "stage1_run_key": stage1_run_key,
                        "segment_id": segment_id,
                        "status": "skipped_too_few_nodes",
                        "n_nodes": int(b - a),
                        "n_stage2_breaks": 0,
                    }
                )
                continue

            sub_nodes = nodes.iloc[a:b].copy().reset_index(drop=True)
            offset_m = float(sub_nodes["dist_m"].iloc[0])
            end_m = float(sub_nodes["dist_m"].iloc[-1])
            sub_nodes["dist_m_global"] = sub_nodes["dist_m"].astype(float)
            sub_nodes["dist_m"] = sub_nodes["dist_m"].astype(float) - offset_m

            sub_centerline = None
            if need_geometry:
                sub_centerline = _substring_centerline(oriented_centerline, offset_m, end_m)
                if sub_centerline is None:
                    stage2_segment_rows.append(
                        {
                            "mip": mip,
                            "stage1_run_key": stage1_run_key,
                            "segment_id": segment_id,
                            "status": "skipped_invalid_centerline_substring",
                            "n_nodes": int(b - a),
                            "n_stage2_breaks": 0,
                        }
                    )
                    continue

            stage2_result = _run_stage_on_prepared_nodes(
                nodes_df=sub_nodes,
                pelt_feature_cols=stage2.pelt_feature_cols,
                window_key=stage2_final_inputs["window_key"],
                penalties=penalties,
                selector_settings=stage2_final_inputs["selector_settings"],
                stable_support_count=stage2_final_inputs["stable_support_count"],
                consensus_cfg=stage2_final_inputs["consensus_cfg"],
                centerline=sub_centerline,
                geometry_feature_cfg=geometry_feature_cfg,
            )
            local_breaks = [float(v) for v in stage2_result.get("stable_breaks_m", [])]
            global_breaks = [offset_m + v for v in local_breaks]
            stage2_breaks_global.extend(global_breaks)
            segment_results.append(
                {
                    "segment_id": segment_id,
                    "node_slice": (a, b),
                    "offset_m": offset_m,
                    "end_m": end_m,
                    "local_result": stage2_result,
                    "local_breaks_m": local_breaks,
                    "global_breaks_m": global_breaks,
                }
            )
            stage2_segment_rows.append(
                {
                    "mip": mip,
                    "stage1_run_key": stage1_run_key,
                    "segment_id": segment_id,
                    "status": "ok",
                    "n_nodes": int(b - a),
                    "n_stage2_breaks": int(len(global_breaks)),
                    "stage2_breaks_m": global_breaks,
                }
            )

        combined_breaks = sorted(
            set(round(v, 6) for v in stage1_breaks + stage2_breaks_global)
        )
        combined_segments = _combined_segments_from_breaks(global_dist, combined_breaks)
        run_key = f"{mip}_two_stage"
        results_dict[run_key] = {
            "stage1_run_key": stage1_run_key,
            "stage1_stable_breaks_m": stage1_breaks,
            "stage2_stable_breaks_m": sorted(stage2_breaks_global),
            "stable_breaks_m": combined_breaks,
            "stable_segments": combined_segments,
            "stage2_segment_results": segment_results,
            "nodes_used": nodes,
        }
        summary_rows.append(
            {
                "run_key": run_key,
                "mip": mip,
                "n_stage1_breaks": int(len(stage1_breaks)),
                "n_stage2_breaks": int(len(stage2_breaks_global)),
                "n_stable_breaks": int(len(combined_breaks)),
                "stable_breaks_m": combined_breaks,
                "stable_breaks_km": [v / 1000.0 for v in combined_breaks],
            }
        )

    reach_summary_df = pd.DataFrame(summary_rows).sort_values("mip").reset_index(drop=True)
    segment_summary_df = pd.DataFrame(stage2_segment_rows)
    with open(outdir / "PELT_two_stage_results_dict.pkl", "wb") as f:
        pickle.dump(results_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
    reach_summary_df.to_csv(outdir / "PELT_two_stage_reach_summary.csv", index=False)
    segment_summary_df.to_csv(outdir / "PELT_two_stage_segment_summary.csv", index=False)

    return {
        "results_dict": results_dict,
        "reach_summary_df": reach_summary_df,
        "segment_summary_df": segment_summary_df,
        "stage2_feature_cols": tuple(stage2.pelt_feature_cols),
        "stage2_window_key": stage2_final_inputs["window_key"],
    }


def run_segmentation_setup(
    setup: SegmentationSetupConfig,
    df,
    dfN,
    mips,
    outdir: str | Path,
    centerlines=None,
    **kwargs,
):
    if len(setup.stages) not in (1, 2):
        raise ValueError("Only one-stage and two-stage setups are supported.")

    setup_outdir = Path(outdir) / setup.name
    setup_outdir.mkdir(parents=True, exist_ok=True)
    single_stage_kwargs = dict(kwargs)
    min_segment_nodes = int(single_stage_kwargs.pop("min_segment_nodes", 10))

    stage_outputs = []
    for stage in setup.stages:
        stage_outputs.append(
            run_single_stage_setup(
                stage=stage,
                df=df,
                dfN=dfN,
                mips=mips,
                outdir=setup_outdir / stage.name,
                centerlines=centerlines,
                **single_stage_kwargs,
            )
        )

    two_stage_outputs = None
    if len(setup.stages) == 2:
        two_stage_outputs = run_two_stage_final_batch(
            stage1_outputs=stage_outputs[0]["final_outputs"],
            stage2=setup.stages[1],
            stage2_final_inputs=stage_outputs[1]["final_inputs"],
            mips=mips,
            outdir=setup_outdir / "two_stage_final",
            centerlines=centerlines,
            penalties=single_stage_kwargs.get("penalties", DEFAULT_PELT_PENALTIES),
            geometry_feature_cfg=single_stage_kwargs.get("geometry_feature_cfg"),
            min_segment_nodes=min_segment_nodes,
            centerline_id_col=single_stage_kwargs.get("centerline_id_col", "main_path_id"),
            centerline_geometry_col=single_stage_kwargs.get("centerline_geometry_col", "line"),
            node_geometry_col=single_stage_kwargs.get("node_geometry_col", "geometry"),
        )

    return {
        "setup": setup,
        "stage_outputs": stage_outputs,
        "two_stage_outputs": two_stage_outputs,
    }


def run_all_segmentation_setups(
    df,
    dfN,
    mips,
    outdir: str | Path,
    centerlines=None,
    setups: Sequence[SegmentationSetupConfig] = SEGMENTATION_SETUPS,
    **kwargs,
) -> Dict[str, object]:
    outputs = {}
    for setup in setups:
        outputs[setup.name] = run_segmentation_setup(
            setup=setup,
            df=df,
            dfN=dfN,
            mips=mips,
            outdir=outdir,
            centerlines=centerlines,
            **kwargs,
        )
    return outputs
