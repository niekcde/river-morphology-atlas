#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable

import geopandas as gpd

import PELT_geometry_features as pgf
import PELT_segmentation_runner as psr


DEFAULT_NOTEBOOK_MIPS = [
    6000007, 6000009, 6000034, 6000041, 6000053, 6000084, 6000092,
    6000141, 6000148, 6000152, 6000212, 6000217, 6000249, 6000279,
    6000282, 6000287, 6000297, 6000318, 6000323, 6000344, 6000399,
    6000436, 6000560, 6000573, 6000622, 6000678, 6001070, 6001096,
]

DEFAULT_PENALTIES = (2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
DEFAULT_MIN_SUPPORT_FRAC_RUNS_VALUES = (0.05, 0.10, 0.15, 0.20, 0.25)
DEFAULT_STOP_REL_IMPROVEMENT_VALUES = (0.04, 0.045, 0.05, 0.06)


def _parse_int_list(value: str) -> list[int]:
    parts = [part.strip() for part in value.split(",")]
    out = [int(part) for part in parts if part]
    if not out:
        raise ValueError("Expected at least one integer.")
    return out


def _load_mips_from_file(path: Path) -> list[int]:
    raw = path.read_text().strip()
    if not raw:
        raise ValueError(f"MIP file is empty: {path}")
    if path.suffix.lower() == ".json":
        values = json.loads(raw)
        if not isinstance(values, list):
            raise ValueError(f"Expected JSON list in {path}")
        return [int(v) for v in values]
    return _parse_int_list(raw.replace("\n", ","))


def _jsonify(value):
    if is_dataclass(value):
        return _jsonify(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _save_pickle(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(_jsonify(obj), f, indent=2)


def _save_final_inputs(final_inputs: dict, tuning_outdir: Path) -> None:
    _save_pickle(final_inputs, tuning_outdir / "PELT_final_inputs.pkl")
    _save_json(final_inputs, tuning_outdir / "PELT_final_inputs.json")


def _load_inputs(sword_dir: Path, continent: str):
    continent_upper = continent.upper()
    reaches_path = sword_dir / f"sword_{continent_upper}_v17c_reaches.parquet"
    nodes_path = sword_dir / f"sword_{continent_upper}_v17c_nodes.parquet"

    dfG = gpd.read_parquet(reaches_path).to_crs("EPSG:3857")
    dfN = gpd.read_parquet(nodes_path).to_crs("EPSG:3857")
    return dfG, dfN, reaches_path, nodes_path


def _choose_mips(args) -> list[int]:
    if args.mips_csv:
        return _parse_int_list(args.mips_csv)
    if args.mips_file:
        return _load_mips_from_file(Path(args.mips_file))
    return list(DEFAULT_NOTEBOOK_MIPS)


def _build_centerlines(dfG, mips: Iterable[int], outdir: Path, use_duckdb_spatial: bool):
    centerlines_path = outdir / "_shared" / "centerlines.parquet"
    return psr.build_centerlines_from_edges(
        dfG[dfG["main_path_id"].isin(list(mips))],
        out_path=centerlines_path,
        assert_all_linestring=True,
        endpoint_gap_tol=160,
        endpoint_gap_connected_tol=1e-6,
        graph_union_grid_size=1e-4,
        use_duckdb_spatial=use_duckdb_spatial,
    )


def _run_single_stage(
    stage,
    dfG,
    dfN,
    mips,
    out_root: Path,
    centerlines,
    geometry_feature_cfg,
    args,
):
    output = psr.run_single_stage_setup(
        stage=stage,
        df=dfG,
        dfN=dfN,
        mips=mips,
        outdir=out_root / stage_output_dir_name(stage),
        centerlines=centerlines,
        penalties=DEFAULT_PENALTIES,
        min_support_frac_runs_values=DEFAULT_MIN_SUPPORT_FRAC_RUNS_VALUES,
        stop_rel_improvement_values=DEFAULT_STOP_REL_IMPROVEMENT_VALUES,
        swot_node_dir=args.swot_node_dir,
        swot_region=args.swot_region,
        centerline_id_col="main_path_id",
        centerline_geometry_col="line",
        node_geometry_col="geometry",
        geometry_feature_cfg=geometry_feature_cfg,
        check_centerline_orientation=True,
        calibrate_consensus=True,
        show_progress=args.show_progress,
        parallel_tuning=args.parallel_tuning,
        max_workers=args.max_workers,
    )
    _save_final_inputs(output["final_inputs"], out_root / stage_output_dir_name(stage) / "tuning")
    return output


def stage_output_dir_name(stage) -> str:
    setup_map = {
        "width_channels": "01_width_channels",
        "sinuosity_curvature": "02_sinuosity_curvature",
        "all_features": "03_all_features",
    }
    try:
        return setup_map[stage.name]
    except KeyError as exc:
        raise ValueError(f"Unexpected single-stage name: {stage.name}") from exc


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the five PELT segmentation setups used in segmentation_test.ipynb.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sword-dir", required=True, help="Directory containing SWORD v17c parquet inputs.")
    parser.add_argument("--swot-node-dir", default="", help="SWOT node directory. Unused for the current feature sets.")
    parser.add_argument("--swot-region", default="SA")
    parser.add_argument("--continent", default="SA")
    parser.add_argument("--outdir", default="PELT_outputs")
    parser.add_argument("--mips-csv", default="", help="Comma-separated main_path_id list to run.")
    parser.add_argument(
        "--mips-file",
        default="",
        help="Optional file containing MIPs as JSON list or CSV/text. "
        "If omitted, the script uses the current 28-MIP notebook list directly.",
    )
    parser.add_argument("--parallel-tuning", action="store_true")
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--min-segment-nodes", type=int, default=10)
    parser.add_argument(
        "--use-duckdb-centerline-merge",
        action="store_true",
        help="Use DuckDB spatial for centerline ST_Collect/ST_LineMerge. "
        "Default is the shapely merge path, which avoids DuckDB spatial extension issues on HPC.",
    )
    args = parser.parse_args()

    sword_dir = Path(args.sword_dir)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading inputs...")
    dfG, dfN, reaches_path, nodes_path = _load_inputs(sword_dir, args.continent)
    mips = _choose_mips(args)
    print(f"Loaded {len(dfG)} reaches and {len(dfN)} nodes.")
    print(f"Running {len(mips)} MIPs.")

    manifest = {
        "sword_dir": sword_dir,
        "reaches_path": reaches_path,
        "nodes_path": nodes_path,
        "swot_node_dir": args.swot_node_dir,
        "swot_region": args.swot_region,
        "continent": args.continent,
        "outdir": outdir,
        "mips": mips,
        "penalties": DEFAULT_PENALTIES,
        "min_support_frac_runs_values": DEFAULT_MIN_SUPPORT_FRAC_RUNS_VALUES,
        "stop_rel_improvement_values": DEFAULT_STOP_REL_IMPROVEMENT_VALUES,
        "min_segment_nodes": args.min_segment_nodes,
        "parallel_tuning": args.parallel_tuning,
        "max_workers": args.max_workers,
        "centerline_merge_kwargs": {
            "endpoint_gap_tol": 160,
            "endpoint_gap_connected_tol": 1e-6,
            "graph_union_grid_size": 1e-4,
            "use_duckdb_spatial": args.use_duckdb_centerline_merge,
        },
    }
    _save_json(manifest, outdir / "workflow_manifest.json")

    print("Building centerlines once for all geometry-based runs...")
    centerlines, centerline_qa = _build_centerlines(
        dfG,
        mips,
        outdir,
        use_duckdb_spatial=args.use_duckdb_centerline_merge,
    )
    geometry_feature_cfg = pgf.GeometryFeatureConfig(
        dist_col="dist_m",
        width_col="multi_width",
    )
    _save_pickle(centerline_qa, outdir / "_shared" / "centerline_geometry_qa.pkl")

    setup_by_name = {setup.name: setup for setup in psr.SEGMENTATION_SETUPS}

    print("Running setup 01: width + nr_channels")
    width_output = _run_single_stage(
        stage=setup_by_name["01_width_channels"].stages[0],
        dfG=dfG,
        dfN=dfN,
        mips=mips,
        out_root=outdir,
        centerlines=None,
        geometry_feature_cfg=None,
        args=args,
    )

    print("Running setup 02: sinuosity + curvature")
    sinu_output = _run_single_stage(
        stage=setup_by_name["02_sinuosity_curvature"].stages[0],
        dfG=dfG,
        dfN=dfN,
        mips=mips,
        out_root=outdir,
        centerlines=centerlines,
        geometry_feature_cfg=geometry_feature_cfg,
        args=args,
    )

    print("Running setup 03: width + nr_channels + sinuosity + curvature")
    all_output = _run_single_stage(
        stage=setup_by_name["03_all_features"].stages[0],
        dfG=dfG,
        dfN=dfN,
        mips=mips,
        out_root=outdir,
        centerlines=centerlines,
        geometry_feature_cfg=geometry_feature_cfg,
        args=args,
    )

    print("Running setup 04 finalization: width/channels then geometry")
    two_stage_width_then_geom = psr.run_two_stage_final_batch(
        stage1_outputs=width_output["final_outputs"],
        stage2=setup_by_name["04_width_channels_then_geometry"].stages[1],
        stage2_final_inputs=sinu_output["final_inputs"],
        mips=mips,
        outdir=outdir / "04_width_channels_then_geometry" / "two_stage_final",
        centerlines=centerlines,
        penalties=DEFAULT_PENALTIES,
        geometry_feature_cfg=geometry_feature_cfg,
        min_segment_nodes=args.min_segment_nodes,
        centerline_id_col="main_path_id",
        centerline_geometry_col="line",
        node_geometry_col="geometry",
    )

    print("Running setup 05 finalization: geometry then width/channels")
    two_stage_geom_then_width = psr.run_two_stage_final_batch(
        stage1_outputs=sinu_output["final_outputs"],
        stage2=setup_by_name["05_geometry_then_width_channels"].stages[1],
        stage2_final_inputs=width_output["final_inputs"],
        mips=mips,
        outdir=outdir / "05_geometry_then_width_channels" / "two_stage_final",
        centerlines=centerlines,
        penalties=DEFAULT_PENALTIES,
        geometry_feature_cfg=geometry_feature_cfg,
        min_segment_nodes=args.min_segment_nodes,
        centerline_id_col="main_path_id",
        centerline_geometry_col="line",
        node_geometry_col="geometry",
    )

    workflow_summary = {
        "single_stage": {
            "01_width_channels": {
                "window_key": width_output["final_inputs"]["window_key"],
                "stable_support_count": width_output["final_inputs"]["stable_support_count"],
            },
            "02_sinuosity_curvature": {
                "window_key": sinu_output["final_inputs"]["window_key"],
                "stable_support_count": sinu_output["final_inputs"]["stable_support_count"],
            },
            "03_all_features": {
                "window_key": all_output["final_inputs"]["window_key"],
                "stable_support_count": all_output["final_inputs"]["stable_support_count"],
            },
        },
        "two_stage": {
            "04_width_channels_then_geometry": {
                "n_reaches": len(two_stage_width_then_geom["reach_summary_df"]),
            },
            "05_geometry_then_width_channels": {
                "n_reaches": len(two_stage_geom_then_width["reach_summary_df"]),
            },
        },
    }
    _save_json(workflow_summary, outdir / "workflow_summary.json")
    print("Workflow complete.")


if __name__ == "__main__":
    main()
