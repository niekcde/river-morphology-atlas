# PELT Pipeline

This pipeline now uses one shared frozen breakpoint-consensus method from the start of tuning through final segmentation.

## Files

- `PELT.py`
  Shared runtime code for feature building, PELT sweeps, breakpoint selection, and breakpoint consensus.
- `PELT_tuning.py`
  Broad grid-search and tuning analysis across window families and selector settings.
- `PELT_consensus_calibration.py`
  One-time calibration utilities for the consensus merge threshold from saved tuning outputs.
- `PELT_finalize.py`
  Final reach segmentation using the tuned window family and retained selector settings.
- `PELT_geometry_features.py`
  Newer sinuosity/curvature feature definitions used when `sinu` or `curv_int` are selected.
- `PELT_segmentation_runner.py`
  Configured runner for the five standard segmentation setups, including two-stage runs.
- `reach_concatenation.py`
  Builds continuous main-path centerlines used by the geometry features.
- `PELT_tuning_parallel_call.py`
  Example batch entry point for the broad tuning run.
- `incorporate_multichannel_segments.py`
  Reach/node preprocessing used before PELT.
- `open_SWOT_files.py`
  SWOT node WSE loading used during reach preparation.

## Frozen Consensus Method

- Consensus clustering method: `complete_linkage`
- Frozen merge threshold: `11.5 km`
- Tuning-stage stable support rule: `stable_support_frac_min = 0.75`
- Final explicit-grid stable support rule: dynamic all-but-one support, with two settings requiring both. This maps `2->2`, `3->2`, `4->3`, `5->4`, `6->5`.

The clustering code lives only in `PELT.py`. `PELT_tuning.py` and `PELT_finalize.py` both call the same implementation.

## Channel-Count Feature

By default, `FeatureConfig.multi_chan_treatment=True`, so requested `nch_s` is computed as a smoothed multi-channel presence feature rather than a continuous channel-count average:

1. Raw nodes with `n_channels > 1` are assigned `1`; single-channel nodes are assigned `0`.
2. For non-raw windows, `nch_s` is the rolling mean of that binary signal.
3. For the raw window, `nch_s` is the unsmoothed binary signal.

This makes `nch_s` represent the local fraction of multi-channel nodes, so the single-to-multi transition is treated differently from numeric changes within already multi-channel reaches. Set `multi_chan_treatment=False` in `FeatureConfig` to recover the older continuous channel-count behavior.

## Standard Workflow

### 1. Broad tuning run

Run the broad grid search over the target reaches and window families.

Typical entry point:

```python
import PELT
import PELT_tuning as pt

grid_outputs = pt.PELT_grid_search_parallel(
    df=dfG,
    dfN=dfN,
    mips=mips,
    input_windows=[0, 2, 3, 4, 5],
    PELT_penalties=(2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0),
    min_support_frac_runs_values=(0.05, 0.10, 0.15, 0.20, 0.25),
    stop_rel_improvement_values=(0.04, 0.045, 0.05, 0.06),
    pelt_feature_cols=("width_s", "nch_s"),
    consensus_cfg=PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    make_plots=False,
    print_timings=False,
    save_exports=True,
    show_progress=False,
    outdir="test_figures",
)
```

Use `pelt_feature_cols=("sinu", "curv_int")` or
`("width_s", "nch_s", "sinu", "curv_int")` for geometry or mixed runs. Geometry
feature runs require concatenated centerlines keyed by `main_path_id`, and should
use positive window families only, for example `input_windows=[2, 3, 4, 5]`.

Main outputs:

- `results_dict`
- `grid_settings_master_df`
- `grid_consensus_master_df`
- `grid_run_summary_master_df`
- `PELT_results_dict.pkl` if `save_exports=True`

### 2. Tuning analysis

Build the analysis tables used to choose the representative window family and retained selector settings.

```python
analysis_tables = pt.run_tuning_analysis(
    results_dict=grid_outputs["results_dict"],
    print_outputs=True,
)
```

Use these outputs to decide:

- the winning window family, for example `w2`
- the retained selector settings for finalization

### 3. Optional method recalibration

This is not part of the normal production run. Use it only if you want to re-estimate the global consensus merge threshold from a new method-development dataset.

```python
import PELT_consensus_calibration as pcc

calibration = pcc.calibrate_consensus_from_results_dict(grid_outputs["results_dict"])
pcc.save_calibration_artifacts(calibration, "test_figures")
```

This module reproduces the full-analysis threshold sweep and the first-stable-plateau rule that produced the frozen `11.5 km` threshold.

### 4. Final reach segmentation

Once the window family and retained selector settings are fixed, run final segmentation for one reach or for a batch of `mips`.

Single reach:

```python
import PELT
import PELT_finalize as pf

final_result = pf.run_final_reach_pipeline(
    df=dfG,
    dfN=dfN,
    mip=113,
    window_key=2,
    selector_settings=pf.get_stage1_w2_selector_settings(),
    consensus_cfg=PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    make_plot=False,
    save_exports=False,
)
```

Batch of reaches:

```python
final_outputs = pf.run_final_batch(
    df=dfG,
    dfN=dfN,
    mips=mips,
    window_key=2,
    pelt_feature_cols=("width_s", "nch_s"),
    selector_settings=pf.get_stage1_w2_selector_settings(),
    consensus_cfg=PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    save_exports=True,
    outdir="test_figures_final",
)
```

For geometry-enabled finalization, pass `centerlines=<dataframe or dict>` where
the dataframe has `main_path_id` and `line` columns. The finalizer asserts that
centerlines are LineStrings, orients them to increasing node `dist_m`, and stores
geometry feature missing-rate QA in each result.

Main final outputs:

- `results_dict`
- `grid_settings_master_df`
- `grid_consensus_master_df`
- `grid_run_summary_master_df`
- `stable_breaks_m` per reach
- `stable_segments` per reach

### 5. Five configured setups

`PELT_segmentation_runner.py` defines the five standard setups:

1. `01_width_channels`: `width_s + nch_s`
2. `02_sinuosity_curvature`: `sinu + curv_int`
3. `03_all_features`: `width_s + nch_s + sinu + curv_int`
4. `04_width_channels_then_geometry`
5. `05_geometry_then_width_channels`

Example:

```python
import PELT_segmentation_runner as psr

centerlines, centerline_qa = psr.build_centerlines_from_edges(dfG)

outputs = psr.run_all_segmentation_setups(
    df=dfG,
    dfN=dfN,
    mips=mips,
    centerlines=centerlines,
    outdir="PELT_outputs",
    swot_node_dir=SWOT_NODE_DIR,
    swot_region="SA",
)
```

Each setup writes to a separate named output directory.

## What Produces the Final Segmentation

For a given `mip`, the final segmentation comes from:

1. `_prepare_reach_nodes(...)`
2. `PELT.run_full_pipeline(...)` to build the multiscale PELT outputs
3. `apply_explicit_selector_grid(...)` in `PELT_finalize.py`
4. shared complete-linkage consensus in `PELT.build_grid_consensus_from_break_rows(...)`

The final segmentation delivered to downstream code is:

- `stable_breaks_m`
- `stable_segments`

These are the outputs to use when another function receives a list of `mips` and needs to compute tuned reach segmentation.

## Recommended Production Pattern

1. Freeze the consensus method once.
2. Run broad tuning with that same frozen consensus config.
3. Pick the representative family and retained selector settings.
4. Run final segmentation on the requested `mips`.
5. Use `stable_breaks_m` and `stable_segments` as the final reach segmentation product.
