import importlib
import pickle
from pathlib import Path

import geopandas as gpd

import PELT
import PELT_finalize as pf
import PELT_tuning as pt
import PELT_consensus_calibration as pcc

importlib.reload(PELT)
importlib.reload(pcc)
importlib.reload(pt)
importlib.reload(pf)


# ------------------------------------------------------------
# 0. Paths and inputs
# ------------------------------------------------------------
# Make sure this exists in your notebook already, or uncomment/edit:
# SWORD_directory = "/Volumes/PhD/SWORD/v17b/"
SWORD_directory = '/Volumes/PhD/SWORD/v17b/adjusted/'
continent = "sa"

SWOT_NODE_DIR = "/Volumes/PhD/SWOT/RiverSP_D_parq/node/"
SWOT_REGION = "SA"

OUTDIR_TUNING = "test_figures_multichan"
OUTDIR_FINAL = "test_figures_final_multichan"


# ------------------------------------------------------------
# 1. Load SWORD data
# ------------------------------------------------------------
df = gpd.read_file(SWORD_directory + f"{continent}_sword_reaches_v17b.gpkg")
dfN = gpd.read_file(SWORD_directory + f"{continent}_sword_nodes_v17b.gpkg")
dfG = gpd.read_file("/Volumes/PhD/SWORD/v17c/global_edges.gpkg")

dfG = dfG[dfG["reach_id"].isin(df["reach_id"].values)].copy()

df = df.to_crs("EPSG:3857")
dfG = dfG.to_crs("EPSG:3857")
dfN = dfN.to_crs("EPSG:3857")

print("Loaded reaches:", len(df))
print("Loaded nodes:", len(dfN))
print("Loaded filtered edges:", len(dfG))


# ------------------------------------------------------------
# 2. Define tuning set
# ------------------------------------------------------------
mips = [
    680, 560, 1951, 2540, 1094, 2509, 381, 30, 35, 3247, 2957, 1033,
    1171, 1147, 113, 1788, 539, 2244, 2947, 236, 1378, 2443, 1617,
    659, 599, 2206, 2202, 1,
]

PELT_PENALTIES = (2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)


# ------------------------------------------------------------
# 3. One full broad tuning grid
# ------------------------------------------------------------
grid_outputs = pt.PELT_grid_search(
    df=dfG,
    dfN=dfN,
    mips=mips,
    input_windows=[0, 2, 3, 4, 5],
    PELT_penalties=PELT_PENALTIES,
    min_support_frac_runs_values=(0.05, 0.10, 0.15, 0.20, 0.25),
    stop_rel_improvement_values=(0.04, 0.045, 0.05, 0.06),
    consensus_cfg=PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
    make_plots=False,
    print_timings=False,
    save_exports=True,
    show_progress=True,
    outdir=OUTDIR_TUNING,
    swot_node_dir=SWOT_NODE_DIR,
    swot_region=SWOT_REGION,
)


# ------------------------------------------------------------
# 4. Calibrate consensus threshold and reapply it cheaply
#    This does not rerun the PELT grid search.
# ------------------------------------------------------------
grid_outputs = pt.calibrate_and_reapply_consensus_to_grid_outputs(
    grid_outputs=grid_outputs,
    save_exports=True,
    outdir=OUTDIR_TUNING,
    calibration_outdir=OUTDIR_TUNING,
    local_upper_values_km=(15, 20, 25, 30, 35, 40, 50, 60, 80, 100, 120),
    target_families=("w2", "w3", "w4", "w5"),
    min_likely_pairs=2,
    min_consecutive=3,
    tol_km=0.0,
    round_to_km=0.5,
    require_all_target_families=False,
    min_plateau_families=3,
)

print("Calibrated consensus threshold km:", grid_outputs["merge_threshold_m"] / 1000.0)
print(grid_outputs["consensus_calibration"]["plateau_summary"].to_frame("value"))
print(grid_outputs["consensus_calibration"]["chosen_summary"].to_frame("value"))
print(grid_outputs["consensus_calibration"]["sweep_df"])


# ------------------------------------------------------------
# 5. Tuning analysis using calibrated consensus clusters
# ------------------------------------------------------------
analysis_tables = pt.run_tuning_analysis(
    results_dict=grid_outputs["results_dict"],
    print_outputs=True,
)

with open(Path(OUTDIR_TUNING) / "PELT_analysis_tables.pkl", "wb") as f:
    pickle.dump(analysis_tables, f, protocol=pickle.HIGHEST_PROTOCOL)


# ------------------------------------------------------------
# 6. Convert tuning outputs to finalization inputs
#    This now uses the dynamic final support rule:
#    2->2, 3->2, 4->3, 5->4, 6->5.
# ------------------------------------------------------------
final_inputs = pf.derive_finalization_inputs_from_analysis_tables(
    analysis_tables,
    consensus_cfg=grid_outputs["consensus_cfg"],
)

print("Winning family:", final_inputs["winning_family"])
print("Window key:", final_inputs["window_key"])
print("Number of retained selector settings:", len(final_inputs["selector_settings"]))
print("Final stable_support_count:", final_inputs["stable_support_count"])
print("Final consensus threshold km:", final_inputs["consensus_cfg"].merge_threshold_m / 1000.0)

print("Selector settings:")
for setting in final_inputs["selector_settings"]:
    print(" ", setting)


# ------------------------------------------------------------
# 7. Final reach segmentation
# ------------------------------------------------------------
final_outputs = pf.run_final_batch(
    df=dfG,
    dfN=dfN,
    mips=mips,
    window_key=final_inputs["window_key"],
    penalties=PELT_PENALTIES,
    pelt_feature_cols=("width_s", "nch_s"),
    selector_settings=final_inputs["selector_settings"],
    stable_support_count=final_inputs["stable_support_count"],
    consensus_cfg=final_inputs["consensus_cfg"],
    swot_node_dir=SWOT_NODE_DIR,
    swot_region=SWOT_REGION,
    make_plots=False,
    save_exports=True,
    show_progress=True,
    print_timings=False,
    outdir=OUTDIR_FINAL,
)


# ------------------------------------------------------------
# 8. Final summaries
# ------------------------------------------------------------
final_summary = pf.summarize_final_batch(final_outputs)

print(final_summary["overall_summary_df"])
print(final_summary["reach_summary_df"].head())

with open(Path(OUTDIR_FINAL) / "PELT_final_summary.pkl", "wb") as f:
    pickle.dump(final_summary, f, protocol=pickle.HIGHEST_PROTOCOL)

example_run_key = f"{mips[0]}_{final_inputs['window_key']}"
print("Example run key:", example_run_key)
print("Stable breaks m:", final_outputs["results_dict"][example_run_key]["stable_breaks_m"])
print("Stable segments:", final_outputs["results_dict"][example_run_key]["stable_segments"])
