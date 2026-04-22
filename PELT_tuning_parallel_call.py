from pathlib import Path
import pickle
import geopandas as gpd

import PELT
import PELT_tuning as pt


def main(dfG, dfN, directory):
    mips = [
        680, 560, 1951, 2540, 1094, 2509, 381, 30, 35, 3247, 2957, 1033,
        1171, 1147, 113, 1788, 539, 2244, 2947, 236, 1378, 2443, 1617,
        659, 599, 2206, 2202, 1,
    ]

    penalties = (2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 160.0)
    min_support_frac_runs_values = (0.05, 0.10, 0.15, 0.20, 0.25)
    stop_rel_improvement_values = (0.04, 0.045, 0.05, 0.06)

    outdir = Path("test_figures_final")

    grid_outputs = pt.PELT_grid_search_parallel(
        df=dfG,
        dfN=dfN,
        mips=mips,
        input_windows=[0, 2, 3, 4, 5],
        PELT_penalties=penalties,
        min_support_frac_runs_values=min_support_frac_runs_values,
        stop_rel_improvement_values=stop_rel_improvement_values,
        consensus_cfg=PELT.DEFAULT_FROZEN_CONSENSUS_CONFIG,
        make_plots=False,
        print_timings=False,
        save_exports=True,
        show_progress=False,
        outdir=outdir,
        swot_node_dir=directory + 'SWOT/node/',
        swot_region="SA",
        max_workers=10,
    )

    analysis_tables = pt.run_tuning_analysis(
        results_dict=grid_outputs["results_dict"],
        print_outputs=True,
    )

    with open(outdir / "PELT_analysis_tables.pkl", "wb") as f:
        pickle.dump(analysis_tables, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    SWORD_directory = '/home/6256481/River_Morphology/morphology_atlas/data/'
    continent = 'sa'
    dfN = gpd.read_file(SWORD_directory + f'{continent}_sword_nodes_v17b.gpkg')
    dfG = gpd.read_file(SWORD_directory + 'global_edges.gpkg')

    dfG = dfG[dfG['reach_id'].isin(dfN['reach_id'].unique())]
    dfG = dfG.to_crs('EPSG:3857')
    dfN = dfN.to_crs('EPSG:3857')
    main(dfG, dfN, SWORD_directory)
