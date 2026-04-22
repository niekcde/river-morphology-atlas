import duckdb
import os

def delete_dot_files(directory):

    for filename in os.listdir(directory):
        if filename.startswith("."):
            full_path = os.path.join(directory, filename)
            if os.path.isfile(full_path):   # avoid removing hidden folders
                os.remove(full_path)
                # print(f"Deleted: {full_path}")

def open_SWOT_files(reachdf, directory, continent, 
                    quality_flag = 1, dark_freq = 0.5, xtrk_dist = [10000, 60000]):
    delete_dot_files(directory)
    
    con = duckdb.connect(":memory:")
    # --- Find by node_id ---
    reachdf['reach_id']= reachdf['reach_id'].astype('str')
    con.register("reach_ids", con.from_df(reachdf))  # <-- FIXED

    ########
    q1 = f"""
        SELECT
            CAST(node_id AS BIGINT) AS node_id,
            MEDIAN(wse_sm) AS wse
        FROM read_parquet('{directory}*{continent}*.parquet')
        WHERE reach_id IN (SELECT reach_id FROM reach_ids)
            AND wse_sm_u <= {quality_flag}
            AND dark_frac <= {dark_freq}
            AND ABS(xtrk_dist) BETWEEN {xtrk_dist[0]} AND {xtrk_dist[1]}
            AND wse_sm > 0
        GROUP BY node_id
        """
    dfRes = con.execute(q1).df()

    return dfRes