#!/bin/bash
set -euo pipefail

source /opt/easybuild/foss2022a/software/Miniconda3/4.12.0/bin/activate RivAlg10

cd /home/6256481/River_Morphology/morphology_atlas/code/

PYTHON_SCRIPT="run_PELT_five_setup_workflow.py"
LOG_DIR="/home/6256481/River_Morphology/morphology_atlas/log"
LOG_FILE="$LOG_DIR/PELT_five_setup_workflow.log"

mkdir -p "$LOG_DIR"

# HPC default: avoid DuckDB spatial extension issues by using the shapely merge path.
# On a local machine with a compatible DuckDB spatial install, you can add:
#   --use-duckdb-centerline-merge
python3 "$PYTHON_SCRIPT" \
  --sword-dir "/home/6256481/River_Morphology/morphology_atlas/data/" \
  --swot-node-dir "/home/6256481/River_Morphology/morphology_atlas/data/SWOT/node/" \
  --continent "SA" \
  --outdir "/home/6256481/River_Morphology/morphology_atlas/output/PELT_outputs" \
  --mips-csv "6000007,6000009,6000034,6000041,6000053,6000084,6000092,6000141,6000148,6000152,6000212,6000217,6000249,6000279,6000282,6000287,6000297,6000318,6000323,6000344,6000399,6000436,6000560,6000573,6000622,6000678,6001070,6001096" \
  --parallel-tuning \
  --max-workers 10 \
  > "$LOG_FILE" 2>&1
