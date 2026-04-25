#!/bin/bash
set -euo pipefail

source /opt/easybuild/foss2022a/software/Miniconda3/4.12.0/bin/activate RivAlg10

cd /home/6256481/River_Morphology/morphology_atlas/code/

PYTHON_SCRIPT="run_PELT_five_setup_workflow.py"
LOG_DIR="/home/6256481/River_Morphology/morphology_atlas/log"
LOG_FILE="$LOG_DIR/PELT_five_setup_workflow.log"

mkdir -p "$LOG_DIR"

python3 "$PYTHON_SCRIPT" \
  --sword-dir "/home/6256481/River_Morphology/morphology_atlas/data/" \
  --swot-node-dir "/home/6256481/River_Morphology/morphology_atlas/data/SWOT/node/" \
  --continent "SA" \
  --outdir "/home/6256481/River_Morphology/morphology_atlas/output/PELT_outputs" \
  --parallel-tuning \
  --max-workers 10 \
  > "$LOG_FILE" 2>&1
