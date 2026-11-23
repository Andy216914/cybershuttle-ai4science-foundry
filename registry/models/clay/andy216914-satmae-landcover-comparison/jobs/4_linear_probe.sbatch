#!/bin/bash
#SBATCH --job-name=poc-linear-probe
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#
# Job 4 (CPU): the same logistic-regression probe on every feature set;
# metrics, paired tests, tables and figures into results/.

set -euo pipefail
trap 'echo "== $SLURM_JOB_NAME FAILED at line $LINENO (exit $?)  $(date -Is)" >&2' ERR
: "${SLURM_SUBMIT_DIR:?Submit with sbatch from this folder (run_all.sh does).}"
cd "$SLURM_SUBMIT_DIR"
# shellcheck source=../config.sh
source config.sh
clay_make_dirs
echo "== $SLURM_JOB_NAME  job=$SLURM_JOB_ID  node=$(hostname)  start=$(date -Is)"
echo "== $(poc_code_version)"
clay_activate
python scripts/linear_probe.py --data "$POC_DATA/packed" --emb "$POC_EMB" --results "$RESULTS"
echo "== Results"
cat "$RESULTS/results_table.md"
echo "== LINEAR PROBE COMPLETED  end=$(date -Is)"
