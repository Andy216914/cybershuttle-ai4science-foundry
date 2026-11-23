#!/bin/bash
#SBATCH --job-name=poc-prepare-data
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#
# Job 1 (CPU): download EuroSAT and its split files (sha256-checked), unpack,
# determine the band order from the data, and pack everything into one array
# file that both models read. Needs the Clay environment (Clay recipe, job 1).

set -euo pipefail
trap 'echo "== $SLURM_JOB_NAME FAILED at line $LINENO (exit $?)  $(date -Is)" >&2' ERR
: "${SLURM_SUBMIT_DIR:?Submit with sbatch from this folder (run_all.sh does).}"
cd "$SLURM_SUBMIT_DIR"
# shellcheck source=../config.sh
source config.sh
clay_make_dirs
echo "== $SLURM_JOB_NAME  job=$SLURM_JOB_ID  node=$(hostname)  start=$(date -Is)"
echo "== $(poc_code_version)"

echo "== Download (skipped when already present with the right checksum)"
poc_fetch "$EUROSAT_BASE/EuroSATallBands.zip" "$POC_DATA/EuroSATallBands.zip" "$EUROSAT_ZIP_SHA256"
poc_fetch "$EUROSAT_BASE/eurosat-train.txt" "$POC_DATA/eurosat-train.txt" "$EUROSAT_TRAIN_SHA256"
poc_fetch "$EUROSAT_BASE/eurosat-val.txt" "$POC_DATA/eurosat-val.txt" "$EUROSAT_VAL_SHA256"
poc_fetch "$EUROSAT_BASE/eurosat-test.txt" "$POC_DATA/eurosat-test.txt" "$EUROSAT_TEST_SHA256"

clay_activate
echo "== Unpack"
if [[ ! -f "$POC_DATA/eurosat/.unpacked" ]]; then
    rm -rf "$POC_DATA/eurosat"
    python -m zipfile -e "$POC_DATA/EuroSATallBands.zip" "$POC_DATA/eurosat"
    touch "$POC_DATA/eurosat/.unpacked"
fi
echo "   $(find "$EUROSAT_TIF_DIR" -name '*.tif' | wc -l) images"

echo "== Pack and check band order"
python scripts/prepare_eurosat.py --tif-dir "$EUROSAT_TIF_DIR" --split-dir "$POC_DATA" \
    --out "$POC_DATA/packed" --results "$RESULTS"
echo "== PREPARE DATA COMPLETED  end=$(date -Is)"
