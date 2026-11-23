#!/bin/bash
#SBATCH --job-name=poc-extract-features
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:45:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#
# Job 3 (GPU): frozen features for all 27,000 images, Clay first, then SatMAE
# (correct band order, and the order its published EuroSAT code assumes).
# run_all.sh may replace the GPU type with the one that starts earliest.

set -euo pipefail
trap 'echo "== $SLURM_JOB_NAME FAILED at line $LINENO (exit $?)  $(date -Is)" >&2' ERR
: "${SLURM_SUBMIT_DIR:?Submit with sbatch from this folder (run_all.sh does).}"
cd "$SLURM_SUBMIT_DIR"
# shellcheck source=../config.sh
source config.sh
clay_make_dirs
echo "== $SLURM_JOB_NAME  job=$SLURM_JOB_ID  node=$(hostname)  start=$(date -Is)"
echo "== $(poc_code_version)"
echo "== GPU"
nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap --format=csv

echo "== Clay v1.5"
clay_activate
export HF_HUB_OFFLINE=1
python "$INFER_DIR/scripts/clay_check_env.py" --lock "$INFER_DIR/scripts/clay_requirements.txt" --teacher "$CLAY_TEACHER"
python scripts/embed_clay.py --data "$POC_DATA/packed" --checkpoint "$CLAY_WEIGHTS/clay-v1.5.ckpt" \
    --metadata "$CLAY_REPO/configs/metadata.yaml" --out "$POC_EMB/clay.npz"
set +u; conda deactivate; set -u

echo "== SatMAE"
poc_satmae_activate
python scripts/embed_satmae.py --data "$POC_DATA/packed" --repo "$SATMAE_REPO" \
    --checkpoint "$SATMAE_CKPT" --out-dir "$POC_EMB"

mkdir -p "$RESULTS"
cp "$POC_EMB/clay.json" "$RESULTS/clay_features_run.json"
cp "$POC_EMB/satmae.json" "$RESULTS/satmae_features_run.json"
echo "== Slurm accounting"
sacct -j "$SLURM_JOB_ID" --format=JobID,JobName,Partition,AllocTRES%40,State,Elapsed,MaxRSS
echo "== EXTRACT FEATURES COMPLETED  end=$(date -Is)"
