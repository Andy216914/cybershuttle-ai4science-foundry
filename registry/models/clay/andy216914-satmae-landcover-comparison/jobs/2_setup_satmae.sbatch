#!/bin/bash
#SBATCH --job-name=poc-setup-satmae
#SBATCH --account=coc
#SBATCH --partition=coc-cpu
#SBATCH --qos=coc-ice
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=results/logs/%x-%j.out
#SBATCH --error=results/logs/%x-%j.err
#
# Job 2 (CPU): SatMAE code at the pinned commit, its environment (Python 3.9,
# torch 1.12.1, timm 0.3.2 with the one-line patch from PR #28), and the
# pretrained ViT-Large checkpoint (md5 from Zenodo). Reused when complete.

set -euo pipefail
trap 'echo "== $SLURM_JOB_NAME FAILED at line $LINENO (exit $?)  $(date -Is)" >&2' ERR
: "${SLURM_SUBMIT_DIR:?Submit with sbatch from this folder (run_all.sh does).}"
cd "$SLURM_SUBMIT_DIR"
# shellcheck source=../config.sh
source config.sh
clay_make_dirs
echo "== $SLURM_JOB_NAME  job=$SLURM_JOB_ID  node=$(hostname)  start=$(date -Is)"
echo "== $(poc_code_version)"

echo "== SatMAE code at $SATMAE_COMMIT"
if [[ ! -d "$SATMAE_REPO/.git" ]]; then git clone --quiet "$SATMAE_GIT_URL" "$SATMAE_REPO"; fi
git -C "$SATMAE_REPO" fetch --quiet origin
git -C "$SATMAE_REPO" checkout --quiet --detach "$SATMAE_COMMIT"
[[ "$(git -C "$SATMAE_REPO" rev-parse HEAD)" == "$SATMAE_COMMIT" ]]

echo "== Environment at $SATMAE_ENV"
STAMP=$(cat env/satmae_environment.yml env/satmae_requirements.txt | sha256sum | cut -d' ' -f1)
if [[ -f "$SATMAE_ENV/.poc_env_complete" && "$(cat "$SATMAE_ENV/.poc_env_complete")" == "$STAMP" ]]; then
    echo "   reusing finished environment"
    poc_satmae_activate
else
    clay_load_conda
    rm -rf "$SATMAE_ENV"
    conda env create --quiet -p "$SATMAE_ENV" -f env/satmae_environment.yml
    set +u; conda activate "$SATMAE_ENV"; set -u
    python -m pip install --progress-bar off -r env/satmae_requirements.txt
    # timm 0.3.2 imports torch._six.container_abcs, removed in torch 1.9 (same fix as PR #28)
    HELPERS=$(python -c "import importlib.util, os; print(os.path.join(os.path.dirname(importlib.util.find_spec('timm').origin), 'models', 'layers', 'helpers.py'))")
    sed -i 's/from torch._six import container_abcs/import collections.abc as container_abcs/' "$HELPERS"
    echo "$STAMP" > "$SATMAE_ENV/.poc_env_complete"
fi
python -c "import sys; sys.path.insert(0, '$SATMAE_REPO'); import torch, torchvision, timm, models_vit_group_channels; print('   torch', torch.__version__, 'cuda', torch.version.cuda, 'torchvision', torchvision.__version__, 'timm', timm.__version__)"
mkdir -p "$RESULTS"
python -m pip freeze > "$RESULTS/satmae_pip_freeze.txt"

echo "== Pretrained checkpoint"
mkdir -p "$(dirname "$SATMAE_CKPT")"
if [[ -f "$SATMAE_CKPT" && "$(md5sum "$SATMAE_CKPT" | cut -d' ' -f1)" == "$SATMAE_CKPT_MD5" ]]; then
    echo "   already present, md5 OK"
else
    curl -fL --retry 5 --retry-delay 10 --silent --show-error -C - -o "$SATMAE_CKPT.part" "$SATMAE_CKPT_URL"
    [[ "$(stat -c %s "$SATMAE_CKPT.part")" == "$SATMAE_CKPT_BYTES" ]] || { echo "ERROR: wrong size; delete $SATMAE_CKPT.part and rerun" >&2; exit 1; }
    [[ "$(md5sum "$SATMAE_CKPT.part" | cut -d' ' -f1)" == "$SATMAE_CKPT_MD5" ]] || { rm -f "$SATMAE_CKPT.part"; echo "ERROR: md5 mismatch" >&2; exit 1; }
    mv "$SATMAE_CKPT.part" "$SATMAE_CKPT"
    echo "   downloaded, md5 OK"
fi
echo "   sha256 $(sha256sum "$SATMAE_CKPT" | cut -d' ' -f1)  $(basename "$SATMAE_CKPT")"
echo "== SETUP SATMAE COMPLETED  end=$(date -Is)"
