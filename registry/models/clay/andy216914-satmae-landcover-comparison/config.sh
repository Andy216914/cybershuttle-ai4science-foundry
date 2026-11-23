# shellcheck shell=bash disable=SC2034  # variables are used by the files that source this one
# Settings for the Clay vs. SatMAE land-cover comparison. Sourced, not run,
# by run_all.sh and every job, from this folder.
#
# Reuses the Clay recipe next to this folder: its work folder, conda
# environment, checkpoint, caches and helper functions (clay_activate, ...).

INFER_DIR="../andy216914-embedding-inference"
# shellcheck source=../andy216914-embedding-inference/scripts/clay_common.sh
source "$INFER_DIR/scripts/clay_common.sh"

# ---- Where things go (scratch, never this repository) ----------------------
POC_WORK="$CLAY_WORKDIR/satmae-landcover"
POC_DATA="$POC_WORK/data"                  # EuroSAT download and the packed arrays
POC_EMB="$POC_WORK/embeddings"             # one .npz of features per model
RESULTS="results"                          # small outputs committed to the repository

# ---- EuroSAT: 27,000 Sentinel-2 images, 13 bands, 10 classes ---------------
# TorchGeo's copy on Hugging Face, pinned to one revision; checksums from
# torchgeo/datasets/eurosat.py and verified on download.
EUROSAT_BASE="https://hf.co/datasets/torchgeo/eurosat/resolve/1ce6f1bfb56db63fd91b6ecc466ea67f2509774c"
EUROSAT_ZIP_SHA256="751f070f9bffa2eed48b24ca2dd0b02959280c08837e8c9a5532a67ba611df59"
EUROSAT_TRAIN_SHA256="1c1d2e855f95deee605a3d992f914d113fddbecf422ec61648057d029a37d695"
EUROSAT_VAL_SHA256="b385741f31daa9f1250cf1e1fe03adfab394e1172e0693df40141af004f60330"
EUROSAT_TEST_SHA256="cf37948894c12bd953930ff54ee9b7abf0b31478abb8d25fd2c6c721db74c592"
EUROSAT_TIF_DIR="$POC_DATA/eurosat/ds/images/remote_sensing/otherDatasets/sentinel_2/tif"

# ---- SatMAE -----------------------------------------------------------------
# Code at the commit used by the SatMAE recipe in PR #28 (DevP1130); the
# pretrained (not fine-tuned) ViT-Large for multispectral Sentinel-2 input,
# Zenodo record 7338613 (DOI 10.5281/zenodo.7338613), md5 from Zenodo.
SATMAE_GIT_URL="https://github.com/sustainlab-group/SatMAE.git"
SATMAE_COMMIT="0b210aceb37a14bbbd897110db5b104b3271d818"
SATMAE_REPO="$POC_WORK/SatMAE"
SATMAE_ENV="$POC_WORK/envs/satmae"
SATMAE_CKPT="$POC_WORK/weights/pretrain-vit-large-e199.pth"
SATMAE_CKPT_URL="https://zenodo.org/records/7338613/files/pretrain-vit-large-e199.pth?download=1"
SATMAE_CKPT_MD5="436bcd815194afdae38f3e023c74e9cd"
SATMAE_CKPT_BYTES=3949597273

# ---- Helpers ----------------------------------------------------------------
# Download URL to FILE unless FILE already has the expected sha256; stop on a mismatch.
poc_fetch() {
    local url="$1" file="$2" sha="$3"
    if [[ -f "$file" ]] && [[ "$(sha256sum "$file" | cut -d' ' -f1)" == "$sha" ]]; then
        echo "   already present, sha256 OK: $(basename "$file")"
        return 0
    fi
    mkdir -p "$(dirname "$file")"
    curl -fL --retry 5 --retry-delay 10 --silent --show-error -o "$file.part" "$url"
    if [[ "$(sha256sum "$file.part" | cut -d' ' -f1)" != "$sha" ]]; then
        rm -f "$file.part"
        echo "ERROR: sha256 mismatch for $(basename "$file"); removed the download." >&2
        return 1
    fi
    mv "$file.part" "$file"
    echo "   downloaded, sha256 OK: $(basename "$file")"
}

# True when the Clay recipe's setup jobs (environment, checkpoint, DINOv2
# cache, Clay source) have completed for this CLAY_WORKDIR. The environment
# check is the same stamp clay_build_env.sbatch writes.
poc_clay_ready() {
    local stamp
    stamp=$(cat "$INFER_DIR/scripts/clay_environment.yml" "$INFER_DIR/scripts/clay_requirements.txt" \
                <(echo "$CLAY_COMMIT") | sha256sum | cut -d' ' -f1)
    [[ -f "$CLAY_ENV/.clay_env_complete" && "$(cat "$CLAY_ENV/.clay_env_complete")" == "$stamp" ]] &&
    [[ -f "$CLAY_WEIGHTS/clay-v1.5.ckpt" && -f "$CLAY_REPO/configs/metadata.yaml" ]] &&
    [[ -d "$HF_HOME/hub/models--timm--vit_large_patch14_reg4_dinov2.lvd142m" ]]
}

# Activate the SatMAE environment (same set -u handling as clay_activate).
poc_satmae_activate() {
    clay_load_conda || return 1
    if [[ ! -f "$SATMAE_ENV/.poc_env_complete" ]]; then
        echo "ERROR: no finished SatMAE environment at $SATMAE_ENV. Run: sbatch jobs/2_setup_satmae.sbatch" >&2
        return 1
    fi
    local nounset=$-
    set +u
    conda activate "$SATMAE_ENV"
    if [[ $nounset == *u* ]]; then set -u; fi
}

# Which code ran: commit, and the Git hash of this folder's files.
poc_code_version() {
    echo "commit $(git rev-parse --short HEAD)  folder tree $(git rev-parse HEAD:./)  uncommitted code changes: $(git status --porcelain -- . ':(exclude)results' | wc -l | tr -d ' ')"
}
