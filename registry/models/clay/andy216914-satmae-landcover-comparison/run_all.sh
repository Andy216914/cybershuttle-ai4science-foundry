#!/bin/bash
# Runs the whole Clay vs. SatMAE comparison on PACE ICE. From this folder:
#
#   bash run_all.sh
#
# 1. Checks that the Clay recipe's setup (environment, checkpoint) exists for
#    this CLAY_WORKDIR; if not, submits that recipe's jobs 1-2 first.
# 2. Asks Slurm when an L40S, A100 or V100 would start (sbatch --test-only)
#    and uses the earliest for the GPU job. Set POC_GPU=l40s (or a100, v100)
#    to choose yourself.
# 3. Submits jobs 1-4 as a dependency chain and returns. Results arrive in
#    results/ (about 30-40 minutes of job time, plus any wait in the queue).
#
# Different Slurm account or partitions: the same variables as the Clay recipe
# (SBATCH_ACCOUNT, SBATCH_QOS, CLAY_CPU_PARTITION, CLAY_GPU_PARTITION).

set -euo pipefail
if [[ ! -f config.sh || ! -d jobs ]]; then echo "ERROR: run this from its own folder." >&2; exit 1; fi
# shellcheck source=config.sh
source config.sh
mkdir -p "$RESULTS/logs"
export CLAY_WORKDIR

submit() {  # submit <job script> [extra sbatch options...]; prints the job ID
    local script="$1"; shift
    clay_sbatch_opts "$script"
    sbatch --parsable ${CLAY_SBATCH_OPTS[@]+"${CLAY_SBATCH_OPTS[@]}"} "$@" "$script"
}

echo "== Clay setup in $CLAY_WORKDIR"
build="" download=""
if poc_clay_ready; then
    echo "   environment and checkpoint present: reused"
else
    echo "   missing: submitting the Clay recipe's setup jobs (environment, downloads)"
    build=$(cd "$INFER_DIR" && submit scripts/clay_build_env.sbatch)
    download=$(cd "$INFER_DIR" && submit scripts/clay_download.sbatch --dependency=afterok:"$build")
    echo "   clay-build-env $build, clay-download $download (logs in $INFER_DIR/benchmarks/logs/)"
fi

echo "== GPU for job 3"
part="${CLAY_GPU_PARTITION:-coc-gpu}"
if [[ -n "${POC_GPU:-}" ]]; then
    gpu="$POC_GPU"
    echo "   gpu:$gpu:1 (chosen with POC_GPU)"
else
    gpu="" best=""
    for g in l40s a100 v100; do
        when=$(sbatch --test-only -p "$part" --gres="gpu:$g:1" jobs/3_extract_features.sbatch 2>&1 |
               grep -oE 'to start at [0-9T:-]+' | awk '{print $4}' || true)
        if [[ -n "$when" ]]; then
            echo "   gpu:$g:1 estimated start $when"
            if [[ -z "$best" || "$when" < "$best" ]]; then best="$when"; gpu="$g"; fi
        else
            echo "   gpu:$g:1 not available in $part"
        fi
    done
    [[ -n "$gpu" ]] || { echo "ERROR: no GPU type accepted in $part; set POC_GPU and CLAY_GPU_PARTITION" >&2; exit 1; }
    echo "   using gpu:$gpu:1"
fi

prep=$(submit jobs/1_prepare_data.sbatch ${build:+--dependency=afterok:$build})
setup=$(submit jobs/2_setup_satmae.sbatch)
feat=$(sbatch --parsable -p "$part" --gres="gpu:$gpu:1" \
       --dependency=afterok:"$prep":"$setup"${download:+:$download} jobs/3_extract_features.sbatch)
probe=$(submit jobs/4_linear_probe.sbatch --dependency=afterok:"$feat")

record="$RESULTS/logs/run_all-$(date +%Y%m%dT%H%M%S).txt"
{
    echo "submitted $(date -Is) by $USER from $(hostname)"
    poc_code_version
    echo "CLAY_WORKDIR $CLAY_WORKDIR"
    echo "clay setup jobs: ${build:-reused} ${download:-}"
    echo "prepare=$prep setup_satmae=$setup extract_features=$feat (gpu:$gpu:1, $part) linear_probe=$probe"
} | tee "$record"
echo
echo "Watch:    squeue -u \$USER"
echo "Logs:     $RESULTS/logs/"
echo "Result:   $RESULTS/results_table.md (after job $probe)"
