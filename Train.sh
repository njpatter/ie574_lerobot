#!/usr/bin/env bash
#SBATCH --job-name=lerobot-train
#SBATCH --partition=teaching
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8 
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# ### Ignore this file, I use it for submitting training to the compute/gpu cluster ###

set -euo pipefail

# Load secrets from .env if present.
[ -f "$(dirname "$0")/.env" ] && source "$(dirname "$0")/.env"

# Print date and time for logging.
echo "[INFO] Starting job at $(date)" 

SIF_IMAGE="${SIF_IMAGE:-lerobot_gpu.sif}"
HOST_DATASET_DIR="${HOST_DATASET_DIR:-$PWD/record-test}"
CONTAINER_DATASET_DIR="${CONTAINER_DATASET_DIR:-/data}"
HOST_CACHE_DIR="${HOST_CACHE_DIR:-$PWD/.cache/lerobot}"
CONTAINER_CACHE_DIR="${CONTAINER_CACHE_DIR:-/tmp/lerobot_cache}"

POLICY_TYPE="${POLICY_TYPE:-act}"
OUTPUT_DIR="${OUTPUT_DIR:-$PWD/outputs/train_act}"
JOB_NAME="${JOB_NAME:-act_train}"
RUN_NAME="${RUN_NAME:-${JOB_NAME}_${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}"

# If true, allows continuing an existing run directory.
RESUME="${RESUME:-false}"

# For local datasets, setting repo_id to the mounted path is convenient.
DATASET_REPO_ID="${DATASET_REPO_ID:-$CONTAINER_DATASET_DIR}"

# Optional W&B values.
WANDB_API_KEY="${WANDB_API_KEY:-}"
WANDB_ENABLE="${WANDB_ENABLE:-true}"
WANDB_PROJECT="${WANDB_PROJECT:-lerobot}"

# Force a non-torchcodec decoder backend on HPC nodes.
VIDEO_BACKEND="${VIDEO_BACKEND:-pyav}"

STEPS="${STEPS:-100000}"
BATCH_SIZE="${BATCH_SIZE:-8}"

mkdir -p logs "$OUTPUT_DIR" "$HOST_CACHE_DIR"

echo "[INFO] Job ID: ${SLURM_JOB_ID:-local}"
echo "[INFO] Node: $(hostname)"
echo "[INFO] Image: $SIF_IMAGE"
echo "[INFO] Host dataset: $HOST_DATASET_DIR"
echo "[INFO] Container dataset mount: $CONTAINER_DATASET_DIR"
echo "[INFO] Host cache dir: $HOST_CACHE_DIR"
echo "[INFO] Container cache dir: $CONTAINER_CACHE_DIR"
echo "[INFO] Video backend: $VIDEO_BACKEND"
echo "[INFO] Run name: $RUN_NAME"
echo "[INFO] Container output dir: /outputs/$RUN_NAME"

singularity exec --nv \
  -B "${HOST_DATASET_DIR}:${CONTAINER_DATASET_DIR}" \
  -B "${OUTPUT_DIR}:/outputs" \
  -B "${HOST_CACHE_DIR}:${CONTAINER_CACHE_DIR}" \
  -B "${HOST_CACHE_DIR}:/home/user_lerobot/.cache" \
  "$SIF_IMAGE" bash -c "
    export XDG_CACHE_HOME='$CONTAINER_CACHE_DIR'
    export HF_HOME='$CONTAINER_CACHE_DIR/huggingface'
    export HF_DATASETS_CACHE='$CONTAINER_CACHE_DIR/huggingface/datasets'
    export WANDB_API_KEY='$WANDB_API_KEY'
    lerobot-train \\
      --dataset.repo_id='$DATASET_REPO_ID' \\
      --dataset.video_backend='$VIDEO_BACKEND' \\
      --policy.type='$POLICY_TYPE' \\
      --output_dir='/outputs/$RUN_NAME' \\
      --job_name='$JOB_NAME' \\
      --resume='$RESUME' \\
      --policy.device=cuda \\
      --wandb.enable='$WANDB_ENABLE' \\
      --wandb.project='$WANDB_PROJECT' \\
      --policy.push_to_hub=false \\
      --steps='$STEPS' \\
      --batch_size='$BATCH_SIZE'
  "

echo "[INFO] Job finished at $(date)"