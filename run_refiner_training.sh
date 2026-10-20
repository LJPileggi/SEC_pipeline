#!/bin/bash
#SBATCH --job-name=mel_refiner_train
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=03:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

NUM_EPOCHS=${1:-20}

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_train_refiner_$SLURM_JOB_ID"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"
export MODELS_GLOBAL="$PROJECT_DIR/.models/vae_model"

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"
mkdir -p "$PROJECT_DIR/.models/refiner_model"

cleanup_job_scratch() {
    trap - SIGTERM SIGINT
    echo "🧹 Purging temporary scratch space..."
    rm -rf "$TEMP_DIR"
    exit 0
}
trap 'cleanup_job_scratch' SIGTERM SIGINT

echo "📦 Stage-in: Staging datasets e pesi CLAP..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments
