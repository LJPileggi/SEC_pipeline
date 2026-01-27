#!/bin/bash
#SBATCH --job-name=test_slime_core
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=test_slime_core_%j.out

# --- 1. CONFIGURATION ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Isolated workspace on local scratch
TEST_WORK_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_slime_core_$SLURM_JOB_ID"
mkdir -p "$TEST_WORK_DIR/roberta-base" "$TEST_WORK_DIR/weights" "$TEST_WORK_DIR/numba_cache"

# Automatic cleanup on completion or signal
cleanup() {
    echo "🧹 Cleaning up test environment: $TEST_WORK_DIR"
    rm -rf "$TEST_WORK_DIR"
}
trap cleanup EXIT SIGTERM SIGINT

# --- 2. ASSET PREPARATION ---
echo "📦 Staging assets to local scratch..."
cp -r "$ROBERTA_PATH/." "$TEST_WORK_DIR/roberta-base/"
cp "$CLAP_WEIGHTS" "$TEST_WORK_DIR/weights/CLAP_weights_2023.pth"

# Environment variables for the container context
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export TEST_WEIGHTS_PATH="/tmp_data/weights/dummy_classifier.pt"

echo "🚀 Launching SLIME Core Logic Test..."

# --- 3. EXECUTION ---
singularity exec --nv \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEST_WORK_DIR:/tmp_data" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 tests/utils/test_slime_logic.py
