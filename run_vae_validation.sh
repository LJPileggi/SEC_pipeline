#!/bin/bash
#SBATCH --job-name=vae_val_standalone
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_val_vae_$SLURM_JOB_ID"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="$PROJECT_DIR/.clap_weights/clap_bn0_constants.npz"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

export MODELS_GLOBAL="$PROJECT_DIR/.models/vae_model"

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"

cleanup_job_scratch() {
    trap - SIGTERM SIGINT
    echo "🧹 Purging temporary scratch space..."
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
    exit 0
}
trap 'cleanup_job_scratch' SIGTERM SIGINT

echo "📦 Stage-in: Staging datasets for VAE standalone validation..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export BOTTLENECK_TYPE="2d"
export LATENT_DIM=256
export NUM_EPOCHS=50

export INJECT_OCTAVE="True"
export VERBOSE="False"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1

echo "🔬 Launching Standalone VAE Validation..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$PROJECT_DIR:/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 src/filterbank_vae/pipeline/validate.py

cleanup_job_scratch
