#!/bin/bash
#SBATCH --job-name=plot_mel_vae
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Argomenti da riga di comando:
# $1: TARGET_FRACTION (default: 3)
# $2: TARGET_EPOCH (es. 50 -> carica epoca 49. Default: "" ovvero latest)
# $3: BOTTLENECK_TYPE (default: 2d)
export TARGET_FRACTION=${1:-3}
TARGET_EPOCH=$2
export BOTTLENECK_TYPE=${3:-"2d"}

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_plot_vae_$SLURM_JOB_ID"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="$PROJECT_DIR/.clap_weights/clap_bn0_constants.npz"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"
VAE_MODELS_PATH="$PROJECT_DIR/.models/vae_model"

export MODELS_GLOBAL="$VAE_MODELS_PATH"
export MODEL_CHECKPOINT_DIR="/tmp_data/models/vae_model"

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/models/vae_model"
mkdir -p "$TEMP_DIR/numba_cache"

cleanup_job_scratch() {
    trap - SIGTERM SIGINT
    echo "🧹 Pulizia scratch space temporaneo..."
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
    exit 0
}
trap 'cleanup_job_scratch' SIGTERM SIGINT

echo "=========================================================="
echo "🎯 VAE Plotting Configuration:"
echo "   • Octave Fraction:  1/$TARGET_FRACTION"
echo "   • Bottleneck Type:  $BOTTLENECK_TYPE"
if [ -n "$TARGET_EPOCH" ] && [ "$TARGET_EPOCH" != "latest" ]; then
    CKPT_EPOCH=$(( TARGET_EPOCH - 1 ))
    echo "   • Target Epoch:     $TARGET_EPOCH (Loading dual_vae_epoch_${CKPT_EPOCH}.pt)"
    export CHECKPOINT_PATH="$TEMP_DIR/models/vae_model/dual_vae_epoch_${CKPT_EPOCH}.pt"
else
    echo "   • Target Epoch:     Latest available checkpoint"
    export CHECKPOINT_PATH=""
fi
echo "=========================================================="

echo "📦 Stage-in: Copia pesi e dataset..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

# Stage-in dei modelli VAE da MODELS_GLOBAL
if [ -d "$MODELS_GLOBAL" ]; then
    echo "📂 Stage-in dei checkpoint VAE da $MODELS_GLOBAL..."
    cp -r "$MODELS_GLOBAL/." "$TEMP_DIR/models/vae_model/" 2>/dev/null
fi

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export LATENT_DIM=512

export INJECT_OCTAVE="True"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1

echo "🎨 Esecuzione script di plotting VAE..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$PROJECT_DIR:/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 src/filterbank_vae/pipeline/plot_mel_reconstruction.py

cleanup_job_scratch
