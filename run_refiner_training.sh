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
VAE_TARGET_EPOCH=${2:-${VAE_TARGET_EPOCH:-50}}

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_train_refiner_$SLURM_JOB_ID"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="$PROJECT_DIR/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="$PROJECT_DIR/.clap_weights/clap_bn0_constants.npz"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"
MODELS_GLOBAL="$PROJECT_DIR/.models/vae_model"

# Risoluzione del singolo file di checkpoint VAE desiderato (gestione zero-index: epoca 50 -> file _49 o _50)
TARGET_FILE=""
VAE_IDX=$((VAE_TARGET_EPOCH - 1))
if [ -f "$MODELS_GLOBAL/dual_vae_epoch_${VAE_IDX}.pt" ]; then
    TARGET_FILE="$MODELS_GLOBAL/dual_vae_epoch_${VAE_IDX}.pt"
elif [ -f "$MODELS_GLOBAL/dual_vae_epoch_${VAE_TARGET_EPOCH}.pt" ]; then
    TARGET_FILE="$MODELS_GLOBAL/dual_vae_epoch_${VAE_TARGET_EPOCH}.pt"
else
    echo "❌ [ERROR] Impossibile trovare il checkpoint VAE per epoca ${VAE_TARGET_EPOCH} in $MODELS_GLOBAL!"
    exit 1
fi

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"
mkdir -p "$TEMP_DIR/single_vae_model"
mkdir -p "$PROJECT_DIR/.models/refiner_model"

cleanup_job_scratch() {
    trap - SIGTERM SIGINT
    echo "🧹 Purging temporary scratch space..."
    rm -rf "$TEMP_DIR"
    exit 0
}
trap 'cleanup_job_scratch' SIGTERM SIGINT

echo "=========================================================="
echo "🎯 Refiner Training Launcher"
echo "   • Refiner Epochs:     $NUM_EPOCHS"
echo "   • VAE Source Epoch:   $VAE_TARGET_EPOCH"
echo "   • Single VAE Staged:  $TARGET_FILE"
echo "   • Refiner Save Dir:   $PROJECT_DIR/.models/refiner_model"
echo "=========================================================="

echo "📦 Stage-in: Staging datasets, pesi CLAP e SINGOLO checkpoint VAE..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/work_dir/weights/clap_bn0_constants.npz"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null
cp "$TARGET_FILE" "$TEMP_DIR/single_vae_model/target_dual_vae.pt"

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"
export LOCAL_VAE_CHECKPOINT="/tmp_data/single_vae_model/target_dual_vae.pt"
export REFINER_CHECKPOINT_DIR="/app/.models/refiner_model"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 22000 + ${SLURM_JOB_ID} % 10000)

echo "🚀 Launching Refiner Training on 4 ranks..."
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \
    singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$PROJECT_DIR:/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 src/filterbank_refiner/pipeline/train.py \
        --epochs "$NUM_EPOCHS" \
        --vae_checkpoint "/tmp_data/single_vae_model/target_dual_vae.pt"

cleanup_job_scratch
