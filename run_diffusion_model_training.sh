#!/bin/bash
#SBATCH --job-name=unet_train_dist
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=06:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# Define job-isolated high-speed scratch directories
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_train_$SLURM_JOB_ID"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/clap_bn0_constants.npz"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

# 🎯 1. DESTINAZIONE PERSISTENTE DEI CHECKPOINT SU SCRATCH
export MODELS_GLOBAL="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.models/diff_model"

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"
mkdir -p "$TEMP_DIR/models/diff_model"
mkdir -p "$MODELS_GLOBAL"

# 🎯 2. STAGE-OUT PERSISTENTE DEI CHECKPOINT PRIMA DELLA PULIZIA
cleanup_job_scratch() {
    trap - SIGTERM SIGINT
    echo "⚠️ Settle process triggered. Saving checkpoints to global SCRATCH..."
    if [ -d "$TEMP_DIR" ]; then
        if [ -d "$TEMP_DIR/models/diff_model" ]; then
            echo "📦 Stage-out: Syncing model checkpoints to $MODELS_GLOBAL..."
            rsync -rlt "$TEMP_DIR/models/diff_model/" "$MODELS_GLOBAL/"
        fi
        echo "🧹 Purging temporary scratch space..."
        rm -rf "$TEMP_DIR"
    fi
    exit 0
}
trap 'cleanup_job_scratch' SIGTERM SIGINT

echo "📦 Stage-in: Moving global WAV HDF5 datasets and model weights checkpoint..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

# Se esistono già dei checkpoint precedenti su SCRATCH, li ripristiniamo nel TEMP_DIR
[ -d "$MODELS_GLOBAL" ] && cp -r "$MODELS_GLOBAL/." "$TEMP_DIR/models/diff_model/" 2>/dev/null

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/work_dir/weights/clap_bn0_constants.npz"

# 🎯 3. DICHIARIAMO DOVE SALVARE E CERCARE I CHECKPOINT
export MODEL_CHECKPOINT_DIR="/tmp_data/models/diff_model"

export INJECT_OCTAVE="True"
export VERBOSE="False"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 20000 + ${SLURM_JOB_ID} % 10000)

# echo "🚀 Launching Distributed Training on 4 ranks (src/filterbank_diffusion/pipeline/train.py)..."
# srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \
#     singularity exec --nv --no-home \
#     --bind "/leonardo_scratch:/leonardo_scratch" \
#     --bind "$TEMP_DIR:/tmp_data" \
#     --bind "$(pwd):/app" --pwd "/app" \
#     "$SIF_FILE" \
#     python3 src/filterbank_diffusion/pipeline/train.py

# 🎯 4. STAGE-OUT IMMEDIATO A FINE TRAINING (Prima della Validation)
# echo "📦 Syncing trained checkpoints to global SCRATCH before validation..."
# rsync -rlt "$TEMP_DIR/models/diff_model/" "$MODELS_GLOBAL/"

echo "🔬 Launching Standalone Reconstruction Validation..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 src/filterbank_diffusion/pipeline/validate.py

cleanup_job_scratch
