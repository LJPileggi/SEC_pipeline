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

# Define job-isolated high-speed scratch directories[cite: 19]
TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_train_$SLURM_JOB_ID"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
mkdir -p "$TEMP_DIR/numba_cache"

cleanup_job_scratch() {
    trap - SIGTERM SIGINT
    echo "⚠️ Settle process triggered. Purging temporary scratch space..."
    if [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
    exit 0
}
trap 'cleanup_job_scratch' SIGTERM SIGINT

echo "📦 Stage-in: Moving global WAV HDF5 datasets and model weights checkpoint..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$(expr 20000 + ${SLURM_JOB_ID} % 10000)
export VERBOSE=False

echo "🚀 Launching Distributed Training on 4 ranks (src/filterbank_diffusion/pipeline/train.py)..."
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \
    singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 -m src/filterbank_diffusion/pipeline/train

echo "🔬 Launching Standalone Reconstruction Validation..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 -m src/filterbank_diffusion/pipeline/validate

cleanup_job_scratch
