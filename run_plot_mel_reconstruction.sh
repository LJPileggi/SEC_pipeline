#!/bin/bash
#SBATCH --job-name=plot_mel_recon
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=01:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_plot_$SLURM_JOB_ID"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"
DIFFUSION_MODELS_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.models/diff_model"

# Frazione di ottava da plottare (es. 3 per 1/3 d'ottava, o 32 per 1/32)
export TARGET_FRACTION=3

export CHECKPOINT_PATH="$DIFFUSION_MODELS_PATH/unet_epoch_499.pt"

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav"
mkdir -p "$TEMP_DIR/work_dir/weights"
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

echo "📦 Stage-in: Copia dati per il plotting dei mel..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/dataSEC/RAW_DATASET/raw_wav/" 2>/dev/null

export BASEDIR="$TEMP_DIR"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export MODELS_GLOBAL="$DIFFUSION_MODELS_PATH"
export INJECT_OCTAVE="True"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1

echo "🎨 Esecuzione script di plotting..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 src/filterbank_diffusion/pipeline/plot_mel_reconstructions.py

cleanup_job_scratch
