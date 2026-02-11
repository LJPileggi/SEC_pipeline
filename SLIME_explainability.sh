#!/bin/bash
#SBATCH --job-name=SLIME_expl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=SLIME_expl_%j.out

# ==============================================================================
# USAGE INSTRUCTIONS:
# Launch this script using sbatch with the --export flag to pass parameters:
#
# sbatch --export=ALL,\
# IDS_FILE="samples.txt",\
# FORMAT="wav",\
# OCTAVE=1,\
# SECS=3,\
# WEIGHTS="/path/to/classifier.pt",\
# EXPL_TYPE="time_frequency",\
# SAMPLES=1000 \
# SLIME_explainability_slurm.sh
#
# Mandatory variables: IDS_FILE, FORMAT, OCTAVE, SECS, WEIGHTS
# Optional variables: CONFIG, SPLIT, EXPL_TYPE (default: time_frequency), SAMPLES (default: 1000)
# ==============================================================================

# --- 1. CONFIGURATION ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Workspace isolation
SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_slime_expl_$SLURM_JOB_ID"
CONTAINER_WORK_DIR="/tmp_data"

mkdir -p "$SCRATCH_TEMP_DIR/work_dir/roberta-base" "$SCRATCH_TEMP_DIR/work_dir/weights" "$SCRATCH_TEMP_DIR/numba_cache"

cleanup() {
    echo "🧹 Cleaning up workspace: $SCRATCH_TEMP_DIR"
    rm -rf "$SCRATCH_TEMP_DIR"
}
trap cleanup EXIT SIGTERM SIGINT

# --- 2. ASSET PREPARATION (Mantra) ---
echo "📦 Staging assets to local scratch..."
cp -r "$ROBERTA_PATH/." "$SCRATCH_TEMP_DIR/work_dir/roberta-base/"
cp "$CLAP_WEIGHTS" "$SCRATCH_TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
cp "$WEIGHTS" "$SCRATCH_TEMP_DIR/work_dir/weights/classifier_to_explain.pt"

# --- 3. ENVIRONMENT VARIABLES ---
export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="$CONTAINER_WORK_DIR/work_dir/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/work_dir/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="$CONTAINER_WORK_DIR/numba_cache"
export PYTHONPATH="/app"
export NODE_TEMP_BASE_DIR="$CONTAINER_WORK_DIR/dataSEC"

# --- 4. EXECUTION (Exact mapping of SLIME_pipeline.py parser) ---
echo "🚀 Launching SLIME Production Pipeline..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$SCRATCH_TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 scripts/SLIME_pipeline.py \
        --ids_file "$IDS_FILE" \
        --config_file "${CONFIG:-config0.yaml}" \
        --audio_format "$FORMAT" \
        --n_octave "$OCTAVE" \
        --cut_secs "$SECS" \
        --split "${SPLIT:-valid}" \
        --weights_path "$CONTAINER_WORK_DIR/work_dir/weights/classifier_to_explain.pt" \
        --expl_type "${EXPL_TYPE:-time_frequency}" \
        --n_samples "${SAMPLES:-1000}"
