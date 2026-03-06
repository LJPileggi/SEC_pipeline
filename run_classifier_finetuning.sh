#!/bin/bash
#SBATCH --job-name=Finetune_Recovery
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --partition=boost_usr_prod

# --- 1. CONFIGURATION ---
MY_USER=$(whoami)
PROJECT_DIR="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline"
DATASEC_DIR="/leonardo_scratch/large/userexternal/${MY_USER}/dataSEC"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
CONFIG_FILE="${PROJECT_DIR}/configs/config0.yaml"

# Model paths
MODEL_DIR="${PROJECT_DIR}/.models"
PRETRAINED_MODEL="${MODEL_DIR}/finetuned_model_Adam_0.01_7_secs.torch"
export FINAL_MODEL_PATH="${MODEL_DIR}/finetuned_model_RECOVERY_7_secs.torch"

# Training parameters
AUDIO_FORMAT="wav"
N_OCTAVE="3"
CUT_SECS=7

# --- 2. RUN TRAINING ---
echo "🚀 Starting Finetune Recovery (Target: ${FINAL_MODEL_PATH})..."

# Bind necessari: 
# - PROJECT_DIR per lo script e i moduli src
# - DATASEC_DIR per caricare gli embeddings HDF5
singularity exec --nv --no-home \
    --bind "${PROJECT_DIR}:/app" \
    --bind "${DATASEC_DIR}:${DATASEC_DIR}" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 train_finetuned_classifier.py \
        --config_file "$CONFIG_FILE" \
        --audio_format "$AUDIO_FORMAT" \
        --n_octave "$N_OCTAVE" \
        --cut_secs "$CUT_SECS" \
        --pretrained_path "$PRETRAINED_MODEL"

echo "✅ Training completed. Check ${FINAL_MODEL_PATH} for results."
