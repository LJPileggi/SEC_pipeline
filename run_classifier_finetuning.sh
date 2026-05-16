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
BASEDIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASEDIR}/SEC_pipeline"
DATASEC_DIR="${BASEDIR}/dataSEC"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
CONFIG_FILE="${PROJECT_DIR}/configs/config0.yaml"

# Model paths
MODEL_DIR="${PROJECT_DIR}/.models"
PRETRAINED_MODEL="${MODEL_DIR}/finetuned_model_Adam_0.01_7_secs.torch"
local_suffix=""
if [ "$N_OCTAVE" -ne 0 ] && [ "$INJECT_OCTAVE_CMD" = "False" ]; then
    local_suffix="_no_inject"
fi

export FINAL_MODEL_PATH="${MODEL_DIR}/finetuned_model_RECOVERY_${CUT_SECS}_secs${local_suffix}.torch"

# Training parameters (Sbloccati da riga di comando con valori di default)
AUDIO_FORMAT=${1:-"wav"}
N_OCTAVE=${2:-"3"}
INJECT_OCTAVE_CMD=${3:-"True"}
CUT_SECS=${4:-7}

# Forzatura logica di coerenza
if [ "$N_OCTAVE" -eq 0 ]; then
    INJECT_OCTAVE_CMD="False"
fi

# --- 2. RUN TRAINING ---
echo "🚀 Starting Finetune Recovery (Target: ${FINAL_MODEL_PATH})..."

# Bind necessari: 
# - PROJECT_DIR per lo script e i moduli src
# - DATASEC_DIR per caricare gli embeddings HDF5
export INJECT_OCTAVE="$INJECT_OCTAVE_CMD"

singularity exec --nv --no-home \
    --bind "${BASEDIR}:/app" \
    --bind "${PROJECT_DIR}:/app/${PROJECT_DIR}" \
    --bind "${DATASEC_DIR}:/app/${DATASEC_DIR}" \
    --pwd "/app/${PROJECT_DIR}" \
    "$SIF_FILE" \
    python3 scripts/train_finetuned_classifier.py \
        --config_file "$CONFIG_FILE" \
        --audio_format "$AUDIO_FORMAT" \
        --n_octave "$N_OCTAVE" \
        --cut_secs "$CUT_SECS" \
        --pretrained_path "$PRETRAINED_MODEL"

echo "✅ Training completed. Check ${FINAL_MODEL_PATH} for results."
