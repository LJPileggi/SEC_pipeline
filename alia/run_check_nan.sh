#!/bin/bash

# --- 1. CONFIGURATION ---
MY_USER=$(whoami)
# Definisco il path alla tua cartella preprocessed
export BASEDIR_PATH="/leonardo_scratch/large/userexternal/${MY_USER}"
export PREPROCESSED_DATASET_PATH="${BASEDIR_PATH}/dataSEC/PREPROCESSED_DATASET"

# Path al container SIF
SIF_FILE="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline/.containers/clap_pipeline.sif"

# --- 2. EXECUTION ---
echo "🛠️ Avvio diagnostica NaN sugli embeddings..."
echo "📂 Scanning: $PREPROCESSED_DATASET_PATH"

# Uso singularity exec per lanciare lo script python dentro l'ambiente del container
# Il bind serve a rendere visibile la cartella scratch a Python
singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    "$SIF_FILE" \
    python3 alia/check_nan_embeddings.py

echo -e "\n✅ Analisi completata."
