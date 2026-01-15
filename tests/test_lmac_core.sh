#!/bin/bash
# ./tests/test_lmac_core.sh

# --- 1. CONFIGURATION ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"

# Creazione del workspace unico su scratch
TEST_WORK_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_lmac_test_$$"
mkdir -p "$TEST_WORK_DIR/roberta-base"
mkdir -p "$TEST_WORK_DIR/weights"

# 🎯 INSERISCI QUI: Configurazione Cache Numba
# Creiamo la cartella fisica nello scratch temporaneo
NUMBA_CACHE_PHYSICAL_DIR="$TEST_WORK_DIR/numba_cache"
mkdir -p "$NUMBA_CACHE_PHYSICAL_DIR"

# Esportiamo la variabile che dirotta Numba verso il mount point del container
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"

# --- 2. PREPARAZIONE ASSET ---
cp -r "$ROBERTA_PATH/." "$TEST_WORK_DIR/roberta-base/"
cp "$CLAP_WEIGHTS" "$TEST_WORK_DIR/weights/CLAP_weights_2023.pth"

# Mock weights per FinetunedModel
export TEST_WEIGHTS_PATH="/tmp_data/weights/dummy_classifier.pt"

# Variabili per Offline Patching
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"

echo "🚀 Launching Full L-MAC Functional Test with Numba redirect..."

# --- 3. EXECUTION ---
# Nota: TEST_WORK_DIR è montato su /tmp_data, quindi Numba scriverà 
# in /tmp_data/numba_cache che corrisponde a scratch.
singularity exec --nv \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEST_WORK_DIR:/tmp_data" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 tests/utils/test_lmac_logic.py

# --- 4. CLEANUP ---
rm -rf "$TEST_WORK_DIR"
echo "✅ Test environment and Numba cache cleaned up."
