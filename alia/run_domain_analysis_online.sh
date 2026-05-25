#!/bin/bash

# --- 1. CONFIGURAZIONE DEI PATH BASE ---
MY_USER=$(whoami)
BASEDIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASEDIR}/SEC_pipeline"
DATASEC_DIR="${BASEDIR}/dataSEC"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
CONFIG_FILE="${PROJECT_DIR}/configs/config0.yaml"

export BASEDIR="$BASEDIR"

# 🎯 PARAMETRI COMPILATI SUI PESI DI FABBRICA (Ispirati da run_embedding_pipeline.sh)
CLAP_SCRATCH_WEIGHTS="${PROJECT_DIR}/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="${PROJECT_DIR}/.clap_weights/roberta-base"

# --- 2. VINCOLI ATOMICI PER IL NODO DI LOGIN (Risorse Safe) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" # Esclude totalmente CUDA per evitare eccezioni driver

# 💉 🎯 ESPORTAZIONE VARIABILI GLOBALI PER LA REDIREZIONE LOCALE
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export INJECT_OCTAVE="False" # Ci serve il preprocessore nativo intatto per estrarre la GT

# 🎯 SOLUZIONE CRITICA: Devia la cache di Numba in una cartella locale con permessi di scrittura
export NUMBA_CACHE_DIR="${PROJECT_DIR}/.numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

# Colleghiamo i percorsi reali dello scratch che la patch Python andrà a intercettare
export LOCAL_CLAP_WEIGHTS_PATH="$CLAP_SCRATCH_WEIGHTS"
export CLAP_TEXT_ENCODER_PATH="$ROBERTA_PATH"

echo "⏳ Esecuzione Analisi di Dominio Online (Modalità Interattiva Offline)..."
echo "   • Pesi CLAP: $LOCAL_CLAP_WEIGHTS_PATH"
echo "   • Testo RoBERTa: $CLAP_TEXT_ENCODER_PATH"

# --- 3. RUN PIPELINE ONLINE ---
# Manteniamo l'esatta struttura dei tuoi bind originali per non spaccare i riferimenti
singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 alia/evaluate_domain_distance_online.py \
        --config_file "$CONFIG_FILE" \
        --n_octave "3" \
        --audio_format "wav" \
        --cut_secs 7 \
        --samples_per_class 50

echo "✅ Analisi conclusa. Controlla la cartella results/domain_analysis_online/"
