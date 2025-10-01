#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           # <--- Esegui 4 processi per nodo
#SBATCH --cpus-per-task=1             # Un CPU core per processo (controlla i requisiti CLAP)
#SBATCH --time=01:00:00
#SBATCH --exclusive                   # O --exclusive, o --gpus-per-task=1
#SBATCH --gres=gpu:4                   # Richiedi 4 GPU per il nodo
#SBATCH -A IscrC_Pb-skite
#SBATCH -p boost_usr_prod

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close
# Setta MASTER_ADDR e MASTER_PORT per la comunicazione DDP
# SLURM_JOB_NODELIST ti darà l'hostname del nodo
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500


# =================================================================
# 1. CONFIGURAZIONE DEI PERCORSI SORGENTE
# =================================================================

# Percorso sul disco di rete (leonardo_scratch)
SOURCE_BASE_DIR="/leonardo_scratch/large/userexternal/lpilegg1"

# Ambiente virtuale
VENV_SOURCE="${SOURCE_BASE_DIR}/SEC_pipeline/.venv"
# Pesi CLAP
WEIGHTS_SOURCE="${SOURCE_BASE_DIR}/clap_weights/CLAP_weights_2023.pth"
# Text Encoder (è una directory)
TEXT_ENCODER_SOURCE="${SOURCE_BASE_DIR}/clap_text_encoder/roberta-base"
# Test raw dataset
ROOT_SOURCE_PATH="${SOURCE_BASE_DIR}/dataSEC/testing/RAW_DATASET"
# Test preprocessed dataset folder
FINAL_TARGET_PATH="${SOURCE_BASE_DIR}/dataSEC/testing/PREPROCESSED_DATASET"


# =================================================================
# 2. CONFIGURAZIONE DEI PERCORSI DESTINAZIONE (su /tmp locale)
# =================================================================

# La directory TMPDIR punta al disco locale (NVMe o RAM-disk)
TEMP_BASE_DIR="/tmp/$USER/SEC_job_run"
mkdir -p "$TEMP_BASE_DIR"

VENV_DEST="${TEMP_BASE_DIR}/.venv"
WEIGHTS_DEST="${TEMP_BASE_DIR}/CLAP_weights_2023.pth"
TEXT_ENCODER_DEST="${TEMP_BASE_DIR}/roberta-base"


# =================================================================
# 3. COPIA DEI FILE (Aggiunge l'overhead di pochi secondi)
# =================================================================

echo "Inizio fase di copia dei dati su disco locale ($TEMP_BASE_DIR)..."
COPY_START_TIME=$(date +%s)

# A. Copia VENV (Risolve i 5 minuti di import latency)
if [ ! -d "$VENV_DEST" ]; then
    echo "  Copia ambiente virtuale in corso..."
    cp -R "$VENV_SOURCE" "$VENV_DEST"
fi

# B. Copia Pesi CLAP (Risolve i 10 minuti di I/O contention)
if [ ! -f "$WEIGHTS_DEST" ]; then
    echo "  Copia pesi CLAP in corso..."
    cp "$WEIGHTS_SOURCE" "$WEIGHTS_DEST"
fi

# C. Copia Text Encoder (Risolve i 10 minuti di I/O contention)
if [ ! -d "$TEXT_ENCODER_DEST" ]; then
    echo "  Copia Text Encoder in corso..."
    cp -R "$TEXT_ENCODER_SOURCE" "$TEXT_ENCODER_DEST"
fi

# D. Copia Raw Dataset (Risolve latenze in operazioni I/O con singoli audio)
if [ ! -d "${TEMP_BASE_DIR}/RAW_DATASET" ]; then
    echo "  Copia Raw Dataset in corso..."
    cp -r $ROOT_SOURCE_PATH ${TEMP_BASE_DIR}/RAW_DATASET
fi

# E. Copia Preprocessed Dataset Folder (Risolve latenze in operazioni I/O di salvataggio)
if [ ! -d "${TEMP_BASE_DIR}/PREPROCESSED_DATASET" ]; then
    echo "  Copia Preprocessed Dataset in corso..."
    cp -r $FINAL_TARGET_PATH ${TEMP_BASE_DIR}/PREPROCESSED_DATASET
fi

COPY_END_TIME=$(date +%s)
COPY_DURATION=$((COPY_END_TIME - COPY_START_TIME))
echo "Fase di copia terminata in $COPY_DURATION secondi."


# =================================================================
# 4. IMPOSTAZIONE AMBIENTE E LANCIO DEL JOB
# =================================================================

# Attiva il VENV COPIATO LOCALMENTE
source "$VENV_DEST/bin/activate"

# Esporta i nuovi percorsi locali per il codice Python
export LOCAL_CLAP_WEIGHTS_PATH="$WEIGHTS_DEST"
export LOCAL_TEXT_ENCODER_PATH="$TEXT_ENCODER_DEST"
export HF_HOME="$TEMP_BASE_DIR/huggingface_cache" # Suggerimento: sposta anche la cache HF locale per evitare lookup lenti
export NODE_TEMP_BASE_DIR="$TEMP_BASE_DIR"

echo "Variabili d'ambiente impostate. Inizio esecuzione script Python."

# Esegui il tuo programma Python (dove il venv locale e i percorsi locali vengono utilizzati)
embed_test="./tests/test_embeddings.py"
# Avvia ogni processo con srun, che imposterà RANK e WORLD_SIZE
srun python3 "$embed_test"


# Pulizia
deactivate
# Non è necessario pulire /tmp alla fine, lo fa il sistema (o il job manager)
