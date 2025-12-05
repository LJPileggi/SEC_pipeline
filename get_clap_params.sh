#!/bin/bash
#
# Script RIVISTO. Ripristina la logica di base del tuo script funzionante, 
# ma con diagnostica aggressiva sul path di origine.

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA 
# ----------------------------------------------------------------------
# Forza la variabile USER.
USER="lpilegg1" 
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"

# 🎯 VERIFICA INIZIALE CRITICA: Assicurati che il file ORIGINALE esista
if [ ! -f "$CLAP_SCRATCH_WEIGHTS" ]; then
    echo "❌ ERRORE DI ORIGINE: Impossibile trovare i pesi CLAP originali al percorso: $CLAP_SCRATCH_WEIGHTS. Controlla il path o i permessi."
    exit 1
fi

# Variabili di Path
SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_data_pipeline_$$"
TEMP_DIR="$SCRATCH_TEMP_DIR/work_dir" 

# Percorsi interni al container.
CONTAINER_SCRATCH_BASE="/scratch_base" 
CONTAINER_WORK_DIR="/app/temp_work"
TEMP_SCRIPT_NAME="clap_inspector_script.py"

# --- 1. PREPARAZIONE DATI SULLO SCRATCH ---

echo "--- 🛠️ Preparazione Dati Temporanei su Scratch ($SCRATCH_TEMP_DIR) ---"

# 1.1. Creazione della cartella di lavoro su Scratch
if ! mkdir -p "$TEMP_DIR"; then
    echo "❌ ERRORE CRITICO: Impossibile creare la directory temporanea su Scratch: $TEMP_DIR."
    exit 1
fi

# 1.2. Copia dei pesi CLAP nella cartella di lavoro su Scratch
echo "Copia dei pesi CLAP su Scratch temporanea ($TEMP_DIR)..."
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/CLAP_weights_2023.pth"
if ! cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS"; then
    echo "❌ ERRORE CRITICO: Impossibile copiare i pesi da $CLAP_SCRATCH_WEIGHTS a $CLAP_LOCAL_WEIGHTS."
    rm -rf "$SCRATCH_TEMP_DIR"
    exit 1
fi

# 1.3. Diagnostica di Esistenza del file copiato
if [ ! -f "$CLAP_LOCAL_WEIGHTS" ]; then
    echo "❌ DIAGNOSTICA FINALE: Il file dei pesi ($CLAP_LOCAL_WEIGHTS) NON ESISTE SULL'HOST dopo la copia. La copia è fallita."
    rm -rf "$SCRATCH_TEMP_DIR"
    exit 1
fi


# --- 2. CONFIGURAZIONE ESECUTIVA (Variabili d'Ambiente) ---

echo "--- ⚙️ Configurazione Ambiente di Esecuzione ---"

# Il percorso dei pesi CLAP DEVE essere il PERCORSO INTERNO AL CONTAINER.
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/CLAP_weights_2023.pth"
# Percorso per l'encoder testuale (preso dall'interno del container)
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 
export NODE_TEMP_BASE_DIR="$CONTAINER_SCRATCH_BASE/dataSEC" 

# --- 3. CREAZIONE DELLO SCRIP PYTHON TEMPORANEO (Minimale) ---
cat << EOF > "$TEMP_SCRIPT_NAME"
import sys
import os
import logging
import torch
sys.path.append('.') 
from src.models import CLAP_initializer 

logging.basicConfig(level=logging.CRITICAL)

# --------------------------------------------------------------------
# LOGICA DI ISPEZIONE
# --------------------------------------------------------------------
try:
    # 1. Inizializzazione del modello (Firma corretta, usa le variabili d'ambiente)
    clap_model, audio_embedding, _ = CLAP_initializer(device='cpu', use_cuda=False) 
    
    # 2. Ispezione dei parametri 
    test_attrs = ['mel_transform', 'spectrogram_extractor', 'log_mel_spec', 'spectrogram']
    
    found = False
    for attr_name in test_attrs:
        try:
            mel_module = getattr(audio_embedding, attr_name)
            
            N_FFT = getattr(mel_module, 'n_fft', None)
            HOP_LENGTH = getattr(mel_module, 'hop_length', None)
            N_MELS = getattr(mel_module, 'n_mels', None)
            SR = getattr(mel_module, 'sr', None)
            
            if N_FFT is not None and HOP_LENGTH is not None and N_MELS is not None and SR is not None:
                print("--------------------------------------------------")
                print("✅ PARAMETRI CLAP REALI TROVATI!")
                print(f"N_FFT: {N_FFT}")
                print(f"HOP_LENGTH: {HOP_LENGTH}")
                print(f"N_MELS: {N_MELS}")
                print(f"SR: {SR} Hz")
                print("--------------------------------------------------")
                found = True
                break
                
        except AttributeError:
            continue
    
    if not found:
        print("❌ FALLIMENTO: Parametri Mel Spectrogram non trovati sull'encoder CLAP.")

except Exception as e:
    print(f"❌ ERRORE CRITICO DURANTE L'INIZIALIZZAZIONE: {e}")

# --------------------------------------------------------------------
EOF

# --- 4. ESECUZIONE DEL CONTAINER ---

echo "--- 🔍 Esecuzione Script Ispezione Parametri CLAP ---"

# Lancio dello script da /app/clap_inspector_script.py (Path corretto)
singularity exec \
    --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" \
    --bind "$(pwd)":/app \
    --pwd /app \
    "$SIF_FILE" \
    python3 "/app/$TEMP_SCRIPT_NAME"

# --- 5. PULIZIA ---
echo "Pulizia script e directory temporanea su Scratch..."
rm -f "$TEMP_SCRIPT_NAME"
rm -rf "$SCRATCH_TEMP_DIR"
echo "Esecuzione completata."
