#!/bin/bash
#
# Script MINIMALE E CORRETTO. Nessuna configurazione YAML, solo inizializzazione CLAP.

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA (Verifica i percorsi)
# ----------------------------------------------------------------------
# Forza la variabile USER.
USER="lpilegg1" 
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS_HOST_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
TEMP_SCRIPT_NAME="clap_inspector_$$.py"

# --- 1. CONFIGURAZIONE AMBIENTE INTERNO AL CONTAINER (Cruciale!) ---
# Directory di lavoro nel container per i pesi.
CONTAINER_WORK_DIR="/app/temp_work"
# Path interno al container.
PESI_CLAP_INTERNI="$CONTAINER_WORK_DIR/CLAP_weights_2023.pth" 

# Imposta le variabili d'ambiente che models.py si aspetta.
export LOCAL_CLAP_WEIGHTS_PATH="$PESI_CLAP_INTERNI" 
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 

# 1.1. Prepara la directory di lavoro temporanea su host
TEMP_DIR_ON_HOST="$(pwd)/temp_clap_inspect_$USER"
mkdir -p "$TEMP_DIR_ON_HOST"

# 1.2. Copia i pesi (necessario per montarli come file locale per il container)
echo "Copia temporanea dei pesi CLAP su $TEMP_DIR_ON_HOST..."
cp "$CLAP_WEIGHTS_HOST_PATH" "$TEMP_DIR_ON_HOST/CLAP_weights_2023.pth"


# --- 2. CREAZIONE DELLO SCRIP PYTHON TEMPORANEO (LOGICA CORRETTA) ---
# Elimina ogni riferimento a YAML e get_config_from_yaml.

cat << EOF > "$TEMP_SCRIPT_NAME"
import sys
import os
import logging
from src.models import CLAP_initializer 

logging.basicConfig(level=logging.CRITICAL)

# --------------------------------------------------------------------
# LOGICA DI ISPEZIONE
# --------------------------------------------------------------------
try:
    # 1. Inizializzazione del modello (Firma corretta, usa le variabili d'ambiente)
    # Usiamo 'cpu' perché non abbiamo garanzie di avere la GPU
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

# --- 3. ESECUZIONE DEL CONTAINER ---

echo "--- 🔍 Esecuzione Script Ispezione Parametri CLAP ---"

# 🎯 CORREZIONE: Non passare più argomenti posizionali non necessari allo script Python.
singularity exec \
    --bind "$TEMP_DIR_ON_HOST:$CONTAINER_WORK_DIR" \
    --bind "$(pwd)/configs:/app/configs" \
    --bind "$(pwd)":/app \
    --pwd /app \
    "$SIF_FILE" \
    python3 "$TEMP_SCRIPT_NAME"

# --- 4. PULIZIA ---
echo "Pulizia script e directory temporanea..."
rm -f "$TEMP_SCRIPT_NAME"
rm -rf "$TEMP_DIR_ON_HOST"
echo "Esecuzione completata."
