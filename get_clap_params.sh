#!/bin/bash
#
# Script CORRETTO DEFINITIVAMENTE. Corregge l'errore di path nello script Python.

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA (Adottata da test_get_clap_embeddings_local.sh)
# ----------------------------------------------------------------------
# Forza la variabile USER.
USER="lpilegg1" 
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"

# Definisce la directory di base TEMPORANEA sul tuo SCRATCH permanente.
SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_data_pipeline_$$"
# Percorsi interni al container.
CONTAINER_WORK_DIR="/app/temp_work"
TEMP_SCRIPT_NAME="clap_inspector_script.py"

# --- 1. PREPARAZIONE DATI SULLO SCRATCH ---

echo "--- 🛠️ Preparazione Dati Temporanei su Scratch ($SCRATCH_TEMP_DIR) ---"

# 1.1. Creazione della cartella di lavoro su Scratch
mkdir -p "$TEMP_DIR" 
TEMP_DIR="$SCRATCH_TEMP_DIR/work_dir"

# 1.2. Copia dei pesi CLAP nella cartella di lavoro su Scratch
echo "Copia dei pesi CLAP su Scratch temporanea ($TEMP_DIR)..."
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/CLAP_weights_2023.pth"
cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS" 

# --- 2. CONFIGURAZIONE ESECUTIVA (Variabili d'Ambiente) ---

echo "--- ⚙️ Configurazione Ambiente di Esecuzione ---"

# Il percorso dei pesi CLAP DEVE essere il PERCORSO INTERNO AL CONTAINER.
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/CLAP_weights_2023.pth"
# Percorso per l'encoder testuale (preso dall'interno del container come nel tuo script)
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 


# --- 3. CREAZIONE DELLO SCRIP PYTHON TEMPORANEO (Minimale) ---
# Crea il file nella directory corrente ($TEMP_SCRIPT_NAME è solo il nome file)
cat << EOF > "$TEMP_SCRIPT_NAME"
import sys
import os
import logging
import torch
# Assicuriamo che Python trovi models.py (che è in /app/src)
sys.path.append(os.path.join(os.getcwd(), 'src')) 
from models import CLAP_initializer 

# Disattiva logging non critico
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

# 🎯 CORREZIONE CHIAVE: Lancio dello script da /app/clap_inspector_script.py
singularity exec \
    --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$(pwd)":/app \
    --pwd /app \
    "$SIF_FILE" \
    python3 "/app/$TEMP_SCRIPT_NAME"

# --- 5. PULIZIA ---
echo "Pulizia script e directory temporanea su Scratch..."
# Rimuove lo script Python dalla directory corrente
rm -f "$TEMP_SCRIPT_NAME"
# Rimuove tutta la directory temporanea su Scratch
rm -rf "$SCRATCH_TEMP_DIR"
echo "Esecuzione completata."
