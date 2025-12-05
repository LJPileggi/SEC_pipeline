#!/bin/bash
#
# Script CORRETTO FINALMENTE. Imposta correttamente tutte le variabili d'ambiente 
# necessarie a CLAP_initializer per funzionare dentro il container.

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA (Verifica i percorsi)
# ----------------------------------------------------------------------
# Forza la variabile USER per espandere i percorsi corretti.
USER="lpilegg1" 
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS_HOST_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CONFIG_FILE="config0.yaml" 
TEMP_SCRIPT_NAME="clap_inspector_$$.py"

# --- 1. CONFIGURAZIONE AMBIENTE INTERNO AL CONTAINER (Cruciale!) ---
# Usiamo /app/temp_work come directory di lavoro.

CONTAINER_WORK_DIR="/app/temp_work"

# 1.1. Definisci i percorsi INTERNI al container che models.py si aspetta

# LOCAL_CLAP_WEIGHTS_PATH deve puntare al file di pesi DENTRO il container.
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/CLAP_weights_2023.pth" 
# CLAP_TEXT_ENCODER_PATH è fisso.
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 

# 1.2. Prepara la directory di lavoro temporanea su host (solo per copiare i pesi)
TEMP_DIR_ON_HOST="$(pwd)/temp_clap_inspect_$USER"
mkdir -p "$TEMP_DIR_ON_HOST"

# 1.3. Copia i pesi (necessario per montarli come file locale per il container)
echo "Copia temporanea dei pesi CLAP..."
cp "$CLAP_WEIGHTS_HOST_PATH" "$TEMP_DIR_ON_HOST/CLAP_weights_2023.pth"
echo "Pesi CLAP copiati in $TEMP_DIR_ON_HOST/CLAP_weights_2023.pth"


# --- 2. CREAZIONE DELLO SCRIP PYTHON TEMPORANEO (Estrazione corretta della config) ---

cat << EOF > "$TEMP_SCRIPT_NAME"
import sys
import os
import logging
from src.models import CLAP_initializer 
from src.utils import get_config_from_yaml 

logging.basicConfig(level=logging.CRITICAL)

# --------------------------------------------------------------------
# LOGICA DI ISPEZIONE
# --------------------------------------------------------------------
try:
    # 🎯 CLAP_PATH (sys.argv[1]) e CONFIG_NAME (sys.argv[2]) sono argomenti, 
    # ma CLAP_initializer ignora il primo, usando LOCAL_CLAP_WEIGHTS_PATH dall'ambiente.
    if len(sys.argv) < 3:
        print("❌ ERRORE CRITICO: Argomenti mancanti nel comando Python.")
        sys.exit(1)
        
    CLAP_PATH = sys.argv[1] # Path dei pesi (ignorato da CLAP_initializer, ma passato)
    CONFIG_NAME = sys.argv[2]
    
    # 1. Caricamento della configurazione (Gestione Tupla)
    raw_config = get_config_from_yaml(CONFIG_NAME)
    
    if isinstance(raw_config, tuple):
        config = raw_config[0]
    elif isinstance(raw_config, dict):
        config = raw_config
    else:
        print(f"❌ ERRORE CRITICO: get_config_from_yaml ha restituito un tipo inatteso: {type(raw_config)}")
        sys.exit(1)

    if not isinstance(config, dict):
        print("❌ ERRORE CRITICO: Configurazione estratta non è un dizionario.")
        sys.exit(1)

    # 2. Inizializzazione del modello (La configurazione è ora un dizionario valido)
    # L'argomento config_file è necessario, anche se la logica interna di CLAP_initializer 
    # usa solo le variabili d'ambiente.
    clap_model, audio_embedding, _, _, _, sr = CLAP_initializer(
        'cpu', False
    )

    # 3. Ispezione dei parametri 
    test_attrs = ['mel_transform', 'spectrogram_extractor', 'log_mel_spec', 'spectrogram']
    
    found = False
    for attr_name in test_attrs:
        try:
            mel_module = getattr(audio_embedding, attr_name)
            
            N_FFT = getattr(mel_module, 'n_fft', None)
            HOP_LENGTH = getattr(mel_module, 'hop_length', None)
            N_MELS = getattr(mel_module, 'n_mels', None)
            
            if N_FFT is not None and HOP_LENGTH is not None and N_MELS is not None:
                print("--------------------------------------------------")
                print("✅ PARAMETRI CLAP REALI TROVATI!")
                print(f"N_FFT: {N_FFT}")
                print(f"HOP_LENGTH: {HOP_LENGTH}")
                print(f"N_MELS: {N_MELS}")
                print(f"SR: {sr} Hz")
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

# Esegui singularity con tutti i bind mount necessari:
# 1. Mount della directory temporanea con i pesi CLAP -> $CONTAINER_WORK_DIR
# 2. Mount della directory configs -> /app/configs
# 3. Mount della directory corrente (per lo script temporaneo) -> /app
singularity exec \
    --bind "$TEMP_DIR_ON_HOST:$CONTAINER_WORK_DIR" \
    --bind "$(pwd)/configs:/app/configs" \
    --bind "$(pwd)":/app \
    --pwd /app \
    "$SIF_FILE" \
    python3 "$TEMP_SCRIPT_NAME" "$CLAP_WEIGHTS_HOST_PATH" "$CONFIG_FILE"

# --- 4. PULIZIA ---
echo "Pulizia script e directory temporanea..."
rm -f "$TEMP_SCRIPT_NAME"
rm -rf "$TEMP_DIR_ON_HOST"
echo "Esecuzione completata."
