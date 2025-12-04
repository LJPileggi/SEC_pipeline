#!/bin/bash
#
# Script ULTIMA VERSIONE per estrarre N_FFT, HOP_LENGTH, N_MELS da CLAP.
# Risolve l'errore 'str expected, not NoneType' cambiando la CWD.

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA (Verifica i percorsi)
# ----------------------------------------------------------------------
USER="lpilegg1"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CONFIG_FILE="config0.yaml" 
TEMP_SCRIPT_NAME="simple_clap_inspect_$$.py"
TEMP_PYTHON_SCRIPT="./$TEMP_SCRIPT_NAME"

# --- 2. CREAZIONE DELLO SCRIP PYTHON TEMPORANEO ---

cat << EOF > "$TEMP_PYTHON_SCRIPT"
import sys
import os
import logging
from src.models import CLAP_initializer 
from src.utils import get_config_from_yaml 

# --------------------------------------------------------------------
# LOGICA DI ISPEZIONE
# --------------------------------------------------------------------
try:
    if len(sys.argv) < 3:
        print("ERRORE: Argomenti mancanti.")
        sys.exit(1)
        
    # Leggi i percorsi da sys.argv
    CLAP_PATH = sys.argv[1]
    CONFIG_NAME = sys.argv[2] # Sarà "config0.yaml"

    # 1. Caricamento della configurazione e del modello
    # get_config_from_yaml userà os.path.join('configs', CONFIG_NAME)
    # Se la CWD è /app/ (cioè la directory dove si trova src), allora cercherà configs/config0.yaml
    
    # 🎯 TENTATIVO 1: Prova a chiamare la funzione con il nome file semplice.
    config = get_config_from_yaml(CONFIG_NAME)
    
    if config is None or not isinstance(config, dict) or not config:
        print(f"❌ ERRORE: get_config_from_yaml ha restituito un valore nullo o vuoto per {CONFIG_NAME}. Controlla la presenza del file in configs/.")
        sys.exit(1)

    # 2. Inizializzazione del modello (firma a 2 argomenti)
    clap_model, audio_embedding, _, _, _, sr = CLAP_initializer(
        'cpu', False
    )

    # 3. Ispezione dell'encoder audio
    test_attrs = ['mel_transform', 'spectrogram_extractor', 'log_mel_spec', 'spectrogram']
    
    # ... (Stampa dei parametri come prima, il codice non è il problema)
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
        print("❌ FALLIMENTO: Parametri Mel Spectrogram non trovati.")

except Exception as e:
    print(f"❌ ERRORE CRITICO DURANTE L'INIZIALIZZAZIONE: {e}")

# --------------------------------------------------------------------
EOF

# --- 3. ESECUZIONE DEL CONTAINER (LA CORREZIONE LOGICA) ---

echo "--- 🔍 Esecuzione Script Ispezione Parametri CLAP ---"

# Esegui lo script Python DALLA DIRECTORY MONTATA /app
# Questa è l'ultima possibilità per risolvere il problema di percorso.
singularity exec \
    --bind "$(pwd)":/app \
    "$SIF_FILE" \
    python3 /app/"$TEMP_PYTHON_SCRIPT" "$CLAP_WEIGHTS_PATH" "$CONFIG_FILE"

# --- 4. PULIZIA ---
echo "Pulizia script temporaneo..."
rm -f "$TEMP_PYTHON_SCRIPT"
echo "Esecuzione completata."
