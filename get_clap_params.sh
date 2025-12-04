#!/bin/bash
#
# Script CORRETTO per estrarre N_FFT, HOP_LENGTH, N_MELS da CLAP.
# Risolve l'errore 'str expected, not NoneType' usando --env.

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA (Verifica i percorsi)
# ----------------------------------------------------------------------
USER="lpilegg1"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CONFIG_FILE="config0.yaml" 
TEMP_SCRIPT_NAME="simple_clap_inspect_$$.py"
TEMP_PYTHON_SCRIPT="./$TEMP_SCRIPT_NAME"

# --- 2. CREAZIONE DELLO SCRIPT PYTHON (Minimalista e Pulito) ---

cat << EOF > "$TEMP_PYTHON_SCRIPT"
import os
import sys

# Le uniche due importazioni necessarie per caricare il tuo modello.
from src.models import CLAP_initializer 
from src.utils import get_config_from_yaml 

# --------------------------------------------------------------------
# LOGICA DI ISPEZIONE
# --------------------------------------------------------------------
try:
    # 1. Caricamento della configurazione e del modello
    # Ora i valori sono garantiti essere stringhe (non None) grazie al comando shell corretto.
    config = get_config_from_yaml(os.environ.get('CONFIG_PATH'))
    clap_model, audio_embedding, _, _, _, sr = CLAP_initializer(
        os.environ.get('CLAP_PATH'), config
    )

    # 2. Ispezione dell'encoder audio (audio_embedding)
    test_attrs = ['mel_transform', 'spectrogram_extractor', 'log_mel_spec']
    
    found = False
    for attr_name in test_attrs:
        try:
            mel_module = getattr(audio_embedding, attr_name)
            
            N_FFT = getattr(mel_module, 'n_fft', None)
            HOP_LENGTH = getattr(mel_module, 'hop_length', None)
            N_MELS = getattr(mel_module, 'n_mels', None)
            
            if N_FFT is not None and HOP_LENGTH is not None and N_MELS is not None:
                print(f"✅ N_FFT: {N_FFT}")
                print(f"✅ HOP_LENGTH: {HOP_LENGTH}")
                print(f"✅ N_MELS: {N_MELS}")
                print(f"✅ SR: {sr} Hz")
                found = True
                break
                
        except AttributeError:
            continue
    
    if not found:
        print("❌ FALLIMENTO: Parametri Mel Spectrogram non trovati. Usa 1024, 512, 64 come fallback.")

except Exception as e:
    print(f"❌ ERRORE CRITICO DURANTE L'INIZIALIZZAZIONE: {e}")

# --------------------------------------------------------------------
EOF

# --- 3. ESECUZIONE DEL CONTAINER (LA CORREZIONE È QUI) ---

echo "--- 🔍 Esecuzione Script Ispezione Parametri CLAP ---"

# 🎯 CORREZIONE: Uso esplicito di --env per garantire che le variabili arrivino al Python.
singularity exec \
    --bind "$(pwd)":/app \
    --env CLAP_PATH="$CLAP_WEIGHTS_PATH" \
    --env CONFIG_PATH="$CONFIG_FILE" \
    "$SIF_FILE" \
    python3 /app/"$TEMP_PYTHON_SCRIPT"

# --- 4. PULIZIA ---
echo "Pulizia script temporaneo..."
rm -f "$TEMP_PYTHON_SCRIPT"
echo "Esecuzione completata."
