#!/bin/bash
#
# Script minimale per estrarre N_FFT, HOP_LENGTH, N_MELS da CLAP.
# Non contiene logica o complicazioni.

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA (Verifica i percorsi)
# ----------------------------------------------------------------------
USER="lpilegg1"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CONFIG_FILE="config0.yaml" 
TEMP_SCRIPT_NAME="simple_clap_inspect_$$.py"
TEMP_PYTHON_SCRIPT="./$TEMP_SCRIPT_NAME"

# --- 2. ESPORTAZIONE DELLE VARIABILI CHIAVE ---
# Usiamo variabili d'ambiente per passare i percorsi allo script Python in modo pulito.
export CLAP_PATH="$CLAP_WEIGHTS_PATH" 
export CONFIG_PATH="$CONFIG_FILE" 

# --- 3. CREAZIONE DELLO SCRIPT PYTHON TEMPORANEO ---
# Il codice più semplice che riesce a caricare il tuo modello.

cat << EOF > "$TEMP_PYTHON_SCRIPT"
import os
import sys
import torch

# Le uniche due importazioni necessarie per caricare il tuo modello.
from src.models import CLAP_initializer 
from src.utils import get_config_from_yaml 

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------------------------
# LOGICA DI ISPEZIONE
# --------------------------------------------------------------------
try:
    # 1. Caricamento della configurazione e del modello
    config = get_config_from_yaml(os.environ.get('CONFIG_PATH'))
    clap_model, audio_embedding, _, _, _, sr = CLAP_initializer(
        DEVICE, torch.cuda.is_available()
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

# --- 4. ESECUZIONE DEL CONTAINER ---

echo "--- 🔍 Esecuzione Script Ispezione Parametri CLAP ---"

singularity exec \
    --bind "$(pwd)":/app \
    "$SIF_FILE" \
    python3 /app/"$TEMP_PYTHON_SCRIPT"

# --- 5. PULIZIA ---
echo "Pulizia script temporaneo..."
rm -f "$TEMP_PYTHON_SCRIPT"
echo "Esecuzione completata."
