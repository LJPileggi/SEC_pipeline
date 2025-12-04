#!/bin/bash
#
# Script CORRETTO FINALMENTE. Risolve l'errore 'str expected, not NoneType'
# correggendo i percorsi assoluti all'interno del container.

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA (Verifica i percorsi)
# ----------------------------------------------------------------------
USER="lpilegg1"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CONFIG_FILE="config0.yaml" # Il nome del file, non il percorso assoluto
TEMP_SCRIPT_NAME="simple_clap_inspect_$$.py"
TEMP_PYTHON_SCRIPT="./$TEMP_SCRIPT_NAME"

# --- 2. CREAZIONE DELLO SCRIP PYTHON TEMPORANEO ---

cat << EOF > "$TEMP_PYTHON_SCRIPT"
import sys
import os
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
    CONFIG_NAME = sys.argv[2] # Ottieni il nome del file

    # 🎯 CORREZIONE: Forza il percorso del file config al percorso assoluto del mount.
    CONFIG_PATH_CORRECTED = os.path.join("/app", CONFIG_NAME)
    
    # 1. Caricamento della configurazione e del modello
    config = get_config_from_yaml(CONFIG_PATH_CORRECTED)
    
    # 2. Inizializzazione del modello (firma a 2 argomenti)
    clap_model, audio_embedding, _, _, _, sr = CLAP_initializer(
        'cpu', False
    )

    # 3. Ispezione dell'encoder audio
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
        print("❌ FALLIMENTO: Parametri Mel Spectrogram non trovati.")

except Exception as e:
    # L'errore ora dovrebbe essere risolto. Se non lo è, la firma è sbagliata.
    print(f"❌ ERRORE CRITICO DURANTE L'INIZIALIZZAZIONE: {e}")
    print("Controlla che la firma di CLAP_initializer sia CLAP_initializer(weights, config)")

# --------------------------------------------------------------------
EOF

# --- 3. ESECUZIONE DEL CONTAINER ---

echo "--- 🔍 Esecuzione Script Ispezione Parametri CLAP ---"

# Passa il percorso dei pesi e il NOME del file di configurazione
singularity exec \
    --bind "$(pwd)":/app \
    "$SIF_FILE" \
    python3 /app/"$TEMP_PYTHON_SCRIPT" "$CLAP_WEIGHTS_PATH" "$CONFIG_FILE"

# --- 4. PULIZIA ---
echo "Pulizia script temporaneo..."
rm -f "$TEMP_PYTHON_SCRIPT"
echo "Esecuzione completata."
