#!/bin/bash
#
# Script CORRETTO FINALMENTE. Utilizza il bind esplicito della cartella configs
# e forza la CWD per replicare l'ambiente di esecuzione funzionante.

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA (Verifica i percorsi)
# ----------------------------------------------------------------------
USER="lpilegg1"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_WEIGHTS_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CONFIG_FILE="config0.yaml" 
TEMP_SCRIPT_NAME="simple_clap_inspect_$$.py"
TEMP_PYTHON_SCRIPT="./$TEMP_PYTHON_SCRIPT_NAME"

# --- 2. CREAZIONE DELLO SCRIP PYTHON TEMPORANEO ---
# Codice più semplice possibile.

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
        
    CLAP_PATH = sys.argv[1]
    CONFIG_NAME = sys.argv[2]
    
    # Non è necessario cambiare la directory qui se usiamo --pwd /app nello script SH
    
    # 1. Caricamento della configurazione e del modello
    # get_config_from_yaml('config0.yaml') cerca 'configs/config0.yaml' rispetto alla CWD (/app)
    config = get_config_from_yaml(CONFIG_NAME)
    
    if config is None or not isinstance(config, dict) or not config:
        print("❌ ERRORE CRITICO: Configurazione nulla o non trovata (CONTROLLARE IL BIND configs).")
        sys.exit(1)

    # 2. Inizializzazione del modello (firma a 2 argomenti)
    clap_model, audio_embedding, _, _, _, sr = CLAP_initializer(
        CLAP_PATH, config
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
        print("❌ FALLIMENTO: Parametri Mel Spectrogram non trovati sull'encoder CLAP.")

except Exception as e:
    print(f"❌ ERRORE CRITICO DURANTE L'INIZIALIZZAZIONE: {e}")

# --------------------------------------------------------------------
EOF

# --- 3. ESECUZIONE DEL CONTAINER (LA CORREZIONE È QUI) ---

echo "--- 🔍 Esecuzione Script Ispezione Parametri CLAP ---"

# 🎯 CORREZIONE: 
# 1. --bind "$(pwd)":/app (monta la radice del progetto per accedere a src e allo script temporaneo)
# 2. --bind "$(pwd)/configs:/app/configs" (monta esplicitamente la cartella config, replicando il codice funzionante)
# 3. --pwd /app (forza la CWD, così 'configs/config0.yaml' viene risolto correttamente)
singularity exec \
    --bind "$(pwd)":/app \
    --bind "$(pwd)/configs:/app/configs" \
    --pwd /app \
    "$SIF_FILE" \
    python3 /app/"$TEMP_PYTHON_SCRIPT" "$CLAP_WEIGHTS_PATH" "$CONFIG_FILE"

# --- 4. PULIZIA ---
echo "Pulizia script temporaneo..."
rm -f "$TEMP_PYTHON_SCRIPT"
echo "Esecuzione completata."
