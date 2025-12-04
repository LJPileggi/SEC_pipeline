#!/bin/bash
#
# Script ULTIMA VERSIONE per estrarre N_FFT, HOP_LENGTH, N_MELS da CLAP.
# Usa sys.argv per risolvere l'errore 'str expected, not NoneType'.

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
# Questo script legge i percorsi direttamente da sys.argv[1] e sys.argv[2]

cat << EOF > "$TEMP_PYTHON_SCRIPT"
import sys
import logging
# Le uniche due importazioni necessarie per caricare il tuo modello.
from src.models import CLAP_initializer 
from src.utils import get_config_from_yaml 

# Configurazione minima di logging
logging.basicConfig(level=logging.CRITICAL) 
logging.getLogger().setLevel(logging.CRITICAL) 

# --------------------------------------------------------------------
# LOGICA DI ISPEZIONE
# --------------------------------------------------------------------
try:
    if len(sys.argv) < 3:
        # Questo non dovrebbe mai accadere con lo script shell corretto
        print("ERRORE: Argomenti mancanti (PESI CLAP e CONFIG FILE).")
        sys.exit(1)
        
    # Leggi i percorsi direttamente dagli argomenti (NON dalle variabili d'ambiente)
    CLAP_PATH = sys.argv[1]
    CONFIG_PATH = sys.argv[2]

    # 1. Caricamento della configurazione e del modello
    config = get_config_from_yaml(CONFIG_PATH)
    
    # Utilizziamo la firma a 2 argomenti (Pesi e Config)
    clap_model, audio_embedding, _, _, _, sr = CLAP_initializer(
        'cpu', False
    )

    # 2. Ispezione dell'encoder audio (audio_embedding)
    test_attrs = ['mel_transform', 'spectrogram_extractor', 'log_mel_spec', 'spectrogram']
    
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
    print(f"❌ ERRORE CRITICO DURANTE L'INIZIALIZZAZIONE: {e}. Controlla la firma di CLAP_initializer.")

# --------------------------------------------------------------------
EOF

# --- 3. ESECUZIONE DEL CONTAINER (Passaggio degli argomenti) ---

echo "--- 🔍 Esecuzione Script Ispezione Parametri CLAP ---"

# 🎯 CORREZIONE: Passa i percorsi come argomenti dopo lo script Python.
singularity exec \
    --bind "$(pwd)":/app \
    "$SIF_FILE" \
    python3 /app/"$TEMP_PYTHON_SCRIPT" "$CLAP_WEIGHTS_PATH" "$CONFIG_FILE"

# --- 4. PULIZIA ---
echo "Pulizia script temporaneo..."
rm -f "$TEMP_PYTHON_SCRIPT"
echo "Esecuzione completata."
