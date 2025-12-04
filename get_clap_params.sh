#!/bin/bash
#
# Script per estrarre i parametri Mel Spectrogram di CLAP in un ambiente Singularity.
# DEVE essere lanciato dalla directory RADICE della pipeline (quella che contiene src/).

# ----------------------------------------------------------------------
# ⚠️ CONFIGURAZIONE NECESSARIA: Adatta questi percorsi al tuo ambiente
# ----------------------------------------------------------------------

# Sostituisci $USER con il tuo username se non è già gestito dalla shell.
# Il percorso del file SIF (contenitore)
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
# Il percorso dei pesi CLAP (usato per inizializzare il modello)
CLAP_WEIGHTS_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
# File di configurazione YAML (necessario per SR e altre configurazioni)
CONFIG_FILE="config0.yaml" 
# Cartella temporanea per lo script Python
TEMP_SCRIPT_NAME="inspect_clap_params_$$.py"
TEMP_PYTHON_SCRIPT="./$TEMP_SCRIPT_NAME"

# --- 2. CREAZIONE DELLO SCRIPT PYTHON TEMPORANEO ---

cat << EOF > "$TEMP_PYTHON_SCRIPT"
import torch
import logging
import sys
import os

# Configurazione minima di logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger().setLevel(logging.CRITICAL) # Sopprimi log eccessivi di CLAP

# Importazioni: Assumiamo che 'src' sia nel PYTHONPATH o che lo script sia lanciato dalla radice
try:
    from src.models import CLAP_initializer 
    from src.utils import get_config_from_yaml 
except ImportError:
    logging.getLogger().setLevel(logging.INFO)
    logging.error("Fallimento nell'importazione da src.models o src.utils. Assicurati di lanciare lo script dalla directory contenente 'src'.")
    sys.exit(1)


# Argomenti passati dalla shell
CLAP_WEIGHTS_PATH = sys.argv[1]
CONFIG_FILE = sys.argv[2]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

logging.getLogger().setLevel(logging.INFO)
logging.info(f"Caricamento modello CLAP con pesi: {CLAP_WEIGHTS_PATH}")

# 1. Carica Configurazione
try:
    # Cerchiamo il config nella directory 'configs' o nella directory corrente
    config = get_config_from_yaml(CONFIG_FILE)
except Exception as e:
    logging.warning(f"Impossibile caricare la configurazione YAML: {e}. Continuo con parametri di fallback.")
    config = {} # Fallback

# 2. Inizializza CLAP Model
try:
    # La firma della funzione CLAP_initializer è inferita dal tuo codice:
    clap_model, audio_embedding, _, _, _, sr = CLAP_initializer(
        DEVICE, torch.cuda.is_available()
    )
    logging.info("Modello CLAP caricato con successo.")

except Exception as e:
    logging.error(f"Errore durante l'inizializzazione del modello CLAP. Impossibile proseguire: {e}")
    sys.exit(1)


# 3. Tentativo di estrazione dei parametri Mel Spectrogram dal modulo encoder (audio_embedding)
logging.info("--------------------------------------------------")
logging.info("Inizio Ispezione Parametri Mel Spectrogram CLAP:")

# Nomi di attributi da testare che contengono il modulo Mel Spectrogram
test_attrs = [
    'mel_transform', 
    'log_mel_spec', 
    'mel_spectrogram_module', 
    'spectrogram_extractor',
    'spectrogram'
]

found = False
for attr_name in test_attrs:
    try:
        # 1. Prova a prendere il modulo Mel Spectrogram dal tuo encoder (audio_embedding)
        mel_module = getattr(audio_embedding, attr_name) 
        
        # 2. Verifica se il modulo ha gli attributi di configurazione (standard PyTorch/torchaudio)
        N_FFT = getattr(mel_module, 'n_fft', None)
        HOP_LENGTH = getattr(mel_module, 'hop_length', None)
        N_MELS = getattr(mel_module, 'n_mels', None)
        
        if N_FFT is not None and HOP_LENGTH is not None and N_MELS is not None:
            logging.info("Parametri CLAP Mel Spectrogram TROVATI:")
            logging.info(f"Modulo Sorgente: audio_embedding.{attr_name}")
            logging.info(f"N_FFT: {N_FFT}")
            logging.info(f"HOP_LENGTH: {HOP_LENGTH}")
            logging.info(f"N_MELS: {N_MELS}")
            logging.info(f"SR (dal config/model): {sr} Hz")
            found = True
            break
            
    except AttributeError:
        continue # Prova il prossimo attributo

if not found:
    logging.warning("FALLIMENTO: Parametri Mel Spectrogram non trovati sui nomi di attributo standard. ")
    logging.warning("Usare come fallback i valori standard CLAP: N_FFT=1024, HOP_LENGTH=512, N_MELS=64.")

logging.info("--------------------------------------------------")

EOF

# --- 3. ESECUZIONE DELLO SCRIPT NEL CONTAINER ---

echo "--- 🔍 Avvio Ispezione Parametri CLAP Mel Spectrogram ---"

# Esecuzione dello script temporaneo:
# Monta la directory corrente (dove si trovano 'src' e 'configs') all'interno del container.
# Monta anche i pesi CLAP.
singularity exec \
    --nv \
    --bind "$(pwd)":/app \
    --bind "$CLAP_WEIGHTS_PATH":"$CLAP_WEIGHTS_PATH" \
    "$SIF_FILE" \
    python3 /app/"$TEMP_PYTHON_SCRIPT" "$CLAP_WEIGHTS_PATH" "$CONFIG_FILE"

# --- 4. PULIZIA ---

echo "Pulizia dello script temporaneo..."
rm -f "$TEMP_PYTHON_SCRIPT"

echo "Esecuzione completata."
