#!/bin/bash

# =================================================================
# CONFIGURAZIONE (DA MODIFICARE)
# =================================================================

# 1. Imposta il PERCORSO ASSOLUTO del tuo ambiente virtuale (venv)
VENV_PATH="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.venv"

# 2. Percorso e nome dello script Python temporaneo
# Usiamo /tmp perché è un filesystem locale veloce
TEMP_SCRIPT_NAME="test_import_time_$$.py"
TEMP_SCRIPT_PATH="/tmp/${TEMP_SCRIPT_NAME}"

# =================================================================
# FASE 1: CREAZIONE DELLO SCRIPT PYTHON DI DEBUG
# =================================================================

echo "Creazione dello script Python temporaneo: ${TEMP_SCRIPT_PATH}..."

# Uso del 'here document' (cat << EOF) per creare lo script
cat << EOF > "${TEMP_SCRIPT_PATH}"
import time
import logging
import sys
import os

# Configurazione del logging per vedere i risultati immediatamente
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(message)s', 
                    handlers=[logging.StreamHandler(sys.stdout)])

logging.info(f"Process ID: {os.getpid()}")
logging.info("--- STARTING IMPORT TIME MEASUREMENT ---")

# --- Test di Import ---
start_time_total = time.time()
start_time = start_time_total

try:
    import torch
    logging.info(f"-> Imported torch in {time.time() - start_time:.4f} seconds.")
    
    start_time = time.time()
    import msclap
    logging.info(f"-> Imported msclap in {time.time() - start_time:.4f} seconds.")

    start_time = time.time()
    import transformers
    logging.info(f"-> Imported transformers in {time.time() - start_time:.4f} seconds.")

    # Aggiungi qui eventuali altre librerie critiche del tuo script principale
    # start_time = time.time()
    # import librosa
    # logging.info(f"-> Imported librosa in {time.time() - start_time:.4f} seconds.")
    
except Exception as e:
    logging.error(f"Errore durante l'import: {e}")
    sys.exit(1)

logging.info("--- ALL IMPORTS COMPLETED ---")
logging.info(f"Total time for all imports: {time.time() - start_time_total:.4f} seconds.")
EOF

# =================================================================
# FASE 2: ESECUZIONE DELLO SCRIPT
# =================================================================

echo " "
echo "Attivazione dell'ambiente virtuale e lancio dello script..."
echo "========================================================="

# Attiva l'ambiente virtuale usando il percorso specificato
source "${VENV_PATH}/bin/activate"

# Esegui lo script Python
python3 "${TEMP_SCRIPT_PATH}"

# Disattiva l'ambiente virtuale (buona pratica)
deactivate

echo "========================================================="
echo "Esecuzione completata."

# =================================================================
# FASE 3: PULIZIA
# =================================================================

echo "Pulizia dello script temporaneo..."
rm "${TEMP_SCRIPT_PATH}"
echo "Pulizia completata."
