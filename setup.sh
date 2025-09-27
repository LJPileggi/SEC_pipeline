#!/bin/bash

rm -rf .venv

module purge
module load python/3.11.7
module load profile/deeplrn
module load profile/chem-phys

which python
python3 --version

echo "Creating virtual environment..."
python3 -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip3 install -r requirements.txt

echo "Creation of working directories..."
mkdir -p ../dataSEC
mkdir -p ../dataSEC/PREPROCESSED_DATASET
mkdir -p ../dataSEC/results
mkdir -p ../dataSEC/results/validation
mkdir -p ../dataSEC/results/finetuned_model
mkdir -p ../dataSEC/testing
mkdir -p ../dataSEC/testing/PREPROCESSED_DATASET
mkdir -p ../dataSEC/testing/results
mkdir -p ../dataSEC/testing/results/validation
mkdir -p ../dataSEC/testing/results/finetuned_model

# --- Configurazione per i download ---
CLAP_WEIGHTS_DIR="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.clap_weights"
CLAP_WEIGHTS_FILE="CLAP_weights_2023.pth"
CLAP_WEIGHTS_PATH="${CLAP_WEIGHTS_DIR}/${CLAP_WEIGHTS_FILE}"
CLAP_WEIGHTS_URL="https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth"

TEXT_ENCODER_DIR="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.clap_text_encoder/roberta-base"
# --- Fine Configurazione ---

# --- 1. Scarica i pesi di CLAP ---
mkdir -p "$CLAP_WEIGHTS_DIR"
if [ ! -f "$CLAP_WEIGHTS_PATH" ]; then
    echo "Downloading CLAP weights to $CLAP_WEIGHTS_DIR..."
    wget -P "$CLAP_WEIGHTS_DIR" "$CLAP_WEIGHTS_URL"
    if [ $? -ne 0 ]; then
        echo "ERROR: Failed to download CLAP weights. Exiting."
        exit 1
    fi
else
    echo "CLAP weights already exist at $CLAP_WEIGHTS_PATH. Skipping download."
fi

# --- 2. Scarica i componenti del Text Encoder di CLAP ---
mkdir -p "$TEXT_ENCODER_DIR"
# Controllo semplificato: se config.json esiste, assumiamo che il modello sia già scaricato
if [ ! -f "${TEXT_ENCODER_DIR}/config.json" ]; then
    echo "Downloading CLAP Text Encoder (RoBERTa-base) components to $TEXT_ENCODER_DIR..."
    
    # Crea uno script Python temporaneo per scaricare il modello tramite transformers
    # Questo è più robusto di scaricare ogni singolo file con wget
    cat << EOF > /tmp/download_roberta.py
import os
from transformers import AutoTokenizer, AutoModel

SAVE_DIR = "$TEXT_ENCODER_DIR"
os.makedirs(SAVE_DIR, exist_ok=True) # Solo per sicurezza

print(f"Downloading RoBERTa base model to {SAVE_DIR}...")

try:
    # Scarica il tokenizer
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    tokenizer.save_pretrained(SAVE_DIR)

    # Scarica il modello
    model = AutoModel.from_pretrained("roberta-base")
    model.save_pretrained(SAVE_DIR)
    print("Text Encoder download complete.")
except Exception as e:
    print(f"ERROR: Failed to download Text Encoder components: {e}", file=sys.stderr)
    sys.exit(1)
EOF

    # Per semplicità, assumiamo che python del tuo ambiente sia nel PATH o lo specifichi
    python /tmp/download_roberta.py
    if [ $? -ne 0 ]; then
        echo "ERROR: Text Encoder download script failed. Exiting."
        rm /tmp/download_roberta.py # Pulisci lo script temporaneo
        exit 1
    fi
    rm /tmp/download_roberta.py # Pulisci lo script temporaneo
else
    echo "CLAP Text Encoder components already exist at $TEXT_ENCODER_DIR. Skipping download."
fi

# --- 3. Altre impostazioni di ambiente per i job Slurm (se necessario) ---
# Ad esempio, potresti voler esportare queste variabili per i tuoi job Slurm
# export CLAP_CKPT_PATH="$CLAP_WEIGHTS_PATH"
# export CLAP_TEXT_ENCODER_PATH="$TEXT_ENCODER_DIR"

echo "Installation completed successfully. Virtual environment correctly set up."
