#!/bin/bash

echo "Setup script for the CLAP environment on Cineca."
echo "Eseguire SOLO UNA VOLTA sul nodo di login."
echo "Args 1 e 2 devono essere il nome del remoto rclone e la cartella sorgente."

# --- 1. VARIABILI E PERCORSI ---
echo "--- 1. VARIABILI E PERCORSI ---"
USER_AREA="$1"
PROJECT_ROOT_DIR="$USER_AREA/SEC_pipeline" 

CLAP_WEIGHTS_DIR="$PROJECT_ROOT_DIR/.clap_weights"
CLAP_WEIGHTS_FILE="CLAP_weights_2023.pth"
CLAP_WEIGHTS_PATH="${CLAP_WEIGHTS_DIR}/${CLAP_WEIGHTS_FILE}"
CLAP_WEIGHTS_URL="https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth"

# 🎯 NUOVO: Asset per il TextEncoder (RoBERTa) - Aggiunto senza rimuovere il resto
ROBERTA_DIR="$CLAP_WEIGHTS_DIR/roberta-base"

CONTAINER_DIR="$PROJECT_ROOT_DIR/.containers"
SIF_PATH="$CONTAINER_DIR/clap_pipeline.sif"


# --- 2. CREAZIONE DELLE DIRECTORY ---
echo "--- 2. CREAZIONE DELLE DIRECTORY ---"
mkdir -p "$CLAP_WEIGHTS_DIR"
mkdir -p "$ROBERTA_DIR" # 🎯 Nuova cartella per RoBERTa
mkdir -p "$CONTAINER_DIR"
mkdir -p "$USER_AREA/dataSEC/PREPROCESSED_DATASET"
mkdir -p "$USER_AREA/dataSEC/results/validation"
mkdir -p "$USER_AREA/dataSEC/results/finetuned_model"


# --- 3. DOWNLOAD CONDIZIONALE DEI PESI CLAP (.pth) ---
echo "--- 3. DOWNLOAD CONDIZIONALE DEI PESI CLAP (.pth) ---"
if [ -f "$CLAP_WEIGHTS_PATH" ]; then
    echo "Pesi CLAP già presenti. Salto il download."
else
    echo "Pesi CLAP non trovati. Avvio il download da $CLAP_WEIGHTS_URL..."
    wget -P "$CLAP_WEIGHTS_DIR" "$CLAP_WEIGHTS_URL"
    if [ $? -ne 0 ]; then
        echo "ERRORE CRITICO: Download dei pesi CLAP fallito."
        exit 1
    fi
    echo "Download dei pesi CLAP completato con successo."
fi

# --- 3.5. 🎯 DOWNLOAD ASSET ROBERTA (TEXT ENCODER) CON FALLBACK ---
echo "--- 3.5. DOWNLOAD ASSET TEXT ENCODER (ROBERTA-BASE) ---"
ROBERTA_FILES=(
    "config.json"
    "pytorch_model.bin"
    "vocab.json"
    "merges.txt"
    "tokenizer_config.json"
    "special_tokens_map.json"
    "tokenizer.json"
    "added_tokens.json"
)

for file in "${ROBERTA_FILES[@]}"; do
    if [ -f "$ROBERTA_DIR/$file" ]; then
        echo "Asset $file già presente."
    else
        echo "Download $file da HuggingFace..."
        # Usiamo wget senza -q per vedere eventuali errori in tempo reale
        wget -P "$ROBERTA_DIR" "https://huggingface.co/roberta-base/resolve/main/$file"
        
        # 🎯 LOGICA DI FALLBACK PER IL 404 (Specialmente per special_tokens_map.json)
        if [ $? -ne 0 ]; then 
            if [ "$file" == "special_tokens_map.json" ]; then
                echo "⚠️ Download fallito per $file (404?). Creazione manuale del file standard..."
                echo '{
  "bos_token": "<s>",
  "cls_token": "<s>",
  "eos_token": "</s>",
  "mask_token": "<mask>",
  "pad_token": "<pad>",
  "sep_token": "</s>",
  "unk_token": "<unk>"
}' > "$ROBERTA_DIR/$file"
                echo "✅ File $file creato manualmente."
            else
                echo "ERRORE CRITICO: Download di $file fallito e non è previsto un fallback."
                exit 1
            fi
        fi
    fi
done


# --- 4. CONTROLLO E INSTALLAZIONE DI RCLONE PER TRASFERIMENTO CLOUD ---
echo "--- 4. CONTROLLO E INSTALLAZIONE DI RCLONE PER TRASFERIMENTO CLOUD ---"

module load rclone 2>/dev/null

RCLONE_ABS_PATH="" 

if ! command -v rclone &> /dev/null; then
    echo "rclone non trovato o modulo non disponibile. Tentativo di installazione locale..."
    
    RCLONE_VERSION="1.66.0" 
    ARCH="amd64" 
    INSTALL_DIR="$PROJECT_ROOT_DIR/bin" 
    TEMP_FILE="rclone-v${RCLONE_VERSION}-linux-${ARCH}.zip"

    mkdir -p "$INSTALL_DIR"
    export PATH="$INSTALL_DIR:$PATH" 

    if wget -q -O "$TEMP_FILE" "https://downloads.rclone.org/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-${ARCH}.zip"; then
        echo "Download rclone completato."
    else
        echo "ERRORE CRITICO: Download di rclone fallito."
        exit 1
    fi
    
    unzip -q "$TEMP_FILE"
    FOLDER_NAME="rclone-v${RCLONE_VERSION}-linux-${ARCH}"
    cp "$FOLDER_NAME/rclone" "$INSTALL_DIR/"
    chmod +x "$INSTALL_DIR/rclone"
    rm -rf "$FOLDER_NAME" "$TEMP_FILE"
    
    RCLONE_ABS_PATH="$INSTALL_DIR/rclone" 
else
    RCLONE_ABS_PATH="$(command -v rclone)" 
fi

echo "rclone è disponibile: $RCLONE_ABS_PATH"


# --- 4.5. CONTROLLO CONFIGURAZIONE MANUALE DI RCLONE ---
RCLONE_REMOTE="$2"
SOURCE_FOLDER="$3"
SIF_NAME="clap_pipeline.sif"

echo "Controllo esistenza configurazione rclone per il remoto '$RCLONE_REMOTE'..."

if ! "$RCLONE_ABS_PATH" config show "$RCLONE_REMOTE" &> /dev/null; then
    echo "========================================================================================="
    echo "ERRORE CRITICO: Remoto rclone '$RCLONE_REMOTE' NON CONFIGURATO."
    echo "Azione manuale richiesta: $RCLONE_ABS_PATH config"
    echo "========================================================================================="
    exit 1 
fi

echo "Configurazione rclone OK."


# --- 5. DOWNLOAD DEL CONTAINER (.SIF) ---
echo "--- 5. DOWNLOAD CONTAINER ---"

if [ -z "$RCLONE_REMOTE" ] || [ -z "$SOURCE_FOLDER" ]; then
    echo "ERRORE CRITICO: Parametri rclone mancanti."
    exit 1
fi

if [ ! -f "$SIF_PATH" ]; then
    echo "Container SIF non trovato. Download da Google Drive..."
    "$RCLONE_ABS_PATH" -v copy "$RCLONE_REMOTE:$SOURCE_FOLDER/$SIF_NAME" "$CONTAINER_DIR"
    
    if [ $? -ne 0 ]; then
        echo "ERRORE CRITICO: Download del container SIF fallito."
        exit 1
    fi
    echo "Download completato."
else
    echo "Immagine singularity (.sif) già presente in '$SIF_PATH'."
fi

echo "Setup dell'ambiente CLAP su Cineca completato con successo."
