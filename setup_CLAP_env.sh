#!/bin/bash

echo "Setup script for the CLAP environment on Cineca."
echo "Run once only on login node."
echo "Args 1 and 2 have to be rclone remote and remote source folder."

# --- 1. VARIABILI E PERCORSI ---
# NOTA BENE: Se stai lavorando nella tua area su /leonardo_scratch/large/$USER/
USER_SCRATCH="/leonardo_scratch/large/userexternal/$USER"
PROJECT_ROOT_DIR="$USER_SCRATCH/SEC_pipeline" 

CLAP_WEIGHTS_DIR="$PROJECT_ROOT_DIR/.clap_weights"
CLAP_WEIGHTS_FILE="CLAP_weights_2023.pth"
CLAP_WEIGHTS_PATH="${CLAP_WEIGHTS_DIR}/${CLAP_WEIGHTS_FILE}"
CLAP_WEIGHTS_URL="https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth"

CONTAINER_DIR="$PROJECT_ROOT_DIR/.containers"
SIF_PATH="$CONTAINER_DIR/clap_pipeline.sif"

# --- 2. CREAZIONE DELLE DIRECTORY ---
echo "Creazione delle directory di progetto e output..."
mkdir -p "$CLAP_WEIGHTS_DIR"
mkdir -p "$CONTAINER_DIR"
mkdir -p "$USER_SCRATCH/dataSEC/PREPROCESSED_DATASET"
mkdir -p "$USER_SCRATCH/dataSEC/results/validation"
mkdir -p "$USER_SCRATCH/dataSEC/results/finetuned_model"

# --- 3. DOWNLOAD CONDIZIONALE DEI PESI CLAP (.pth) ---
echo "Controllo esistenza pesi CLAP..."
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

# --- 4. CONTROLLO E INSTALLAZIONE DI RCLONE PER TRASFERIMENTO CLOUD ---
echo "CONTROLLO RCLONE"
# 1. Tenta di caricare il modulo CINECA (il metodo preferito)
module load rclone 2>/dev/null

# Controlla se rclone è ora disponibile
if ! command -v rclone &> /dev/null; then
    echo "rclone non trovato o modulo non disponibile. Tentativo di installazione locale in area Scratch..."
    
    # NUOVA POSIZIONE DI INSTALLAZIONE NELL'AREA SCRATCH PER EVITARE LA QUOTA HOME
    RCLONE_VERSION="1.66.0" 
    ARCH="amd64" 
    INSTALL_DIR="$PROJECT_ROOT_DIR/bin" # Installazione nello Scratch
    TEMP_FILE="rclone-v${RCLONE_VERSION}-linux-${ARCH}.zip"

    # Creazione della cartella bin e aggiunta al PATH per questa sessione
    mkdir -p "$INSTALL_DIR"
    export PATH="$INSTALL_DIR:$PATH" # Aggiunge il path prima del download

    # Download del binario
    if wget -q -O "$TEMP_FILE" "https://downloads.rclone.org/v${RCLONE_VERSION}/rclone-v${RCLONE_VERSION}-linux-${ARCH}.zip"; then
        echo "Download rclone completato."
    else
        echo "ERRORE CRITICO: Download di rclone fallito. Controllare la versione e l'URL."
        exit 1
    fi
    
    # Decompressione e installazione
    unzip -q "$TEMP_FILE"
    FOLDER_NAME="rclone-v${RCLONE_VERSION}-linux-${ARCH}"
    
    # Spostamento dell'eseguibile nella cartella bin personale nello scratch
    cp "$FOLDER_NAME/rclone" "$INSTALL_DIR/"
    chmod +x "$INSTALL_DIR/rclone"
    
    # Pulizia
    rm -rf "$FOLDER_NAME" "$TEMP_FILE"
    
    echo "rclone v${RCLONE_VERSION} installato con successo in $INSTALL_DIR/."
fi

echo "rclone è disponibile: $(command -v rclone)"
# --- FINE CONTROLLO RCLONE ---

# --- 5. DOWNLOAD DEL CONTAINER (.SIF) ---
echo "Controllo e download del container da Hub (Utente: $YOUR_DOCKER_USERNAME)..."

# Eseguire il pull nell'area locale
if [ ! -f "$SIF_PATH" ]; then # Controlla se il SIF finale esiste
    echo "Setup correct rclone config for transfer."
    module load rclone 2>/dev/null
    rclone config

    RCLONE_REMOTE="$1"
    SOURCE_FOLDER="$2"

    # Verifica che i parametri siano stati passati correttamente
    if [ -z "$RCLONE_REMOTE" ] || [ -z "$SOURCE_FOLDER" ]; then
        echo "ERRORE CRITICO: Parametri rclone mancanti. Esegui lo script con ./setup_CLAP_env.sh <NOME_REMOTO> <NOME_CARTELLA_DRIVE>"
        exit 1
    fi
    
    # Questo comando deve essere eseguito la prima volta per configurare il remoto di rclone
    # NON deve essere nello script di setup automatico se l'autenticazione è interattiva.
    # rclone config

    echo "Downloading SIF file from remote source..."
    rclone copy "$RCLONE_REMOTE:$SOURCE_FOLDER/clap_pipeline.sif" "$CONTAINER_DIR"

    if [ $? -ne 0 ]; then
        echo "CRITICAL ERROR: container download in $CONTAINER_DIR failed."
        exit 1
    fi
    echo "Download e conversione del container completati con successo nell'area locale."
else
    echo "Immagine singularity (.sif) già presente. Salto il pull."
fi

echo "Setup dell'ambiente CLAP su Cineca completato con successo. Immagine pronta per l'uso."
