#!/bin/bash

echo "Setup script for the CLAP environment on Cineca."
echo "Run once only on login node."
echo "Args 1 and 2 have to be rclone remote and remote source folder."

# --- 1. VARIABILI E PERCORSI ---
echo "--- 1. VARIABILI E PERCORSI ---"
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
echo "--- 2. CREAZIONE DELLE DIRECTORY ---"
mkdir -p "$CLAP_WEIGHTS_DIR"
mkdir -p "$CONTAINER_DIR"
mkdir -p "$USER_SCRATCH/dataSEC/PREPROCESSED_DATASET"
mkdir -p "$USER_SCRATCH/dataSEC/results/validation"
mkdir -p "$USER_SCRATCH/dataSEC/results/finetuned_model"

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

# --- 4. CONTROLLO E INSTALLAZIONE DI RCLONE PER TRASFERIMENTO CLOUD ---
echo "--- 4. CONTROLLO E INSTALLAZIONE DI RCLONE PER TRASFERIMENTO CLOUD ---"
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
echo "--- 5. DOWNLOAD CONTAINER ---"

RCLONE_REMOTE="$1"
SOURCE_FOLDER="$2"
SIF_NAME="clap_pipeline.sif" # Nome del file SIF su Drive

# 1. Verifica che i parametri siano stati passati
if [ -z "$RCLONE_REMOTE" ] || [ -z "$SOURCE_FOLDER" ]; then
    echo "ERRORE CRITICO: Parametri rclone mancanti. Esegui lo script con ./setup_CLAP_env.sh <NOME_REMOTO_RCLONE> <NOME_CARTELLA_DRIVE>"
    exit 1
fi

# 2. Controllo Condizionale sulla CONFIGURAZIONE di rclone
echo "Controllo esistenza configurazione rclone per il remoto '$RCLONE_REMOTE'..."

# rclone config show <remoto> restituisce un codice di uscita 0 solo se il remoto esiste
if ! rclone config show "$RCLONE_REMOTE" &> /dev/null; then
    echo "ERRORE CRITICO: Remoto rclone '$RCLONE_REMOTE' NON CONFIGURATO."
    echo "Il sistema non può procedere senza l'autenticazione a Google Drive."
    echo "--> Azione richiesta: Esegui il comando 'rclone config' sul nodo di login e configura il remoto con il nome '$RCLONE_REMOTE'."
    exit 1
fi

echo "Configurazione rclone OK. Il remoto '$RCLONE_REMOTE' è pronto."

# 3. Download Condizionale del Container SIF
if [ ! -f "$SIF_PATH" ]; then
    echo "Container SIF non trovato. Tentativo di download da Google Drive..."
    
    # Esegue il comando rclone copy (aggiunto -v per debug)
    rclone -v copy "$RCLONE_REMOTE:$SOURCE_FOLDER/$SIF_NAME" "$CONTAINER_DIR"
    
    if [ $? -ne 0 ]; then
        echo "ERRORE CRITICO: Download del container SIF con rclone fallito. Controllare i log sopra."
        exit 1
    fi
    echo "Download del container SIF completato con successo nell'area locale."
else
    echo "Immagine singularity (.sif) già presente in '$SIF_PATH'. Salto il download."
fi

echo "--- FINE DOWNLOAD CONTAINER ---"
