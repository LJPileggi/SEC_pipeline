#!/bin/bash

echo "Setup script for the CLAP environment on Cineca."
echo "Eseguire SOLO UNA VOLTA sul nodo di login."
echo "Args 1 e 2 devono essere il nome del remoto rclone e la cartella sorgente."

# --- 1. VARIABILI E PERCORSI ---
echo "--- 1. VARIABILI E PERCORSI ---"
# NOTA BENE: Se stai lavorando nella tua area su /leonardo_scratch/large/$USER/
USER_AREA="$1"
PROJECT_ROOT_DIR="$USER_AREA/SEC_pipeline" 

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


# --- 4. CONTROLLO E INSTALLAZIONE DI RCLONE PER TRASFERIMENTO CLOUD ---
echo "--- 4. CONTROLLO E INSTALLAZIONE DI RCLONE PER TRASFERIMENTO CLOUD ---"

# 1. Tenta di caricare il modulo CINECA (il metodo preferito)
module load rclone 2>/dev/null

RCLONE_ABS_PATH="" # Inizializza la variabile per il percorso assoluto

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
    
    RCLONE_ABS_PATH="$INSTALL_DIR/rclone" # Definisci il percorso assoluto se installazione locale
else
    RCLONE_ABS_PATH="$(command -v rclone)" # Usa il percorso di quello trovato (dal modulo o da home)
fi

echo "rclone è disponibile: $RCLONE_ABS_PATH"
# --- FINE CONTROLLO RCLONE ---


# --- 4.5. CONTROLLO CONFIGURAZIONE MANUALE DI RCLONE ---
# Questo blocco verifica se il remoto è configurato e FERMA lo script se non lo è.

RCLONE_REMOTE="$2"
SOURCE_FOLDER="$3"
SIF_NAME="clap_pipeline.sif"

# 1. Controllo Condizionale sulla CONFIGURAZIONE di rclone
echo "Controllo esistenza configurazione rclone per il remoto '$RCLONE_REMOTE'..."

# Utilizza il percorso assoluto per rclone config show
if ! "$RCLONE_ABS_PATH" config show "$RCLONE_REMOTE" &> /dev/null; then
    
    echo "========================================================================================="
    echo "======================= AZIONE MANUALE RICHIESTA (RCLONE) ==============================="
    echo "ERRORE CRITICO: Remoto rclone '$RCLONE_REMOTE' NON CONFIGURATO."
    echo "Il sistema non può procedere senza l'autenticazione a Google Drive."
    echo ""
    echo "--> AZIONE RICHIESTA: Devi configurare rclone manualmente, in modo interattivo, prima di procedere."
    echo "1. TERMINA L'ESECUZIONE DELLO SCRIPT (Ctrl+C)."
    echo "2. RILANCIA QUESTO COMANDO NEL TUO TERMINALE CINECA (una volta):"
    echo ""
    echo "   $RCLONE_ABS_PATH config"
    echo ""
    echo "   # Segui i prompt, scegli 'drive', lascia client_id/secret vuoti, usa 'n' per Auto config."
    echo "   # ASSICURATI DI DARE AL REMOTO IL NOME ESATTO: $RCLONE_REMOTE"
    echo ""
    echo "3. DOPO AVER COMPLETATO LA CONFIGURAZIONE, RILANCIA LO SCRIPT DI SETUP."
    echo "========================================================================================="
    
    # FERMARE LO SCRIPT
    exit 1 
fi

echo "Configurazione rclone OK. Il remoto '$RCLONE_REMOTE' è pronto per il download."

# --- FINE CONTROLLO CONFIGURAZIONE ---


# --- 5. DOWNLOAD DEL CONTAINER (.SIF) ---
echo "--- 5. DOWNLOAD CONTAINER ---"

# 1. Verifica che i parametri siano stati passati (dopo il controllo di configurazione)
if [ -z "$RCLONE_REMOTE" ] || [ -z "$SOURCE_FOLDER" ]; then
    echo "ERRORE CRITICO: Parametri rclone mancanti. Esegui lo script con ./setup_CLAP_env.sh <NOME_REMOTO_RCLONE> <NOME_CARTELLA_DRIVE>"
    exit 1
fi

# 2. Download Condizionale del Container SIF
if [ ! -f "$SIF_PATH" ]; then
    echo "Container SIF non trovato. Tentativo di download da Google Drive..."
    
    # Esegue il comando rclone copy usando il percorso assoluto (con -v per debug)
    "$RCLONE_ABS_PATH" -v copy "$RCLONE_REMOTE:$SOURCE_FOLDER/$SIF_NAME" "$CONTAINER_DIR"
    
    if [ $? -ne 0 ]; then
        echo "ERRORE CRITICO: Download del container SIF con rclone fallito. Controllare i log sopra."
        exit 1
    fi
    echo "Download del container SIF completato con successo nell'area locale."
else
    echo "Immagine singularity (.sif) già presente in '$SIF_PATH'. Salto il download."
fi

echo "--- FINE DOWNLOAD CONTAINER ---"

echo "Setup dell'ambiente CLAP su Cineca completato con successo. Immagine pronta per l'uso."
