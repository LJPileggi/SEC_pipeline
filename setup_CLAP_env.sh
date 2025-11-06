#!/bin/bash
#
# Script di Setup Iniziale dell'Ambiente CLAP su Cineca (Flusso singularity/Docker)
# Eseguire SOLO UNA VOLTA sul nodo di login.

# --- 1. VARIABILI DI CONFIGURAZIONE ---
# Cerca la variabile nell'ambiente ($DOCKER_USER) o come primo argomento ($1)
YOUR_DOCKER_USERNAME="${DOCKER_USER:-$1}" 

# Verifica se l'username è stato fornito
if [ -z "$YOUR_DOCKER_USERNAME" ]; then
    echo "ERRORE CRITICO: Lo username di Docker Hub non è stato specificato."
    echo "Uso: bash $0 <YOUR_DOCKER_USERNAME>"
    echo "O imposta la variabile d'ambiente: export DOCKER_USER=<username> && bash $0"
    exit 1
fi


# --- 2. VARIABILI E PERCORSI ---
# NOTA BENE: Se stai lavorando nella tua area su /leonardo_scratch/large/$USER/
USER_SCRATCH="/leonardo_scratch/large/userexternal/$USER"
PROJECT_ROOT_DIR="$USER_SCRATCH/SEC_pipeline" 

# [NUOVA AGGIUNTA FONDAMENTALE] 
# Forziamo Singularity a usare la nostra area scratch per i file temporanei e la cache, 
# aggirando i limiti di quota della directory HOME.
export SINGULARITY_TMPDIR="$USER_SCRATCH/singularity_tmp"

echo "DEBUG: Valore di SINGULARITY_TMPDIR appena impostato: $SINGULARITY_TMPDIR"
# Visualizza l'intero ambiente per cercare potenziali sovrascritture
env | grep SINGULARITY_TMPDIR

# Creiamo la cartella temporanea se non esiste
mkdir -p "$SINGULARITY_TMPDIR"

CLAP_WEIGHTS_DIR="$PROJECT_ROOT_DIR/.clap_weights"
CLAP_WEIGHTS_FILE="CLAP_weights_2023.pth"
CLAP_WEIGHTS_PATH="${CLAP_WEIGHTS_DIR}/${CLAP_WEIGHTS_FILE}"
CLAP_WEIGHTS_URL="https://huggingface.co/microsoft/msclap/resolve/main/CLAP_weights_2023.pth"

CONTAINER_DIR="$PROJECT_ROOT_DIR/.containers"
SIF_PATH="$CONTAINER_DIR/clap_pipeline.sif"


# --- 3. CREAZIONE DELLE DIRECTORY ---
echo "Creazione delle directory di progetto e output..."
mkdir -p "$CLAP_WEIGHTS_DIR"
mkdir -p "$CONTAINER_DIR"
mkdir -p "$USER_SCRATCH/dataSEC/PREPROCESSED_DATASET"
mkdir -p "$USER_SCRATCH/dataSEC/results/validation"
mkdir -p "$USER_SCRATCH/dataSEC/results/finetuned_model"
mkdir -p "$USER_SCRATCH/dataSEC/testing/PREPROCESSED_DATASET"
mkdir -p "$USER_SCRATCH/dataSEC/testing/results/validation"
mkdir -p "$USER_SCRATCH/dataSEC/testing/results/finetuned_model"


# --- 4. DOWNLOAD CONDIZIONALE DEI PESI CLAP (.pth) ---
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

echo "DEBUG: Valore di SINGULARITY_TMPDIR prima del pull: $SINGULARITY_TMPDIR"
env | grep SINGULARITY_TMPDIR

# --- 5. DOWNLOAD E CONVERSIONE DEL CONTAINER (.SIF) ---
echo "Controllo e download/conversione dell'immagine Docker da Hub (Utente: $YOUR_DOCKER_USERNAME)..."
if [ -f "$SIF_PATH" ]; then
    echo "Immagine singularity (.sif) già presente. Salto il pull."
else
    singularity pull "$SIF_PATH" docker://"$YOUR_DOCKER_USERNAME"/clap_pipeline:latest
    if [ $? -ne 0 ]; then
        echo "ERRORE CRITICO: Pull del container fallito. Controlla che l'immagine sia pubblica o che tu sia autenticato."
        exit 1
    fi
    echo "Download e conversione del container completati con successo."
fi

echo "Setup dell'ambiente CLAP su Cineca completato con successo. Immagine pronta per l'uso."
