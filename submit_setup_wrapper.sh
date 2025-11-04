#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00       # 30 minuti sono un tempo ragionevole per il download
#SBATCH --mem=8GB             # Richiediamo 8GB di RAM per la fase di 'squashfs'
#SBATCH -p bsc_serial         # Partizione generale per task che non richiedono GPU
#SBATCH -A IscrC_Pb-skite     # Il tuo account

# --- VARIABILE DOCKER HUB ---
# Il modo più pulito è definire la variabile d'ambiente qui.
# Sostituisci 'ljpileggi' con l'username corretto, se necessario.
DOCKER_USER="ljpileggi"

# --- ESECUZIONE DELLO SCRIPT DI SETUP ---
# Esegui lo script setup_CLAP_env.sh. Lo script leggerà $DOCKER_USER dall'ambiente.
echo "Avvio lo script di setup su nodo di calcolo con DOCKER_USER=$DOCKER_USER..."
/bin/bash ./setup_CLAP_env.sh "$DOCKER_USER"

if [ $? -ne 0 ]; then
    echo "Il job SLURM è terminato con ERRORE durante il setup."
    exit 1
fi

echo "Job di pull e setup completato con successo."
