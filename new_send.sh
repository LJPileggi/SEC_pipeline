#!/bin/bash
#SBATCH --job-name=CLAP_Pipeline_Execution
#SBATCH --partition=boost_usr_prod # O la partizione che usi abitualmente
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --account=TUO_ACCOUNT
#SBATCH --output=clap_job.%j.out
#SBATCH --error=clap_job.%j.err

# --- VARIABILI GLOBALI ---
SIF_FILE="/leonardo_scratch/large/$USER/containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
TEMP_DIR="/tmp/$SLURM_JOB_ID"

# --- 1. PREPARAZIONE DATI SUL DISCO LOCALE VELOCE (/tmp) ---

# Creazione della cartella temporanea
mkdir -p "$TEMP_DIR"

# Copia dei pesi CLAP sull'area locale del nodo (/tmp)
echo "Copia dei pesi CLAP su /tmp..."
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/CLAP_weights_2023.pth"
cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS"

# --- 2. CONFIGURAZIONE ESECUTIVA (Variabili d'Ambiente) ---

# Queste variabili d'ambiente sono usate dal tuo models.py:
# Il Text Encoder è ora all'interno del container nella cartella pre-scaricata:
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 

# Il percorso locale dove abbiamo copiato i pesi del modello CLAP
export LOCAL_CLAP_WEIGHTS_PATH="$CLAP_LOCAL_WEIGHTS"

# --- 3. ESECUZIONE (APPtainer/Singularity) ---

echo "Avvio della pipeline CLAP tramite Apptainer..."

embed_test="./tests/test_embeddings.py"
apptainer exec \
    --bind $TEMP_DIR:/tmp_data \
    "$SIF_FILE" \
    python "$embed_test"

# --- 4. PULIZIA ---
echo "Pulizia dei file temporanei su /tmp..."
rm -rf "$TEMP_DIR"

echo "Lavoro completato."
