#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           # <--- Esegui 4 processi per nodo
#SBATCH --cpus-per-task=1             # Un CPU core per processo (controlla i requisiti CLAP)
#SBATCH --time=01:00:00
#SBATCH --exclusive                   # O --exclusive, o --gpus-per-task=1
#SBATCH --gres=gpu:4                   # Richiedi 4 GPU per il nodo
#SBATCH -A IscrC_Pb-skite
#SBATCH -p boost_usr_prod

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

TEMP_LOG_MERGE_SCRIPT="/tmp/temp_log_merge_script_$$.py"
cat << EOF > "$TEMP_PYTHON_SCRIPT"
import sys
import os

from src.utils import join_logs
from src.dirs_config import basedir_preprocessed, basedir_preprocessed_test

if __name__ == '__main__':
    # when executing this script, call 'python3 TEMP_LOG_MERGE_SCRIPT {audio_format} {n_octave} {test}'
    log_dir = os.path.join(basedir_preprocessed if not bool(sys.argv[3]) else basedir_preprocessed_test,
                                                              f'{sys.argv[1]}', f'{sys.argv[2]}_octave')
    join_logs(log_dir)
EOF

AUDIO_FORMAT="wav"
N_OCTAVE=3
TEST=True

apptainer exec \
    --bind $TEMP_DIR:/tmp_data \
    "$SIF_FILE" \
    python "$TEMP_LOG_MERGE_SCRIPT" "$AUDIO_FORMAT" "$N_OCTAVE" "$TEST"

# --- 4. PULIZIA ---
echo "Pulizia dei file temporanei su /tmp..."
rm -rf "$TEMP_DIR"

echo "Lavoro completato."
