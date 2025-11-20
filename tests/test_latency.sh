#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           # <--- Esegui 4 processi per nodo
#SBATCH --cpus-per-task=1             # Un CPU core per processo (controlla i requisiti CLAP)
#SBATCH --time=01:00:00
#SBATCH --exclusive                   # O --exclusive, o --gpus-per-task=1
#SBATCH --gres=gpu:4                   # Richiedi 4 GPU per il nodo
#SBATCH -A IscrC_Pb-skite
#SBATCH -p boost_usr_prod

module load python/3.x

# --- VARIABILI GLOBALI ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
TEMP_DIR="/tmp/$SLURM_JOB_ID"
PYTHON_SCRIPT="/tests/test_imports_latency.py" # Script Python da eseguire

# --------------------------------------------------------------------------------
# MISURAZIONE GLOBALE: Inizio Esecuzione
# --------------------------------------------------------------------------------
START_TIME=$(date +%s.%N)
echo "---------------------------------------------------"
echo "Inizio Esecuzione Test di Latenza: $START_TIME"
echo "---------------------------------------------------"

# --- 1. PREPARAZIONE DATI SUL DISCO LOCALE VELOCE (/tmp) ---

# Creazione della cartella temporanea
TIME_START_MKDIR=$(date +%s.%N)
mkdir -p "$TEMP_DIR"
TIME_END_MKDIR=$(date +%s.%N)
echo "Tempo MKDIR: $(python3 -c "print(f'{($TIME_END_MKDIR - $TIME_START_MKDIR):.6f}')") s"

# Copia dei pesi CLAP sull'area locale del nodo (/tmp)
echo "Copia dei pesi CLAP su /tmp..."
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/CLAP_weights_2023.pth"
TIME_START_CP=$(date +%s.%N)
cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS"
TIME_END_CP=$(date +%s.%N)
echo "Tempo COPIA PESI CLAP: $(python3 -c "print(f'{($TIME_END_CP - $TIME_START_CP):.6f}')") s"

# --- 2. CONFIGURAZIONE ESECUTIVA (Variabili d'Ambiente) ---

# Queste variabili d'ambiente sono usate dal tuo models.py:
# Il Text Encoder è ora all'interno del container nella cartella pre-scaricata:
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 

# Il percorso locale dove abbiamo copiato i pesi del modello CLAP
export LOCAL_CLAP_WEIGHTS_PATH="$CLAP_LOCAL_WEIGHTS"

# --- 3. ESECUZIONE (Apptainer/Singularity) ---

echo "Avvio del test di import tramite singularity..."

# Esegui lo script Python che misura il tempo di import
TIME_START_singularity=$(date +%s.%N)

singularity exec \
    --bind $TEMP_DIR:/tmp_data \
    "$SIF_FILE" \
    python "$PYTHON_SCRIPT"

EXIT_CODE=$?

TIME_END_singularity=$(date +%s.%N)
echo "Tempo ESECUZIONE singularity TOTALE: $(python3 -c "print(f'{($TIME_END_singularity - $TIME_START_singularity):.6f}')") s"

# --- 4. PULIZIA ---
TIME_START_RM=$(date +%s.%N)
rm -rf "$TEMP_DIR"
TIME_END_RM=$(date +%s.%N)
echo "Tempo PULIZIA: $(python3 -c "print(f'{($TIME_END_RM - $TIME_START_RM):.6f}')") s"

# --------------------------------------------------------------------------------
# MISURAZIONE GLOBALE: Fine Esecuzione
# --------------------------------------------------------------------------------
END_TIME=$(date +%s.%N)
echo "---------------------------------------------------"
echo "Tempo Totale Script: $(python3 -c "print(f'{($END_TIME - $START_TIME):.6f}')") s"
echo "---------------------------------------------------"
exit $EXIT_CODE
