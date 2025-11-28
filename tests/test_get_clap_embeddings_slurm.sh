#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           
#SBATCH --cpus-per-task=1             
#SBATCH --time=01:00:00               
#SBATCH --exclusive                   
#SBATCH --gres=gpu:4                   
#SBATCH -A IscrC_Pb-skite
#SBATCH -p boost_usr_prod

# --- 1. VARIABILI GLOBALI E PERCORSI ---

SIF_FILE="/leonardo_scratch/large/$USER/containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
TEMP_DIR="/tmp/$SLURM_JOB_ID"
TEMP_PYTHON_SCRIPT_PATH="$TEMP_DIR/create_h5_data.py" # Script Python temporaneo per la generazione dati
PERSISTENT_DESTINATION="/leonardo_scratch/large/$USER/SEC_pipeline/benchmark_logs/$SLURM_JOB_ID"

# Variabili di configurazione per il Benchmark Rapido
BENCHMARK_CONFIG_FILE="configs/config_benchmark.yaml" 
BENCHMARK_AUDIO_FORMAT="wav"
BENCHMARK_N_OCTAVE="1"

# --- 2. PREPARAZIONE DATI SUL DISCO LOCALE VELOCE (/tmp) ---

# 2.1. Creazione della cartella temporanea e della sua struttura interna
# La pipeline si aspetta i file HDF5 in TEMP_DIR/dataSEC/RAW_DATASET/
mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET"

# 2.2. Copia dei pesi CLAP
echo "Copia dei pesi CLAP su /tmp..."
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/CLAP_weights_2023.pth"
cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS"

# 2.3. GENERAZIONE DINAMICA DEL DATASET HDF5 (Passaggio Cruciale)
echo "Generazione dinamica del dataset HDF5 di test in $TEMP_DIR/dataSEC/RAW_DATASET/..."

# Crea lo script Python temporaneo usando il codice di create_fake_raw_audio_h5.py
# NOTA: Assicurati che create_fake_raw_audio_h5.py sia disponibile nel SIF o sia
# stato copiato/bindato correttamente se il suo codice è necessario.
cat << EOF > "$TEMP_PYTHON_SCRIPT_PATH"
import sys
import os
sys.path.append('.') # Assicurati che i moduli locali siano importabili

# Importa TUTTE le funzioni necessarie dal tuo file create_fake_raw_audio_h5.py
# (Questo snippet deve essere eseguito all'interno del container)
from src.utils import create_fake_raw_audio_h5 # Assumo che la funzione sia in src.utils o simile

# Percorso di destinazione che mappa a $TEMP_DIR/dataSEC/RAW_DATASET/
# Visto che siamo dentro il container, usiamo /tmp_data/dataSEC/RAW_DATASET
TARGET_DIR = os.path.join(os.getenv('NODE_TEMP_BASE_DIR', '/tmp_data/dataSEC'), 'RAW_DATASET') 

if __name__ == '__main__':
    # Esegui la funzione per generare i file HDF5
    print(f"Generazione file in: {TARGET_DIR}")
    create_fake_raw_audio_h5(TARGET_DIR)
EOF

# Esegui lo script Python temporaneo all'interno del container
singularity exec \
    --bind $TEMP_DIR:/tmp_data \
    "$SIF_FILE" \
    python3 "$TEMP_PYTHON_SCRIPT_PATH"

# --- 3. CONFIGURAZIONE ESECUTIVA (Variabili d'Ambiente) ---

# Variabili usate da models.py e dirs_config.py
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 
export LOCAL_CLAP_WEIGHTS_PATH="$CLAP_LOCAL_WEIGHTS"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC" 

# VARIABILE "NO SAVE EMBEDDINGS" (richiede la modifica in utils.py)
export NO_EMBEDDING_SAVE="True" 

# --- 4. ESECUZIONE DELLA PIPELINE (Singola Run) ---

echo "--------------------------------------------------------"
echo "Avvio del Benchmark Rapido CLAP"
echo "CONFIG: $BENCHMARK_CONFIG_FILE"
echo "SALVATAGGIO EMBEDDINGS: DISATTIVATO"
echo "--------------------------------------------------------"

singularity exec \
    --bind $TEMP_DIR:/tmp_data \
    --bind $(pwd)/configs:/app/configs \
    "$SIF_FILE" \
    python3 ./get_clap_embeddings.py \
        --config_file "$BENCHMARK_CONFIG_FILE" \
        --n_octave "$BENCHMARK_N_OCTAVE" \
        --audio_format "$BENCHMARK_AUDIO_FORMAT"


# --- 5. TRASFERIMENTO DEI LOG E PULIZIA FINALE (Passaggio Corretto) ---

# La pipeline ha creato log e file temporanei sotto $TEMP_DIR/dataSEC/...
echo "--------------------------------------------------------"
echo "Trasferimento dei log e dei dati temporanei su disco persistente..."
echo "Destinazione: $PERSISTENT_DESTINATION"
echo "--------------------------------------------------------"

mkdir -p "$PERSISTENT_DESTINATION"

# Copia l'intera struttura 'dataSEC' (che include tutti i log e i file temporanei)
# L'unica cosa che si salva è la cartella 'dataSEC' (e tutto ciò che c'è sotto)
cp -r "$TEMP_DIR/dataSEC" "$PERSISTENT_DESTINATION/"

# Pulizia di tutti i dati locali (incluso il dataset HDF5 di test e i log originali)
echo "Pulizia della cartella temporanea locale..."
rm -rf "$TEMP_DIR" 
echo "Pulizia completata. I log sono stati salvati in $PERSISTENT_DESTINATION"
