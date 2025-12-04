#!/bin/bash
#
# Script per l'esecuzione INTERATTIVA (da lanciare su un nodo di calcolo già allocato, es. tramite 'salloc')

# --- 1. VARIABILI GLOBALI E PERCORSI ---

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"

# 🎯 NUOVO: Definiamo la directory di base TEMPORANEA sul tuo SCRATCH permanente.
SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_data_pipeline_$$"
# Questo è il percorso nel container dove monteremo la base Scratch
CONTAINER_SCRATCH_BASE="/scratch_base" 
# Container mount point per la cartella di lavoro (dove ci sono pesi e script)
CONTAINER_WORK_DIR="/app/temp_work"

# Cartella di lavoro/temporanea, ora SU SCRATCH
TEMP_DIR="$SCRATCH_TEMP_DIR/work_dir" 
TEMP_PYTHON_SCRIPT_PATH="$TEMP_DIR/create_h5_data.py"
TEMP_LOG_MERGE_SCRIPT="$TEMP_DIR/temp_log_merge_script_$$.py" 
TEMP_LOG_ANALYSE_SCRIPT="$TEMP_DIR/temp_log_analyse_script_$$.py" 


# Variabili di configurazione per il Benchmark Rapido
BENCHMARK_CONFIG_FILE="test_config.yaml" 
BENCHMARK_AUDIO_FORMAT="wav"
BENCHMARK_N_OCTAVE="1"


# --- 2. PREPARAZIONE DATI SULLO SCRATCH PERMANENTE ---

echo "--- 🛠️ Preparazione Dati Temporanei su Scratch ($SCRATCH_TEMP_DIR) ---"

# 2.1. Creazione della cartella temporanea e della sua struttura interna
# Creiamo la struttura necessaria su Scratch
mkdir -p "$TEMP_DIR" # Cartella di lavoro per script/log
mkdir -p "$SCRATCH_TEMP_DIR/dataSEC/RAW_DATASET" # Cartella per i dati HDF5

# 2.2. Copia dei pesi CLAP (Ora scrivono su $TEMP_DIR, che è su Scratch)
echo "Copia dei pesi CLAP su Scratch temporanea ($TEMP_DIR)..."
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/CLAP_weights_2023.pth"
# Questa operazione ora è sicura perché $TEMP_DIR è su Scratch
cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS" 


# --- 3. CONFIGURAZIONE ESECUTIVA (Variabili d'Ambiente) ---

echo "--- ⚙️ Configurazione Ambiente di Esecuzione ---"

# Queste variabili d'ambiente sono usate dal tuo models.py:
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 

# Il percorso dei pesi CLAP DEVE essere il PERCORSO INTERNO AL CONTAINER.
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/CLAP_weights_2023.pth"

# Variabile per evitare il salvataggio degli embeddings in un test rapido
export NO_EMBEDDING_SAVE="True" 

# Reindirizziamo basedir su Scratch
export NODE_TEMP_BASE_DIR="$CONTAINER_SCRATCH_BASE/dataSEC"


# --- 4. GENERAZIONE DINAMICA DEL DATASET HDF5 ---

echo "Generazione dinamica del dataset HDF5 di test in $SCRATCH_TEMP_DIR/dataSEC/RAW_DATASET"

# 4.1. Creazione dello script Python temporaneo per la generazione (Scrive in $TEMP_DIR, su Scratch)
cat << EOF > "$TEMP_PYTHON_SCRIPT_PATH"
import sys
import os
sys.path.append('.')

from tests.utils.create_fake_raw_audio_h5 import create_fake_raw_audio_h5 

# Path interno al container per il dataset RAW. Legge NODE_TEMP_BASE_DIR.
TARGET_DIR = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'RAW_DATASET') 

if __name__ == '__main__':
    print(f"Generazione file in: {TARGET_DIR}")
    create_fake_raw_audio_h5(TARGET_DIR)
EOF

# 4.2. Esegui lo script Python temporaneo. Bind mount puliti.
# 🎯 CORREZIONE: Comando singularity EXEC su un'UNICA RIGA per evitare interruzioni.
singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" "$SIF_FILE" python3 "$CONTAINER_WORK_DIR/create_h5_data.py"


# --- 5. ESECUZIONE DELLA PIPELINE (get_clap_embeddings.py) ---

echo "--- 🚀 Avvio Esecuzione Interattiva ---"

# 🎯 CORREZIONE: Comando singularity EXEC su un'UNICA RIGA.
singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$(pwd)/configs:/app/configs" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" "$SIF_FILE" python3 scripts/get_clap_embeddings.py --config_file "$BENCHMARK_CONFIG_FILE" --n_octave "$BENCHMARK_N_OCTAVE" --audio_format "$BENCHMARK_AUDIO_FORMAT"


# --- 6. ANALISI FINALE DEI LOG (DOPO LA MERGE) ---
# ... (Inserisci qui la sezione 6 se è necessaria. Non ho file di log di merge/analisi, 
# ma assicurati che anche quel blocco usi la sintassi corretta di Singularity e i percorsi interni) ...


# --- 7. PULIZIA FINALE ---

echo "--------------------------------------------------------"
echo "Pulizia di tutti i file temporanei su Scratch: $SCRATCH_TEMP_DIR"
echo "--------------------------------------------------------"

rm -rf "$SCRATCH_TEMP_DIR"
echo "Pulizia completata."
