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

# 🎯 NUOVO: Script Python temporaneo per l'analisi dei tempi.
TEMP_ANALYSE_SCRIPT_PATH="$TEMP_DIR/analyse_wrapper.py"


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


# --- 4. GENERAZIONE DINAMICA DEL DATASET HDF5 (NESSUNA MODIFICA) ---

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
singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" "$SIF_FILE" python3 "$CONTAINER_WORK_DIR/create_h5_data.py"


# --- 5. ESECUZIONE DELLA PIPELINE (get_clap_embeddings.py) (NESSUNA MODIFICA) ---

echo "--- 🚀 Avvio Esecuzione Interattiva ---"

# Comando singularity EXEC come funzionante in precedenza.
singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$(pwd)/configs:/app/configs" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" "$SIF_FILE" python3 scripts/get_clap_embeddings.py --config_file "$BENCHMARK_CONFIG_FILE" --n_octave "$BENCHMARK_N_OCTAVE" --audio_format "$BENCHMARK_AUDIO_FORMAT"


# --- 6. ANALISI FINALE DEI LOG (CORRETTA) ---

echo "--------------------------------------------------------"
echo "--- 📊 Avvio Analisi Tempi di Esecuzione (utils/analyse_test_execution_times.py) ---"
echo "--------------------------------------------------------"

# 6.1. Creazione dello script Python wrapper per l'analisi (Scrive in $TEMP_DIR, su Scratch)
# Questo script importa e lancia il modulo di analisi
cat << EOF > "$TEMP_ANALYSE_SCRIPT_PATH"
import sys
import os
# Aggiunge la directory corrente (. ovvero /app) al PATH per trovare utils
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'utils'))) 
sys.path.append('.') 

# Importa lo script di analisi dal percorso relativo
from analyse_test_execution_times import analyze_execution_times
import argparse

def analyze_wrapper():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--n_octave', type=int)
    parser.add_argument('--audio_format', type=str)
    args = parser.parse_args()
    
    # Chiama la funzione principale dello script di analisi
    # NOTA: analyse_test_execution_times.py ha una funzione main o analoga che deve essere chiamata
    # Qui chiamiamo direttamente analyze_execution_times (assumendo che faccia il lavoro)
    print("Analisi in corso...")
    results = analyze_execution_times(args.audio_format, str(args.n_octave), args.config_file)
    print("\\n--- Risultati Analisi ---\\n")
    # Qui dovresti stampare i risultati ottenuti da analyze_execution_times se ritorna un dict
    import json
    print(json.dumps(results, indent=4))
    print("\\n-------------------------")

if __name__ == '__main__':
    analyze_wrapper()
EOF

# 6.2. Esegui lo script Python temporaneo (Ora lo lanciamo da CONTAINER_WORK_DIR, che è su Scratch)
singularity exec \
    --bind "$(pwd)":/app \
    --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" \
    --env NODE_TEMP_BASE_DIR="$CONTAINER_SCRATCH_BASE/dataSEC" \
    "$SIF_FILE" \
    python3 "$CONTAINER_WORK_DIR/analyse_wrapper.py" \
    --config_file "$BENCHMARK_CONFIG_FILE" \
    --n_octave "$BENCHMARK_N_OCTAVE" \
    --audio_format "$BENCHMARK_AUDIO_FORMAT"


# --- 7. PULIZIA FINALE (POTENZIATA) ---

echo "--------------------------------------------------------"
echo "Pulizia di tutti i file temporanei su Scratch: $SCRATCH_TEMP_DIR"
echo "--------------------------------------------------------"

# 🎯 Correzione: La pulizia ora elimina tutta la directory temporanea, inclusi i dati HDF5.
rm -rf "$SCRATCH_TEMP_DIR"
echo "Esecuzione e Analisi completate."
