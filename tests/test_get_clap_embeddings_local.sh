#!/bin/bash
#
# Script per l'esecuzione INTERATTIVA (da lanciare su un nodo di calcolo già allocato, es. tramite 'salloc')

# --- 1. VARIABILI GLOBALI E PERCORSI ---

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"

# 🎯 Definiamo la directory di base TEMPORANEA sul tuo SCRATCH permanente.
SCRATCH_TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_data_pipeline_$$"
# Questo è il percorso nel container dove monteremo la base Scratch
CONTAINER_SCRATCH_BASE="/scratch_base" 
# Container mount point per la cartella di lavoro (dove ci sono pesi e script)
CONTAINER_WORK_DIR="/app/temp_work"

# Cartella di lavoro/temporanea, ora SU SCRATCH
TEMP_DIR="$SCRATCH_TEMP_DIR/work_dir" 
TEMP_PYTHON_SCRIPT_PATH="$TEMP_DIR/create_h5_data.py"

# 🎯 Script Python wrapper per l'analisi (il main richiesto)
TEMP_ANALYSE_WRAPPER_PATH="$TEMP_DIR/analysis_main_wrapper.py"


# Variabili di configurazione per il Benchmark Rapido
BENCHMARK_CONFIG_FILE="test_config.yaml" 
BENCHMARK_AUDIO_FORMAT="wav"
BENCHMARK_N_OCTAVE="1"


# --- 2. PREPARAZIONE DATI SULLO SCRATCH PERMANENTE ---

echo "--- 🛠️ Preparazione Dati Temporanei su Scratch ($SCRATCH_TEMP_DIR) ---"

# 2.1. Creazione della cartella temporanea e della sua struttura interna
mkdir -p "$TEMP_DIR" 
mkdir -p "$SCRATCH_TEMP_DIR/dataSEC/RAW_DATASET" 

# 2.2. Copia dei pesi CLAP
echo "Copia dei pesi CLAP su Scratch temporanea ($TEMP_DIR)..."
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/CLAP_weights_2023.pth"
cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS" 


# --- 3. CONFIGURAZIONE ESECUTIVA (Variabili d'Ambiente) ---

echo "--- ⚙️ Configurazione Ambiente di Esecuzione ---"

export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 
export LOCAL_CLAP_WEIGHTS_PATH="$CONTAINER_WORK_DIR/CLAP_weights_2023.pth"
export NO_EMBEDDING_SAVE="True" 
export NODE_TEMP_BASE_DIR="$CONTAINER_SCRATCH_BASE/dataSEC"


# --- 4. GENERAZIONE DINAMICA DEL DATASET HDF5 (INVARIATO) ---

echo "Generazione dinamica del dataset HDF5 di test in $SCRATCH_TEMP_DIR/dataSEC/RAW_DATASET"

# 4.1. Creazione dello script Python temporaneo per la generazione
cat << EOF > "$TEMP_PYTHON_SCRIPT_PATH"
import sys
import os
sys.path.append('.')

# Questo import funziona perché la root del progetto è nel PATH tramite bind.
from tests.utils.create_fake_raw_audio_h5 import create_fake_raw_audio_h5 

TARGET_DIR = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'RAW_DATASET') 

if __name__ == '__main__':
    print(f"Generazione file in: {TARGET_DIR}")
    create_fake_raw_audio_h5(TARGET_DIR)
EOF

# 4.2. Esegui lo script Python temporaneo.
singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" "$SIF_FILE" python3 "$CONTAINER_WORK_DIR/create_h5_data.py"


# --- 5. ESECUZIONE DELLA PIPELINE (get_clap_embeddings.py) (INVARIATO) ---

echo "--- 🚀 Avvio Esecuzione Interattiva ---"

singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$(pwd)/configs:/app/configs" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" "$SIF_FILE" python3 scripts/get_clap_embeddings.py --config_file "$BENCHMARK_CONFIG_FILE" --n_octave "$BENCHMARK_N_OCTAVE" --audio_format "$BENCHMARK_AUDIO_FORMAT"


# --- 6. ANALISI FINALE DEI LOG (CORRETTA IMPLEMENTAZIONE DEL MAIN WRAPPER) ---

echo "--------------------------------------------------------"
echo "--- 📊 Avvio Analisi Tempi di Esecuzione (analysis_main_wrapper.py) ---"
echo "--------------------------------------------------------"

# 6.1. Creazione dello script Python wrapper, che funge da main per le funzioni di analisi
cat << EOF > "$TEMP_ANALYSE_WRAPPER_PATH"
import os
import sys
import argparse
import json

# Aggiunge la root del progetto (/app) al path per importare utils/
sys.path.append('/app') 

# Importazione del modulo
try:
    from utils import analyse_test_execution_times
except ImportError:
    # Questo fallback è poco probabile dato il bind, ma è una sicurezza
    sys.path.append('/app/utils') 
    import analyse_test_execution_times

# 🎯 Reindirizziamo la cartella di base per i log al percorso su Scratch nel container
analyse_test_execution_times.config_test_folder = os.path.join(os.getenv('NODE_TEMP_BASE_DIR')) 


def main():
    parser = argparse.ArgumentParser(description='Wrapper per l\'analisi dei tempi di esecuzione CLAP.')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--n_octave', type=str, required=True)
    parser.add_argument('--audio_format', type=str, required=True)
    args = parser.parse_args()
    
    print("Avvio analisi dei tempi di esecuzione...")
    
    # 🎯 1. Chiamata a analyze_execution_times per ottenere i risultati
    results = analyse_test_execution_times.analyze_execution_times(
        audio_format=args.audio_format, 
        n_octave=args.n_octave, 
        config_file=args.config_file
    )
    
    # 🎯 2. Chiamata a print_analysis_results per stampare i risultati
    print("\n--- Risultati Analisi ---\n")
    analyse_test_execution_times.print_analysis_results(results)
    print("\n-------------------------\n")


if __name__ == '__main__':
    main()
EOF

# 6.2. Esegui lo script Python wrapper.
singularity exec \
    --bind "$(pwd)":/app \
    --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" \
    --env NODE_TEMP_BASE_DIR="$CONTAINER_SCRATCH_BASE/dataSEC" \
    "$SIF_FILE" \
    python3 "$CONTAINER_WORK_DIR/analysis_main_wrapper.py" \
    --config_file "$BENCHMARK_CONFIG_FILE" \
    --n_octave "$BENCHMARK_N_OCTAVE" \
    --audio_format "$BENCHMARK_AUDIO_FORMAT"


# --- 7. PULIZIA FINALE (COMPLETA) ---

echo "--------------------------------------------------------"
echo "Pulizia di tutti i file temporanei su Scratch: $SCRATCH_TEMP_DIR"
echo "(Rimuove: Dataset HDF5, Log, Embeddings e Pesi temporanei)"
echo "--------------------------------------------------------"

# Elimina ricorsivamente tutta la cartella temporanea, pulendo completamente.
rm -rf "$SCRATCH_TEMP_DIR"
echo "Esecuzione e Analisi completate."
