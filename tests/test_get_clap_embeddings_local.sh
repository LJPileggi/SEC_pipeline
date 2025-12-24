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
# 🎯 NUOVO: Script Python wrapper per la fusione dei log
TEMP_JOIN_LOGS_PATH="$TEMP_DIR/join_logs_wrapper.py"


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
# MODELLO COERENTE: sys.path.append('.')
sys.path.append('.')

# PATH COERENTE: tests.utils.nome_file
from tests.utils.create_fake_raw_audio_h5 import create_fake_raw_audio_h5 

# Path interno al container per il dataset RAW. Legge NODE_TEMP_BASE_DIR.
TARGET_DIR = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'RAW_DATASET') 

if __name__ == '__main__':
    print(f"Generazione file in: {TARGET_DIR}")
    create_fake_raw_audio_h5(TARGET_DIR)
EOF

# 4.2. Esegui lo script Python temporaneo. BIND COERENTE.
singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" "$SIF_FILE" python3 "$CONTAINER_WORK_DIR/create_h5_data.py"


# --- 5. ESECUZIONE DELLA PIPELINE (get_clap_embeddings.py) (INVARIANTE) ---

echo "--- 🚀 Avvio Esecuzione Interattiva ---\n"

singularity exec --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" --bind "$(pwd)/configs:/app/configs" --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" "$SIF_FILE" python3 scripts/get_clap_embeddings.py --config_file "$BENCHMARK_CONFIG_FILE" --n_octave "$BENCHMARK_N_OCTAVE" --audio_format "$BENCHMARK_AUDIO_FORMAT"


# --- 5.5. 🎯 NUOVO: UNIONE SEQUENZIALE DEI LOG (join_logs) ---

echo "--- 🔗 Esecuzione Sequenziale di join_logs ---"

# 5.5.1. Creazione dello script Python wrapper per join_logs
cat << EOF > "$TEMP_JOIN_LOGS_PATH"
import sys
import os
# MODELLO COERENTE: sys.path.append('.')
sys.path.append('.')

from src.utils import join_logs
# Usiamo basedir_preprocessed che è dove i log dovrebbero essere stati scritti
from src.dirs_config import basedir_preprocessed 
import argparse
import logging
# Disattiviamo il logging che può interferire
logging.basicConfig(level=logging.CRITICAL)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_format', type=str, required=True)
    parser.add_argument('--n_octave', type=str, required=True)
    args = parser.parse_args()

    # Costruisci il percorso log_dir che i worker hanno usato
    log_dir = os.path.join(basedir_preprocessed, args.audio_format, f'{args.n_octave}_octave')
    
    print(f"Unione dei log da: {log_dir}")
    try:
        join_logs(log_dir)
        print("Log uniti con successo in log.json.")
    except Exception as e:
        print(f"ERRORE CRITICO nell'unione dei log: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
EOF

# 5.5.2. Esegui il wrapper di join_logs.
singularity exec \
    --bind "$TEMP_DIR:$CONTAINER_WORK_DIR" \
    --bind "$SCRATCH_TEMP_DIR:$CONTAINER_SCRATCH_BASE" \
    "$SIF_FILE" \
    python3 "$CONTAINER_WORK_DIR/join_logs_wrapper.py" \
    --audio_format "$BENCHMARK_AUDIO_FORMAT" \
    --n_octave "$BENCHMARK_N_OCTAVE"


# --- 6. ANALISI FINALE DEI LOG (CORRETTA COERENZA DEI PATH) ---

echo "--------------------------------------------------------"
echo "--- 📊 Avvio Analisi Tempi di Esecuzione (analysis_main_wrapper.py) ---"
echo "--------------------------------------------------------"

# 6.1. Creazione dello script Python wrapper
cat << EOF > "$TEMP_ANALYSE_WRAPPER_PATH"
import os
import sys
import argparse

# 🎯 MODELLO COERENTE: sys.path.append('.')
sys.path.append('.') 

# Importiamo il modulo completo con il path COERENTE: tests.utils.nome_file
import tests.utils.analyse_test_execution_times as analysis_module

# Reindirizziamo la variabile globale config_test_folder nel modulo importato per puntare alla cartella temporanea
analysis_module.config_test_folder = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'PREPROCESSED_DATASET')


def main():
    parser = argparse.ArgumentParser(description='Wrapper per l\'analisi dei tempi di esecuzione CLAP.')
    parser.add_argument('--config_file', type=str, required=True)
    parser.add_argument('--n_octave', type=str, required=True)
    parser.add_argument('--audio_format', type=str, required=True)
    args = parser.parse_args()
    
    print("Avvio analisi dei tempi di esecuzione...")
    
    # 1. Chiamata a analyze_execution_times
    results = analysis_module.analyze_execution_times(
        audio_format=args.audio_format, 
        n_octave=args.n_octave, 
        config_file=args.config_file
    )
    
    # 2. Chiamata a print_analysis_results
    print("\n--- Risultati Analisi ---\n")
    analysis_module.print_analysis_results(results)
    print("\n-------------------------\n")


if __name__ == '__main__':
    main()
EOF

# 6.2. Esegui lo script Python wrapper. BIND COERENTE.
singularity exec \
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
