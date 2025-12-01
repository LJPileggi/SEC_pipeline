#!/bin/bash
#
# Script per l'esecuzione INTERATTIVA (da lanciare su un nodo di calcolo già allocato, es. tramite 'salloc')

# --- 1. VARIABILI GLOBALI E PERCORSI ---

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"

# Cartella temporanea (volatile)
TEMP_DIR="/tmp/clap_test_$$" 
TEMP_PYTHON_SCRIPT_PATH="$TEMP_DIR/create_h5_data.py"

# Variabili di configurazione per il Benchmark Rapido
BENCHMARK_CONFIG_FILE="test_config.yaml" 
BENCHMARK_AUDIO_FORMAT="wav"
BENCHMARK_N_OCTAVE="1"

# --- 2. PREPARAZIONE DATI SUL DISCO LOCALE VELOCE (/tmp) ---

echo "--- 🛠️ Preparazione Dati Temporanei in $TEMP_DIR ---"

# 2.1. Creazione della cartella temporanea e della sua struttura interna
# RAW_DATASET è dove verranno generati i file HDF5 di test
mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET"

# 2.2. Copia dei pesi CLAP
echo "Copia dei pesi CLAP su /tmp..."
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/CLAP_weights_2023.pth"
cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS"

# 2.3. GENERAZIONE DINAMICA DEL DATASET HDF5
echo "Generazione dinamica del dataset HDF5 di test..."

cat << EOF > "$TEMP_PYTHON_SCRIPT_PATH"
import sys
import os
sys.path.append('.')

# Assumiamo che la funzione sia in src.utils
from tests.utils.create_fake_raw_audio_h5 import create_fake_raw_audio_h5 

# Path interno al container per il dataset RAW
TARGET_DIR = os.path.join(os.getenv('NODE_TEMP_BASE_DIR', '/tmp_data/dataSEC'), 'RAW_DATASET') 

if __name__ == '__main__':
    print(f"Generazione file in: {TARGET_DIR}")
    # Il codice in create_fake_raw_audio_h5 deve essere aggiornato per essere importabile
    # Se create_fake_raw_audio_h5 è definito nel file omonimo, la sintassi di import
    # deve essere gestita. Assumiamo che sia in un modulo che possiamo importare.
    create_fake_raw_audio_h5(TARGET_DIR)
EOF

# Esegui lo script Python temporaneo
singularity exec \
    --bind "$TEMP_DIR:/tmp_data" \
    "$SIF_FILE" \
    python3 "$TEMP_PYTHON_SCRIPT_PATH"
    
# --- 3. CONFIGURAZIONE ESECUTIVA (Variabili d'Ambiente) ---

echo "--- ⚙️ Configurazione Ambiente di Esecuzione ---"

# Variabili usate da models.py e dirs_config.py
export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 
export LOCAL_CLAP_WEIGHTS_PATH="$CLAP_LOCAL_WEIGHTS"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC" 

# VARIABILE "NO SAVE EMBEDDINGS" (richiede la modifica in utils.py)
export NO_EMBEDDING_SAVE="True" 

# --- 4. ESECUZIONE DELLA PIPELINE (get_clap_embeddings.py) ---

echo "--- 🚀 Avvio Esecuzione Interattiva ---"

singularity exec \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd)/configs:/app/configs" \
    "$SIF_FILE" \
    python3 scripts/get_clap_embeddings.py \
        --config_file "$BENCHMARK_CONFIG_FILE" \
        --n_octave "$BENCHMARK_N_OCTAVE" \
        --audio_format "$BENCHMARK_AUDIO_FORMAT"

# 5. ANALISI FINALE DEI LOG (DOPO LA MERGE)
# -------------------------------------------------------------

# Definiamo il percorso dello script Python temporaneo per l'analisi
TEMP_LOG_ANALYSE_SCRIPT="/tmp/temp_log_analyse_script_$$.py"

# Creiamo lo script Python per l'analisi.
# Questo script carica la funzione di analisi e la esegue.

cat << EOF > "$TEMP_LOG_ANALYSE_SCRIPT"
import sys
import os
import argparse
# Aggiusta il percorso per l'importazione
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'SEC_pipeline')))
sys.path.append('.')

# Importa l'analizzatore di log (il file che mi hai appena confermato)
from tests.utils.analyse_test_execution_times import analyze_execution_times, print_analysis_results 

# Parametri passati dalla shell: audio_format, n_octave, config_file
if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Errore: argomenti mancanti per lo script di analisi.")
        sys.exit(1)
        
    audio_format = sys.argv[1]
    n_octave = sys.argv[2]
    config_file = sys.argv[3]
    
    # 1. Esegui l'analisi
    results = analyze_execution_times(audio_format, n_octave, config_file)
    
    # 2. Stampa i risultati
    print_analysis_results(results)

EOF

# Esegui lo script di analisi usando i parametri definiti in precedenza
# Audio format e n_octave vengono passati dalla shell
echo "Avvio dell'analisi dei tempi di esecuzione..."
singularity exec \
    --bind $TEMP_DIR:/tmp_data \
    "$SIF_FILE" \
    python3 "$TEMP_LOG_ANALYSE_SCRIPT" $AUDIO_FORMAT $N_OCTAVE $CONFIG_FILE

# Rimozione degli script temporanei
rm -f "$TEMP_LOG_ANALYSE_SCRIPT"
# Ricordati di rimuovere anche TEMP_LOG_MERGE_SCRIPT e altri script temporanei!
rm -f "$TEMP_LOG_MERGE_SCRIPT" 
rm -f "$TEMP_PYTHON_SCRIPT"

# --- 5. PULIZIA FINALE ---

echo "--------------------------------------------------------"
echo "ATTENZIONE: Nessun file di log è stato trasferito su disco persistente. I log si trovano su $TEMP_DIR e verranno cancellati."
echo "--------------------------------------------------------"

# Pulizia di tutti i dati locali 
echo "Pulizia della cartella temporanea locale ($TEMP_DIR)..."
rm -rf "$TEMP_DIR" 
echo "Pulizia completata."
