#!/bin/bash
# run_idofasc.sh

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <input_dataset_path> <output_results_path>"
    exit 1
fi

# Percorsi passati come argomenti posizionali
INPUT_PATH=$1
OUTPUT_PATH=$2

# Definizione percorsi interni al progetto
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"

# Assicuriamoci che la cartella di output esista
mkdir -p "$OUTPUT_PATH"

echo "🚀 Launching IDOFASC_v7.py in interactive mode..."
echo "📂 Input: $INPUT_PATH"
echo "📂 Output: $OUTPUT_PATH"

# Esportiamo le variabili per il processo Python
export INPUT_DATASET_PATH="$INPUT_PATH"
export OUTPUT_RESULTS_PATH="$OUTPUT_PATH"

# Esecuzione tramite Singularity
# Usiamo --no-home per evitare conflitti e bindiamo le directory necessarie
singularity exec --no-home \
    --bind "/leonardo:/leonardo" \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$PROJECT_DIR:/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 alia/IDOFASC_v7.py
