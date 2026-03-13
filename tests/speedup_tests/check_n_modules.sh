#!/bin/bash

MY_USER=$(whoami)
BASE_DIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASE_DIR}/SEC_pipeline"
SIF="${PROJECT_DIR}/.containers/clap_pipeline.sif"

# Lista delle librerie da analizzare
LIBRARIES=("h5py" "librosa" "msclap" "numpy" "pandas" "scipy" "soundfile" "transformers" "torch")

echo -e "Library         | Active Modules (Isolated Tree)"
echo -e "-----------------------------------------------"

for LIB in "${LIBRARIES[@]}"; do
    # Eseguiamo un processo isolato dentro il SIF per ogni libreria
    COUNT=$(singularity exec "$SIF" python3 -c "
import sys
import importlib
# Moduli base prima dell'import
before = set(sys.modules.keys())
try:
    importlib.import_module('$LIB')
    # Moduli totali dopo l'import
    after = set(sys.modules.keys())
    # Risultato: solo i moduli caricati per questa specifica libreria e sue dipendenze
    print(len(after - before))
except:
    print('Error')
")
    printf "%-15s | %-25s\n" "$LIB" "$COUNT"
done
