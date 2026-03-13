#!/bin/bash

MY_USER=$(whoami)
BASE_DIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASE_DIR}/SEC_pipeline"
TMP_DIR="${PROJECT_DIR}/.tmp"
SIF="${PROJECT_DIR}/.containers/clap_pipeline.sif"

mkdir -p "$TMP_DIR"

PYTHON_SCRIPT="$TMP_DIR/count_modules_internal.py"

cat << 'EOF' > "$PYTHON_SCRIPT"
import sys
import importlib

def count_active_imports(lib_name):
    # Catturiamo lo stato dei moduli prima dell'import
    before = set(sys.modules.keys())
    
    try:
        # Forziamo l'import della libreria
        importlib.import_module(lib_name)
        
        # Catturiamo lo stato dopo l'import
        after = set(sys.modules.keys())
        
        # La differenza sono i moduli effettivamente caricati (inclusi i sottomoduli e le dipendenze)
        new_modules = after - before
        return len(new_modules)
    except Exception as e:
        return f"Error: {e}"

libraries = [
    'h5py', 'librosa', 'msclap', 'numpy', 
    'pandas', 'scipy', 'soundfile', 'transformers', 'torch'
]

print(f"{'Library':<15} | {'Active Modules Loaded':<25}")
print("-" * 45)

# Analizziamo una libreria alla volta in processi separati o pulendo sys.modules
# (Per precisione estrema, meglio testarle una per volta per non "sporcare" i conteggi)
for lib in libraries:
    # Reset parziale o logica a isolamento:
    count = count_active_imports(lib)
    print(f"{lib:<15} | {count:<25}")
EOF

echo "Running module count inside SIF..."
singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    "$SIF" \
    python3 "$PYTHON_SCRIPT"

rm -rf "$TMP_DIR"
echo "Done. Temporary files removed."
