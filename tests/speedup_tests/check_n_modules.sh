#!/bin/bash

MY_USER=$(whoami)
BASE_DIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASE_DIR}/SEC_pipeline"
TMP_DIR="${PROJECT_DIR}/.tmp"
SIF="${PROJECT_DIR}/.containers/clap_pipeline.sif"

mkdir -p "$TMP_DIR"

PYTHON_SCRIPT="$TMP_DIR/count_modules_internal.py"

cat << 'EOF' > "$PYTHON_SCRIPT"
import os
import importlib
import sys

def count_physical_modules(package_name):
    try:
        # Importiamo la libreria solo per beccare il percorso fisico (__file__)
        module = importlib.import_module(package_name)
        root_path = os.path.dirname(module.__file__)
        
        count = 0
        # Scansioniamo fisicamente la cartella della libreria
        for root, dirs, files in os.walk(root_path):
            for file in files:
                # Contiamo ogni file .py come un modulo/sottomodulo
                if file.endswith('.py'):
                    count += 1
        return count
    except Exception as e:
        return f"Error: {str(e)}"

libraries = [
    'h5py', 'librosa', 'msclap', 'numpy', 
    'pandas', 'scipy', 'soundfile', 'transformers', 'torch'
]

print(f"{'Library':<15} | {'Physical Python Files':<25}")
print("-" * 45)

for lib in libraries:
    count = count_physical_modules(lib)
    print(f"{lib:<15} | {count:<25}")
EOF

echo "Running module count inside SIF..."
singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    "$SIF" \
    python3 "$PYTHON_SCRIPT"

rm -rf "$TMP_DIR"
echo "Done. Temporary files removed."
