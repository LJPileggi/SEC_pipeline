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
from pathlib import Path

def count_files_and_dirs(package_name):
    try:
        spec = importlib.util.find_spec(package_name)
        if spec is None or spec.submodule_search_locations is None:
            return 1
        
        root_path = Path(spec.submodule_search_locations[0])
        count = 0
        # Camminiamo nel filesystem senza importare nulla
        for path in root_path.rglob('*'):
            # Contiamo file .py (moduli) e cartelle con __init__.py (packages)
            if path.suffix == '.py' or (path.is_dir() and (path / '__init__.py').exists()):
                count += 1
        return count
    except Exception:
        return None

libraries = [
    'h5py', 'librosa', 'msclap', 'numpy', 
    'pandas', 'scipy', 'soundfile', 'transformers', 'torch'
]

print(f"{'Library':<15} | {'Physical Modules Count':<25}")
print("-" * 45)

for lib in libraries:
    count = count_files_and_dirs(lib)
    print(f"{lib:<15} | {count if count else 'Error/Not Found':<25}")
EOF

echo "Running module count inside SIF..."
singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    "$SIF" \
    python3 "$PYTHON_SCRIPT"

rm -rf "$TMP_DIR"
echo "Done. Temporary files removed."
