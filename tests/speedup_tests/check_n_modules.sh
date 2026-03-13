#!/bin/bash

MY_USER=$(whoami)
BASE_DIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASE_DIR}/SEC_pipeline"
TMP_DIR="${PROJECT_DIR}/.tmp"
SIF="${PROJECT_DIR}/.containers/clap_pipeline.sif"

mkdir -p "$TMP_DIR"

PYTHON_SCRIPT="$TMP_DIR/count_modules_internal.py"

cat << 'EOF' > "$PYTHON_SCRIPT"
import pkgutil
import importlib
import sys

def count_submodules(package_name):
    try:
        package = importlib.import_module(package_name)
        if not hasattr(package, "__path__"):
            return 1
        
        path = package.__path__
        # walk_packages è il modo più robusto per contare la gerarchia reale
        modules = [modname for loader, modname, ispkg in pkgutil.walk_packages(path, package.__name__ + ".")]
        return len(modules) + 1
    except ImportError:
        return None

libraries = [
    'h5py', 'librosa', 'msclap', 'numpy', 
    'pandas', 'scipy', 'soundfile', 'transformers', 'torch'
]

print(f"{'Library':<15} | {'Submodules Count':<20}")
print("-" * 40)

for lib in libraries:
    count = count_submodules(lib)
    if count:
        print(f"{lib:<15} | {count:<20}")
    else:
        print(f"{lib:<15} | Not installed")
EOF

echo "Running module count inside SIF..."
singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    "$SIF" \
    python3 "$PYTHON_SCRIPT"

rm -rf "$TMP_DIR"
echo "Done. Temporary files removed."
