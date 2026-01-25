#!/bin/bash

# --- PARAMETERS ---
AUDIO_FORMAT=$1
N_OCTAVE=$2

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <audio_format> <n_octave>"
    echo "Example: $0 wav 1"
    exit 1
fi

# --- GLOBAL ASSETS ---
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
# The mantra: base directory is sibling of SEC_pipeline
DATASEC_BASE="/leonardo_scratch/large/userexternal/$USER/dataSEC"

# --- RUNTIME ---
echo "🔗 Preparing log merging for $AUDIO_FORMAT at ${N_OCTAVE}_octave..."

# Create a temporary Python wrapper
cat << 'EOF' > .tmp_join_logs.py
import sys, os
sys.path.append('.') # Ensure src is reachable
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed

def main():
    # Construct path based on the preprocessed base directory and format
    base_path = os.path.join(basedir_preprocessed, os.environ['FMT'], f"{os.environ['OCT']}_octave")
    
    if not os.path.exists(base_path):
        print(f"❌ Error: Path not found: {base_path}")
        return

    for entry in os.listdir(base_path):
        target_dir = os.path.join(base_path, entry)
        if os.path.isdir(target_dir) and entry.endswith("_secs"):
            print(f"Merging logs in: {entry}")
            join_logs(target_dir) #

if __name__ == '__main__':
    main()
EOF

# Execute merging inside Singularity
export FMT=$AUDIO_FORMAT
export OCT=$N_OCTAVE

singularity exec \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 .tmp_join_logs.py

# --- CLEANUP ---
rm .tmp_join_logs.py
echo "✅ Cleanup complete. All rank logs merged into log.json."
