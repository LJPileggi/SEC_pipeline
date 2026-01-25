#!/bin/bash

# --- PARAMETERS ---
AUDIO_FORMAT=$1
N_OCTAVE=$2

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <audio_format> <n_octave>"
    echo "Example: $0 wav 3"
    exit 1
fi

# --- CONFIGURAZIONE PERCORSI (ISPIRATA AL TEST) ---
# Usiamo percorsi assoluti per non fallire il mount
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"

echo "🔗 Avvio merging log per $AUDIO_FORMAT - Ottava $N_OCTAVE..."
echo "📍 Percorso base: $DATASEC_GLOBAL"

# --- GENERAZIONE WRAPPER PYTHON ---
cat << 'EOF' > .tmp_join_logs.py
import sys, os
sys.path.append('.') 
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed

def main():
    # Recuperiamo i parametri dalle variabili d'ambiente passate da shell
    fmt = os.environ.get('FMT')
    octave = os.environ.get('OCT')
    
    # basedir_preprocessed userà NODE_TEMP_BASE_DIR se definita
    base_path = os.path.join(basedir_preprocessed, fmt, f"{octave}_octave")
    
    print(f"📂 Verifico percorso nel container: {base_path}")
    
    if not os.path.exists(base_path):
        print(f"❌ Error: Path not found: {base_path}")
        return

    for entry in os.listdir(base_path):
        target_dir = os.path.join(base_path, entry)
        if os.path.isdir(target_dir) and entry.endswith("_secs"):
            print(f"🔗 Unisco log in: {entry}")
            join_logs(target_dir)
if __name__ == '__main__': main()
EOF

# --- ESECUZIONE (IL CUORE DELLA SOLUZIONE) ---
# 1. Esportiamo NODE_TEMP_BASE_DIR come percorso ASSOLUTO interno al container
# 2. Bindiamo la cartella fisica DATASEC_GLOBAL sulla destinazione del container
export FMT=$AUDIO_FORMAT
export OCT=$N_OCTAVE

singularity exec \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$(pwd):/app" \
    --bind "$DATASEC_GLOBAL:/tmp_data/dataSEC" \
    --env NODE_TEMP_BASE_DIR="/tmp_data/dataSEC" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 .tmp_join_logs.py

# --- CLEANUP ---
rm .tmp_join_logs.py
echo "✅ Operazione completata."
