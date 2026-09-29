#!/bin/bash
#SBATCH --job-name=check_counts
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:15:00
#SBATCH --mem=16GB
#SBATCH --account=IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

FORMAT=${1:-wav}
DATASET=${2:-dataSEC}

PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"

if [[ $DATASET == "dataSEC" ]]; then
    H5_SOURCE_DIR="/leonardo_scratch/large/userexternal/$USER/dataSEC/RAW_DATASET/raw_$FORMAT"
elif [[ $DATASET == "ESC-50" ]]; then
    H5_SOURCE_DIR="/leonardo_scratch/large/userexternal/$USER/ESC50_HDF5/raw_$FORMAT"
else
    echo "❌ ERRORE: Dataset $DATASET non riconosciuto"
    exit 1
fi

PY_SCRIPT=$(mktemp /tmp/inspect_h5_XXXXXX.py)

cat << 'EOF' > "$PY_SCRIPT"
import os
import sys
import h5py

input_dir = sys.argv[1]
audio_format = sys.argv[2]
dataset_name = sys.argv[3]

if not os.path.exists(input_dir):
    print(f"❌ Errore: cartella non trovata -> {input_dir}")
    sys.exit(1)

h5_files = sorted([f for f in os.listdir(input_dir) if f.endswith(f'_{audio_format}_dataset.h5')])

if not h5_files:
    print(f"❌ Nessun file _{audio_format}_dataset.h5 trovato in {input_dir}")
    sys.exit(1)

print("\n" + "=" * 95)
print(f"🔍 ANALISI CONTEGGI DATASET: {dataset_name} ({audio_format.upper()})")
print("=" * 95)
print(f"{'Classe':<40} | {'Campioni Tot':<15} | {'Tracce Univoche':<18} | {'Duplicati/Chunk'}")
print("-" * 95)

tot_samples = 0
tot_unique = 0

for h5_file in h5_files:
    label = h5_file.replace(f'_{audio_format}_dataset.h5', '')
    h5_path = os.path.join(input_dir, h5_file)
    
    with h5py.File(h5_path, 'r') as hf:
        meta_ds = hf[f'metadata_{audio_format}']
        n_samples = len(meta_ds)
        
        names = [meta_ds[i]['track_name'].decode('utf-8') for i in range(n_samples)]
        unique_names = set(names)
        n_unique = len(unique_names)
        diff = n_samples - n_unique
        
        tot_samples += n_samples
        tot_unique += n_unique
        
        print(f"{label:<40} | {n_samples:<15} | {n_unique:<18} | {diff}")

print("-" * 95)
print(f"{'TOTALE':<40} | {tot_samples:<15} | {tot_unique:<18} | {tot_samples - tot_unique}")
print("=" * 95 + "\n")
EOF

echo "📂 Ispezione da: $H5_SOURCE_DIR"

singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "/tmp:/tmp" \
    --bind "$PROJECT_DIR:/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 "$PY_SCRIPT" "$H5_SOURCE_DIR" "$FORMAT" "$DATASET"

rm -f "$PY_SCRIPT"
