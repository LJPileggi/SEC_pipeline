#!/bin/bash

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
RESULTS_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline_buffering_results"
TMP_DIR="${PROJECT_DIR}/.tmp"

STREAM_LOG="${TMP_DIR}/buffering_benchmark_stream.log"
RAW_DATA="${RESULTS_DIR}/buffering_results.csv"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
SLURM_SCRIPT="${TMP_DIR}/buffering_test_slurm.sh"

# Benchmark parameters
ITERATIONS=10
SAMPLES_PER_TEST=500
MAX_BUFFER_RAM_MB=256

# --- 2. SETUP CLEANUP TRAP ---
cleanup() {
    if [ ! -z "$TAIL_PID" ]; then kill "$TAIL_PID" 2>/dev/null; fi
}
trap cleanup EXIT SIGTERM SIGINT

mkdir -p "$TMP_DIR"
mkdir -p "$RESULTS_DIR"
echo "Phase,Cut_Secs,N_Octave,Buffer_Size,Wall_Time" > "$RAW_DATA"

touch "$STREAM_LOG" 
echo "📝 Log initialized at $(date)" > "$STREAM_LOG"

# --- 3. PROBE CREATION (Internal Python script) ---
# IDENTICO AL TUO ORIGINALE
cat << 'EOF' > "${TMP_DIR}/probe_buffering.py"
import time
import os
import sys
import numpy as np
import torch
from src.utils import HDF5EmbeddingDatasetsManager

def run_bench(cut_secs, n_octave, buffer_size, n_samples, h5_path):
    n_bins = 12 * n_octave 
    n_frames = 100 * cut_secs
    embedding_dim = 1024
    spec_shape = (n_bins, n_frames)
    
    if os.path.exists(h5_path): os.remove(h5_path)
    
    manager = HDF5EmbeddingDatasetsManager(h5_path, 'w', buffer_size=buffer_size)
    manager.initialize_hdf5(embedding_dim, spec_shape, 'wav', cut_secs, n_octave, 44100, 42, 0.0, 'train')
    
    emb = np.random.uniform(-1, 1, embedding_dim).astype(np.float64)
    spec = np.random.uniform(-1, 1, spec_shape).astype(np.float64)
    
    start_time = time.perf_counter()
    for i in range(n_samples):
        manager.add_to_data_buffer(emb, spec, f"hash_{i}", f"track_{i}", "class_test")
    
    manager.flush_buffers()
    end_time = time.perf_counter()
    manager.hf.close()
    return end_time - start_time

if __name__ == "__main__":
    c_sec, n_oct, b_size, n_samp, h5_target = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    elapsed = run_bench(c_sec, n_oct, b_size, n_samp, h5_target)
    print(f"RESULT|{c_sec}|{n_oct}|{b_size}|{elapsed:.6f}", flush=True)
EOF

# --- 4. SLURM SCRIPT GENERATION ---
cat << 'EOF' > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=hdf5_buff_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null

# 🎯 FIX: Ordine assegnazione corretto per combaciare con sbatch (Log, Data, SIF, TMP)
STREAM_LOG=$1; RAW_DATA=$2; SIF_FILE=$3; TMP_DIR=$4

CUT_SECS=(1 2 5 10 15)
N_OCTAVES=(1 3 6 12 24)
BUFFER_SIZES=(1 2 4 8 16 32 64 128 256 512 1024)

for c in "${CUT_SECS[@]}"; do
    for o in "${N_OCTAVES[@]}"; do
        REC_SIZE=$(( (1024 * 8) + (12 * o * 100 * c * 8) ))
        MAX_B=$(( 256 * 1024 * 1024 / REC_SIZE ))
        
        for b in "${BUFFER_SIZES[@]}"; do
            if [ "$b" -gt "$MAX_B" ] && [ "$b" -ne 1 ]; then continue; fi
            
            echo "🧪 Testing: Cut=${c}s, Oct=${o}, Buffer=${b}" >> "$STREAM_LOG"
            
            for i in {1..10}; do
                H5_TMP="/tmp/bench_buffer_$(hostname)_${i}.h5"
                
                singularity exec --nv --no-home \
                    --bind "/leonardo_scratch:/leonardo_scratch" \
                    --bind "/tmp:/tmp" \
                    "$SIF_FILE" \
                    python3 -u "${TMP_DIR}/probe_buffering.py" "$c" "$o" "$b" 500 "$H5_TMP" >> "$RAW_DATA" 2>&1
                
                [ -f "$H5_TMP" ] && rm "$H5_TMP"
            done
        done
    done
done
EOF

# --- 5. SUBMISSION AND MONITORING ---
echo "📤 Submitting Job to SLURM..."
# 🎯 FIX: Ordine parametri allineato allo script Slurm (Log, Data, SIF, TMP)
JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT" "$STREAM_LOG" "$RAW_DATA" "$SIF_FILE" "$TMP_DIR")

echo -e "\n📊 MONITORING JOB $JOB_ID (Real-time stream):"

tail -f "$STREAM_LOG" &
TAIL_PID=$!

while true; do
    STATUS=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
    case "$STATUS" in
        COMPLETED|FAILED|TIMEOUT|CANCELLED) break ;;
        *) sleep 30 ;;
    esac
done

# --- 6. PLOTTING RESULTS ---
echo -e "\n📊 Benchmark completed. Generating plots in $RESULTS_DIR..."

# 🎯 FIX: Seaborn rimosso, usato Matplotlib puro. Identico per il resto.
singularity exec --no-home --bind "$RESULTS_DIR:$RESULTS_DIR" "$SIF_FILE" python3 -u - <<PY_PLOT
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

res_dir = "${RESULTS_DIR}"
raw_data = "${RAW_DATA}"

if not os.path.exists(raw_data):
    print(f"❌ Error: Data file {raw_data} not found.")
    exit(1)

# Nota: Il tuo CSV prodotto dal probe usa '|' come separatore nel print, 
# ma la testata echo usa ','. Assicurati che il formato sia coerente.
# Qui assumo formato standard CSV come da tua testata echo.
df = pd.read_csv(raw_data)

df['Label'] = df['Cut_Secs'].astype(str) + "s_" + df['N_Octave'].astype(str) + "oct"
stats = df.groupby(['Label', 'Buffer_Size'])['Wall_Time'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(12, 8))
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
unique_labels = stats['Label'].unique()

for i, label in enumerate(unique_labels):
    sub = stats[stats['Label'] == label]
    m = markers[i % len(markers)]
    plt.errorbar(sub['Buffer_Size'], sub['mean'], yerr=sub['std'], label=label, marker=m, capsize=3, linestyle='-', linewidth=1.5)

plt.xscale('log', base=2); plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.xlabel("Buffer Size (Samples)"); plt.ylabel("Wall Time (s)")
plt.title("I/O Buffering Performance Analysis (SEC Pipeline)")
plt.legend(title="Configurations", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig(os.path.join(res_dir, "buffering_curves_analysis.png"), dpi=300, bbox_inches='tight')
plt.close()
PY_PLOT

rm -rf "$TMP_DIR"
echo "🧹 Temporary files removed. Process finished."
