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
sys.path.insert(0, '/app')
import numpy as np
import torch
from src.utils import HDF5EmbeddingDatasetsManager

def run_bench(cut_secs, n_octave, buffer_size, n_samples, h5_path):
    n_bins = 12 * n_octave 
    n_frames = 100 * cut_secs
    embedding_dim = 1024
    spec_shape = (n_bins, n_frames)
    
    if os.path.exists(h5_path): os.remove(h5_path)
    
    manager = HDF5EmbeddingDatasetsManager(h5_path, 'a', partitions=set(('splits',)), buffer_size=buffer_size)
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
    c_sec, n_oct, b_size, n_samp, h5_target = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
    elapsed = run_bench(c_sec, n_oct, b_size, n_samp, h5_target)
    print(f"RESULT,{c_sec},{n_oct},{b_size},{elapsed:.6f}", flush=True)
EOF

# --- 4. SLURM SCRIPT GENERATION ---
cat << 'EOF' > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=hdf5_buff_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null

STREAM_LOG=$1; RAW_DATA=$2; SIF_FILE=$3; TMP_DIR=$4; PROJECT_DIR=$5

CUT_SECS=(1 5 10 15)
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
                    --bind "$PROJECT_DIR:/app" \
                    --bind "/tmp:/tmp" \
                    --pwd "/app" \
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
# 🎯 FIX: Aggiunto PROJECT_DIR come 5° parametro per lo script Slurm
JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT" "$STREAM_LOG" "$RAW_DATA" "$SIF_FILE" "$TMP_DIR" "$PROJECT_DIR")

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
echo -e "\n📊 Generating summary and plots in $RESULTS_DIR..."

singularity exec --no-home --bind "$RESULTS_DIR:$RESULTS_DIR" "$SIF_FILE" python3 -u - <<PY_PLOT
import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
sys.path.insert(0, '/app')
import numpy as np

res_dir = "${RESULTS_DIR}"
raw_data = "${RAW_DATA}"

if not os.path.exists(raw_data):
    print(f"❌ Error: Data file {raw_data} not found.")
    exit(1)

# Parsing the generated lines (matching the 'RESULT,' format)
data = []
with open(raw_data, 'r') as f:
    for line in f:
        if line.startswith('RESULT,'):
            parts = line.strip().split(',')
            # Using int(parts[1]) for Cut_Secs as required
            data.append([int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])])

if not data:
    print("❌ Error: No valid results found in CSV. Check formatting.")
    exit(1)

df = pd.DataFrame(data, columns=['Cut_Secs', 'N_Octave', 'Buffer_Size', 'Wall_Time'])

# 🎯 SUMMARY STATISTICS & OVERWRITE
# Grouping by config to calculate mean/std and overwriting the raw CSV as requested
stats = df.groupby(['Cut_Secs', 'N_Octave', 'Buffer_Size'])['Wall_Time'].agg(['mean', 'std']).reset_index()
stats.to_csv(raw_data, index=False)
print(f"✅ Summary statistics saved (overwriting raw data): {raw_data}")

# 📊 MULTI-PLOT GENERATION (2x2 Grid for the 4 Cut_Secs)
unique_cuts = sorted(stats['Cut_Secs'].unique())
fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True)
axes = axes.flatten()

# Define markers to distinguish between different octaves
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']

for i, cut in enumerate(unique_cuts):
    ax = axes[i]
    cut_df = stats[stats['Cut_Secs'] == cut]
    
    unique_octaves = sorted(cut_df['N_Octave'].unique())
    for j, oct_val in enumerate(unique_octaves):
        sub = cut_df[cut_df['N_Octave'] == oct_val]
        m = markers[j % len(markers)]
        
        ax.errorbar(
            sub['Buffer_Size'], 
            sub['mean'], 
            yerr=sub['std'], 
            label=f"{oct_val} oct", 
            marker=m, 
            capsize=3,
            linestyle='-',
            linewidth=1.5
        )
    
    ax.set_xscale('log', base=2)
    ax.grid(True, which="both", linestyle='--', alpha=0.5)
    ax.set_title(f"I/O Performance: {cut}s Audio", fontsize=14, fontweight='bold')
    ax.set_ylabel("Wall Time (s)", fontsize=12)
    
    # Only add X-label to the bottom row for clarity
    if i >= 2:
        ax.set_xlabel("Buffer Size (Samples)", fontsize=12)
    
    ax.legend(title="Octaves", loc='upper right', fontsize=10)

plt.tight_layout()
plot_path = os.path.join(res_dir, "buffering_analysis_grid.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ Grid plot successfully generated: {plot_path}")
PY_PLOT

rm -rf "$TMP_DIR"
echo "🧹 Temporary files removed. Process finished."
