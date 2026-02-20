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

# 🎯 FIX: Initialize log file here to prevent tail from failing
touch "$STREAM_LOG" 
echo "📝 Log initialized at $(date)" > "$STREAM_LOG"

# --- 3. PROBE CREATION (Internal Python script) ---
cat << 'EOF' > "${TMP_DIR}/probe_buffering.py"
import time
import os
import sys
import numpy as np
# (Rest of your python probe logic here...)
# I assume the probe logic you already have is working.
EOF

# --- 4. ORCHESTRATOR LOOP ---

echo "📤 Submitting Job to SLURM..."
# 🎯 FIX: Correct order of arguments: 1:log, 2:data, 3:sif, 4:tmp_dir, 5:results_dir
JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT" "$STREAM_LOG" "$RAW_DATA" "$SIF_FILE" "$TMP_DIR" "$RESULTS_DIR")

echo -e "\n📊 MONITORING JOB $JOB_ID (Real-time stream):"
tail -f "$STREAM_LOG" &
TAIL_PID=$!

# Wait for job completion
while true; do
    STATUS=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
    case "$STATUS" in
        COMPLETED|FAILED|TIMEOUT|CANCELLED) break ;;
        *) sleep 30 ;;
    esac
done

kill -9 "$TAIL_PID" 2>/dev/null

# --- 5. SLURM SCRIPT GENERATION ---
cat << 'EOF' > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=sec_buffering_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null

# 🎯 FIX: Correct assignment of positional parameters
STREAM_LOG=$1
RAW_DATA=$2
SIF_FILE=$3
TMP_DIR=$4
RESULTS_DIR=$5

NODE=$(hostname)
echo "🚀 Node $NODE: Starting buffering benchmark" >> "$STREAM_LOG"

# Test matrix
CUT_VALS=(7.0 15.0)
OCT_VALS=(1 3)
# Powers of 2 from 2^10 to 2^18
BUFFER_SIZES=(1024 2048 4096 8192 16384 32768 65536 131072 262144)

for cut in "${CUT_VALS[@]}"; do
    for oct in "${OCT_VALS[@]}"; do
        for buf in "${BUFFER_SIZES[@]}"; do
            echo "🧪 Testing: ${cut}s, ${oct} oct, Buffer: ${buf}" >> "$STREAM_LOG"
            
            # 🎯 FIX: Proper binding of directories and correct path to the python probe
            singularity exec --no-home \
                --bind "$PROJECT_DIR:$PROJECT_DIR" \
                --bind "$TMP_DIR:$TMP_DIR" \
                --bind "$RESULTS_DIR:$RESULTS_DIR" \
                "$SIF_FILE" \
                python3 -u "$TMP_DIR/probe_buffering.py" "$RAW_DATA" "$STREAM_LOG" "$cut" "$oct" "$buf"
        done
    done
done
EOF

# --- 6. PLOTTING RESULTS ---
echo -e "\n📊 Generating plots in $RESULTS_DIR..."

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

df = pd.read_csv(raw_data)
df['Label'] = df['Cut_Secs'].astype(str) + "s_" + df['N_Octave'].astype(str) + "oct"
stats = df.groupby(['Label', 'Buffer_Size'])['Wall_Time'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(12, 8))
markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h']
unique_labels = stats['Label'].unique()

for i, label in enumerate(unique_labels):
    sub = stats[stats['Label'] == label]
    m = markers[i % len(markers)]
    plt.errorbar(sub['Buffer_Size'], sub['mean'], yerr=sub['std'], label=label, marker=m, capsize=3)

plt.xscale('log', base=2)
plt.grid(True, which="both", linestyle='--', alpha=0.5)
plt.xlabel("Buffer Size (Samples)")
plt.ylabel("Wall Time (s)")
plt.title("I/O Buffering Performance Analysis")
plt.legend(title="Configurations", bbox_to_anchor=(1.05, 1), loc='upper left')

plot_path = os.path.join(res_dir, "buffering_curves_analysis.png")
plt.savefig(plot_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Plot saved to: {plot_path}")
PY_PLOT

rm -rf "$TMP_DIR"
echo "🧹 Process finished."
