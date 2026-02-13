#!/bin/bash

# ==============================================================================
# HPC BENCHMARK: VIRTUAL ENV (LUSTRE) VS. SINGULARITY CONTAINER (SIF)
# 
# This script measures Python import latencies to quantify the I/O efficiency 
# gains of using a single-file container image over thousands of scattered 
# library files on a parallel filesystem like Lustre.
# ==============================================================================

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TMP_DIR="${PROJECT_DIR}/.tmp"
STREAM_LOG="${TMP_DIR}/sif_vs_venv_benchmark_stream.log"
VENV_PATH="${TMP_DIR}/venv_benchmark"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
SLURM_SCRIPT="${TMP_DIR}/imports_test_slurm.sh"

# --- 2. SETUP CLEANUP TRAP ---
# Ora gestisce anche SIGINT (Keyboard Interrupt) e segnali di terminazione
cleanup() {
    echo -e "\n🧹 Cleaning up benchmark environment: $TMP_DIR"
    rm -rf "$TMP_DIR"
    # Kill tail process if still running
    if [ ! -z "$TAIL_PID" ]; then kill "$TAIL_PID" 2>/dev/null; fi
}
trap cleanup EXIT SIGTERM SIGINT

# Load the native Cineca Python module
echo "🔧 Loading CINECA Python module..."
module purge
module load profile/base
module load python/3.11.6--gcc--8.5.0 || { echo "❌ Failed to load Python module"; exit 1; }

echo "🚀 Starting Benchmark Setup..."
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
touch "$STREAM_LOG"

# --- 3. VIRTUAL ENVIRONMENT SETUP (Baseline) ---
echo "📦 Creating Virtual Environment on Lustre..."
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install --upgrade pip
pip install -r "$REQ_FILE"
deactivate

# --- 4. PREPARE PROBE SCRIPT ---
cat << 'EOF' > "${TMP_DIR}/probe_imports.py"
import time
import sys
import os

mode = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
modules = ["torch", "msclap", "transformers", "librosa", "soundfile", "numpy", "pandas", "h5py"]

print(f"\n--- Python Results for {mode} ---")
for mod in modules:
    start = time.perf_counter()
    try:
        __import__(mod)
        end = time.perf_counter()
        print(f"🔹 {mod:<15}: {end - start:.4f}s")
    except Exception as e:
        print(f"❌ {mod:<15}: ERROR ({e})")
print(f"--- End of {mode} ---\n")
EOF

# --- 5. GENERATE SLURM BATCH SCRIPT ---
cat << EOF > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=sif_vs_venv
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=00:45:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

echo "🧪 STARTING TEST A: Virtual Environment (Lustre)" | tee -a "$STREAM_LOG"
source "$VENV_PATH/bin/activate"
# -u per unbuffered output, 2>&1 per catturare stderr
python3 -u "$TMP_DIR/probe_imports.py" "VENV_LUSTRE" >> "$STREAM_LOG" 2>&1
deactivate

echo "🧪 STARTING TEST B: Singularity Container (SIF)" | tee -a "$STREAM_LOG"
# -u per unbuffered output, 2>&1 per catturare stderr
singularity exec --nv --no-home \\
    --bind "$PROJECT_DIR:/app" \\
    --bind "$TMP_DIR:/tmp_bench" \\
    "$SIF_FILE" \\
    python3 -u /tmp_bench/probe_imports.py "CONTAINER_SIF" >> "$STREAM_LOG" 2>&1
EOF

# --- 6. SUBMISSION AND REAL-TIME MONITORING ---
echo "📤 Submitting Job to SLURM..."
JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT")
if [ $? -ne 0 ]; then echo "❌ sbatch submission failed"; exit 1; fi

echo "📊 MONITORING JOB $JOB_ID (Real-time stream):"
echo "----------------------------------------------------------------"

tail -f "$STREAM_LOG" &
TAIL_PID=$!

while true; do
    STATUS=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
    case "$STATUS" in
        RUNNING|PENDING|REQUEUED|COMPLETING)
            sleep 3
            ;;
        COMPLETED)
            break
            ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
            echo -e "\n❌ FATAL: Job $JOB_ID terminated with status: $STATUS"
            exit 1
            ;;
        *)
            sleep 2
            ;;
    esac
done

sleep 2
kill $TAIL_PID 2>/dev/null
echo -e "\n✅ Benchmark Complete."
