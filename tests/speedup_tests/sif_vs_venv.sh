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
# Ensure the path is clean and absolute
TMP_DIR="${PROJECT_DIR}/.tmp"
STREAM_LOG="${TMP_DIR}/sif_vs_venv_benchmark_stream.log"
VENV_PATH="${TMP_DIR}/venv_benchmark"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
SLURM_SCRIPT="${TMP_DIR}/imports_test_slurm.sh"

# Using Python 3.10 to satisfy numba/torch requirements (>=3.8, <3.12)
PYTHON_EXEC="python3.10"

echo "🚀 Starting Benchmark Setup..."
# Clean previous failed attempts
rm -rf "$TMP_DIR"
mkdir -p "$TMP_DIR"
touch "$STREAM_LOG"

# --- 2. VIRTUAL ENVIRONMENT SETUP (Baseline) ---
echo "📦 Creating Virtual Environment on Lustre (Using $PYTHON_EXEC)..."
$PYTHON_EXEC -m venv "$VENV_PATH" || { echo "❌ Failed to create venv"; exit 1; }
source "$VENV_PATH/bin/activate"
pip install --upgrade pip > /dev/null
echo "📥 Installing requirements (this might take a few minutes)..."
pip install -r "$REQ_FILE" || { echo "❌ Failed to install requirements"; deactivate; exit 1; }
deactivate
echo "✅ Baseline environment ready."

# --- 3. GENERATE SLURM BATCH SCRIPT ---
cat <<EOF > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=import_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null 

log_bench() {
    echo -e "\$1" | tee -a "$STREAM_LOG"
}

# --- Python Probe Generation ---
cat <<PY > "$TMP_DIR/probe_imports.py"
import time
import sys
import os

def benchmark_imports(label):
    modules = ['torch', 'numpy', 'pandas', 'msclap', 'transformers', 'scipy', 'h5py', 'librosa', 'soundfile']
    print(f"\n--- BENCHMARK: {label} ---")
    total_start = time.perf_counter()
    for mod in modules:
        start = time.perf_counter()
        try:
            __import__(mod)
            end = time.perf_counter()
            print(f"  {mod:15}: {end - start:.4f}s")
        except ImportError:
            print(f"  {mod:15}: FAILED")
    print(f"TOTAL IMPORT TIME: {time.perf_counter() - total_start:.4f}s")

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    benchmark_imports(mode)
PY

log_bench "\n🧪 STARTING TEST A: Virtual Environment (Lustre)"
source "$VENV_PATH/bin/activate"
python3 "$TMP_DIR/probe_imports.py" "VENV_ON_LUSTRE" | tee -a "$STREAM_LOG"
deactivate

log_bench "\n🧪 STARTING TEST B: Singularity Container (SIF)"
singularity exec --nv --no-home \\
    --bind "$PROJECT_DIR:/app" \\
    --bind "$TMP_DIR:/tmp_bench" \\
    "$SIF_FILE" \\
    python3 /tmp_bench/probe_imports.py "CONTAINER_SIF" | tee -a "$STREAM_LOG"
EOF

# --- 4. SUBMISSION AND REAL-TIME MONITORING ---
echo "📤 Submitting Job to SLURM..."
JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT")
if [ $? -ne 0 ]; then echo "❌ sbatch submission failed"; exit 1; fi

echo "📊 MONITORING JOB $JOB_ID (Real-time stream):"
echo "----------------------------------------------------------------"

tail -f "$STREAM_LOG" &
TAIL_PID=$!

# Improved Job Monitoring Loop
while true; do
    # Check job status via sacct for high reliability
    STATUS=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
    
    case "$STATUS" in
        RUNNING|PENDING|REQUEUED|COMPLETING)
            # Job is still active
            sleep 3
            ;;
        COMPLETED)
            # Success
            break
            ;;
        FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF_MEMORY)
            echo -e "\n❌ FATAL: Job $JOB_ID terminated with status: $STATUS"
            kill $TAIL_PID 2>/dev/null
            exit 1
            ;;
        *)
            # Fallback for empty status during transition
            sleep 2
            ;;
    esac
done

sleep 2
kill $TAIL_PID 2>/dev/null
echo -e "\n----------------------------------------------------------------"
echo "✅ Job $JOB_ID finished successfully."

# --- 5. CLEANUP ---
echo "🧹 Cleaning up..."
rm -rf "$TMP_DIR"
echo "✨ Benchmark cycle completed."
