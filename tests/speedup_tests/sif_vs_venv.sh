#!/bin/bash

# ==============================================================================
# HPC BENCHMARK: VIRTUAL ENV (LUSTRE) VS. SINGULARITY CONTAINER (SIF)
# 
# This script measures Python import latencies to quantify the I/O efficiency 
# gains of using a single-file container image over thousands of scattered 
# library files on a parallel filesystem like Lustre.
# ==============================================================================

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/"
# Temporary directory for logs, venv, and probe scripts
TMP_DIR="$PROJECT_DIR/.tmp"
# Dedicated log file to bypass Slurm .out buffering issues
STREAM_LOG="$TMP_DIR/sif_vs_venv_benchmark_stream.log"
VENV_PATH="$TMP_DIR/venv_benchmark"
REQ_FILE="$PROJECT_DIR/requirements.txt"
SIF_FILE="$PROJECT_DIR/.containers/clap_pipeline.sif"

echo "🚀 Starting Benchmark Setup..."
mkdir -p "$TMP_DIR"
touch "$STREAM_LOG"

# --- 2. VIRTUAL ENVIRONMENT SETUP (Baseline) ---
# We create the venv on Lustre to expose the metadata bottleneck
echo "📦 Creating Virtual Environment on Lustre (Metadata stress test)..."
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
pip install --upgrade pip > /dev/null
pip install -r "$REQ_FILE" > /dev/null
deactivate
echo "✅ Baseline environment ready."

# --- 3. GENERATE SLURM BATCH SCRIPT ---
# This script will execute on the compute node and stream data back to the log
cat <<EOF > "$TMP_DIR/imports_test_slurm.sh"
#!/bin/bash
#SBATCH --job-name=import_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=00:20:00
#SBATCH --output=/dev/null # Redirect standard output to null to use our custom stream

# Internal logging function to write to our shared stream file
log_bench() {
    echo -e "\$1" | tee -a "$STREAM_LOG"
}

# --- Python Probe Generation ---
cat <<PY > "$TMP_DIR/probe_imports.py"
import time
import sys
import os

def benchmark_imports(label):
    # Core modules extracted from project requirements
    modules = ['torch', 'numpy', 'pandas', 'msclap', 'transformers', 'scipy', 'h5py', 'librosa', 'soundfile']
    results = {}
    
    print(f"\n--- BENCHMARK: {label} ---")
    total_start = time.perf_counter()
    for mod in modules:
        start = time.perf_counter()
        try:
            __import__(mod)
            end = time.perf_counter()
            latency = end - start
            print(f"  {mod:15}: {latency:.4f}s")
        except ImportError:
            print(f"  {mod:15}: FAILED")
    
    total_end = time.perf_counter()
    print(f"TOTAL IMPORT TIME: {total_end - total_start:.4f}s")

if __name__ == '__main__':
    mode = sys.argv[1] if len(sys.argv) > 1 else "Unknown"
    benchmark_imports(mode)
PY

# --- TEST A: VIRTUAL ENVIRONMENT ---
log_bench "\n🧪 STARTING TEST A: Virtual Environment (Scattered files on Lustre)"
source "$VENV_PATH/bin/activate"
python3 "$TMP_DIR/probe_imports.py" "VENV_ON_LUSTRE" | tee -a "$STREAM_LOG"
deactivate

# --- TEST B: SINGULARITY CONTAINER ---
log_bench "\n🧪 STARTING TEST B: Singularity Container (Monolithic SIF Image)"
# Using --no-home to prevent interference from local .local/lib or configs
singularity exec --nv --no-home \\
    --bind "$PROJECT_DIR:/app" \\
    --bind "$TMP_DIR:/tmp_bench" \\
    "$SIF_FILE" \\
    python3 /tmp_bench/probe_imports.py "CONTAINER_SIF" | tee -a "$STREAM_LOG"
EOF

# --- 4. SUBMISSION AND REAL-TIME STREAMING ---
echo "📤 Submitting Job to SLURM..."
JOB_ID=$(sbatch --parsable "$TMP_DIR/submit_imports_test.sh")

echo "📊 REAL-TIME STREAM FROM COMPUTE NODE (Job ID: $JOB_ID):"
echo "----------------------------------------------------------------"

# Start tail in background to stream our custom log file
tail -f "$STREAM_LOG" &
TAIL_PID=$!

# Monitor the job status
while squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
    sleep 2
done

# Allow a small buffer for the final filesystem flush
sleep 2
kill $TAIL_PID 2>/dev/null

echo -e "\n----------------------------------------------------------------"
echo "✅ Job $JOB_ID finished. All performance metrics captured."

# --- 5. CLEANUP ---
# MANTRA: Leave no trace on the shared filesystem
echo "🧹 Cleaning up temporary assets and Virtual Environment..."
rm -rf "$TMP_DIR"
echo "✨ Benchmark cycle completed successfully."
