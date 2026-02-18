#!/bin/bash

# --- 1. CONFIGURATION (Strictly your naming and paths) ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TMP_DIR="${PROJECT_DIR}/.tmp"
RUN_ID=$(date +%H%M%S)
STREAM_LOG="${TMP_DIR}/sif_vs_venv_benchmark_stream.log"
VENV_PATH="${TMP_DIR}/venv_benchmark"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
SLURM_SCRIPT="${TMP_DIR}/imports_test_slurm.sh"
DONE_FILE="${TMP_DIR}/job.done"

# --- 2. SETUP CLEANUP TRAP ---
cleanup() {
    if [ ! -z "$TAIL_PID" ]; then kill "$TAIL_PID" 2>/dev/null; fi
}
trap cleanup EXIT SIGTERM SIGINT

mkdir -p "$TMP_DIR"
touch "$STREAM_LOG"
rm -f "$DONE_FILE"

# --- 3. VENV SETUP ---
echo "🔧 Loading CINECA Python module..."
module purge
module load profile/base
module load python
if [ ! -d "$VENV_PATH" ]; then
    echo "📦 Creating benchmark virtual environment..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip >> /dev/null 2>&1
    if [ -f "$REQ_FILE" ]; then
        pip install -r "$REQ_FILE" >> /dev/null 2>&1
    else
        pip install msclap numpy pandas h5py scipy librosa soundfile transformers torch >> /dev/null 2>&1
    fi
    deactivate
    echo "✅ VENV ready."
fi

# --- 4. PROBE CREATION (Identical) ---
cat << 'EOF' > "${TMP_DIR}/probe_imports.py"
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache_' + str(os.getpid())

mode = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
modules = ["numpy", "pandas", "h5py", "scipy", "librosa", "soundfile", "transformers", "torch", "msclap"]
results = []
for mod in modules:
    start = time.perf_counter()
    try:
        __import__(mod)
        end = time.perf_counter()
        results.append(f"🔹 {mod:<15}: {end - start:.4f}s")
    except Exception as e:
        results.append(f"❌ {mod:<15}: ERROR ({e})")
print(f"\n--- Python Results for {mode} ---")
print("\n".join(results))
print(f"--- End of {mode} ---\n", flush=True)
EOF

# --- 5. SLURM SCRIPT GENERATION ---
cat << 'EOF' > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=bench_pair
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null

STREAM_LOG=$1; VENV_PATH=$2; TMP_DIR=$3; SIF_FILE=$4; PROJECT_DIR=$5; DONE_FILE=$6

echo -e "\n🚀 NODE: $(hostname) | START: $(date)" >> "$STREAM_LOG"

# --- TEST A: VENV ---
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$VENV_PATH/lib/python3.11/site-packages"
python3 -u "$TMP_DIR/probe_imports.py" "VENV_COLD" >> "$STREAM_LOG" 2>&1
python3 -u "$TMP_DIR/probe_imports.py" "VENV_WARM" >> "$STREAM_LOG" 2>&1
deactivate

# --- TEST B: SIF ---
singularity exec -e --nv --no-home --bind "$PROJECT_DIR:/app" --bind "$TMP_DIR:/tmp_bench" "$SIF_FILE" \
    python3 -u /tmp_bench/probe_imports.py "SIF_COLD" >> "$STREAM_LOG" 2>&1
singularity exec -e --nv --no-home --bind "$PROJECT_DIR:/app" --bind "$TMP_DIR:/tmp_bench" "$SIF_FILE" \
    python3 -u /tmp_bench/probe_imports.py "SIF_WARM" >> "$STREAM_LOG" 2>&1

echo "🏁 NODE FINISH: $(date)" >> "$STREAM_LOG"
touch "$DONE_FILE"
EOF

# --- 6. SUBMISSION AND MONITORING (No more sacct!) ---
echo "📤 Submitting Single Pair Job (Cold/Warm)..."
sbatch "$SLURM_SCRIPT" "$STREAM_LOG" "$VENV_PATH" "$TMP_DIR" "$SIF_FILE" "$PROJECT_DIR" "$DONE_FILE"

echo "📊 MONITORING STREAM:"
tail -f "$STREAM_LOG" &
TAIL_PID=$!

# Wait for the done file to appear
while [ ! -f "$DONE_FILE" ]; do
    sleep 10
done

kill "$TAIL_PID" 2>/dev/null

# --- 7. FINAL CLEANUP ---
echo -e "\n🧹 Final cleanup of benchmark environment: $TMP_DIR"
rm -rf "$TMP_DIR"
echo -e "✅ Process finished."
