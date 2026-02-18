#!/bin/bash

# --- 1. CONFIGURATION (Strictly following your naming conventions) ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TMP_DIR="${PROJECT_DIR}/.tmp"
# Using a unique log name per run to avoid overwriting
RUN_ID=$(date +%H%M%S)
STREAM_LOG="${TMP_DIR}/bench_${RUN_ID}.log"
VENV_PATH="${TMP_DIR}/venv_benchmark"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
SLURM_SCRIPT="${TMP_DIR}/slurm_import_task.sh"

# --- 2. SETUP ---
mkdir -p "$TMP_DIR"
touch "$STREAM_LOG"

# --- 3. VENV SETUP (Only if not present) ---
echo "🔧 Checking CINECA Python module and VENV..."
module purge
module load profile/base
module load python
if [ ! -d "$VENV_PATH" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip >> /dev/null 2>&1
    if [ -f "$REQ_FILE" ]; then
        pip install -r "$REQ_FILE" >> /dev/null 2>&1
    else
        pip install msclap numpy pandas h5py scipy librosa soundfile transformers torch >> /dev/null 2>&1
    fi
    deactivate
fi

# --- 4. PROBE CREATION (Zero Noise Policy) ---
cat << 'EOF' > "${TMP_DIR}/probe_imports.py"
import time
import sys
import os

# 🤐 Silence System Noises
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache_' + str(os.getpid())

mode = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
modules = ["numpy", "pandas", "h5py", "scipy", "librosa", "soundfile", "transformers", "torch", "msclap"]

# Result collection to print ONLY at the end
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

# --- 5. SLURM SCRIPT (Single Node Pair) ---
cat << 'EOF' > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=bench_pair
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null

STREAM_LOG=$1; VENV_PATH=$2; TMP_DIR=$3; SIF_FILE=$4; PROJECT_DIR=$5

# Clean environment inside the node
export TF_CPP_MIN_LOG_LEVEL=3
export MPLCONFIGDIR=/tmp/matplotlib_$(hostname)

echo -e "\n🚀 NODE: $(hostname) | START: $(date)" >> "$STREAM_LOG"

# --- TEST A: VENV (Cold then Warm) ---
source "$VENV_PATH/bin/activate"
python3 -u "$TMP_DIR/probe_imports.py" "VENV_COLD" >> "$STREAM_LOG" 2>&1
python3 -u "$TMP_DIR/probe_imports.py" "VENV_WARM" >> "$STREAM_LOG" 2>&1
deactivate

# --- TEST B: SIF (Cold then Warm) ---
singularity exec --nv --no-home --bind "$PROJECT_DIR:/app" --bind "$TMP_DIR:/tmp_bench" "$SIF_FILE" \
    python3 -u /tmp_bench/probe_imports.py "SIF_COLD" >> "$STREAM_LOG" 2>&1
singularity exec --nv --no-home --bind "$PROJECT_DIR:/app" --bind "$TMP_DIR:/tmp_bench" "$SIF_FILE" \
    python3 -u /tmp_bench/probe_imports.py "SIF_WARM" >> "$STREAM_LOG" 2>&1

echo "🏁 NODE FINISH: $(date)" >> "$STREAM_LOG"
EOF

# --- 6. EXECUTION ---
echo "📤 Submitting Single Pair Job (Cold/Warm)..."
JOB_ID=$(sbatch --parsable "$SLURM_SCRIPT" "$STREAM_LOG" "$VENV_PATH" "$TMP_DIR" "$SIF_FILE" "$PROJECT_DIR")

echo "📊 MONITORING JOB $JOB_ID:"
tail -f "$STREAM_LOG" &
TAIL_PID=$!

while true; do
    STATUS=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
    case "$STATUS" in
        COMPLETED|FAILED|TIMEOUT|CANCELLED) break ;;
        *) sleep 10 ;;
    esac
done

kill "$TAIL_PID" 2>/dev/null
echo -e "\n✅ Run completata. Risultati in: $STREAM_LOG"
