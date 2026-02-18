#!/bin/bash

# --- 1. CONFIGURATION (Identica all'originale) ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TMP_DIR="${PROJECT_DIR}/.tmp"
STREAM_LOG="${TMP_DIR}/sif_vs_venv_benchmark_stream.log"
VENV_PATH="${TMP_DIR}/venv_benchmark"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
SLURM_SCRIPT="${TMP_DIR}/imports_test_slurm.sh"
BLACKLIST_FILE="${TMP_DIR}/node_blacklist.txt"

# --- 2. SETUP CLEANUP TRAP ---
cleanup() {
    if [ ! -z "$TAIL_PID" ]; then kill "$TAIL_PID" 2>/dev/null; fi
    # Il TMP_DIR viene rimosso solo alla fine del loop principale
}
trap cleanup EXIT SIGTERM SIGINT

# Creazione cartella e log iniziale per evitare errori di tail
mkdir -p "$TMP_DIR"
touch "$BLACKLIST_FILE"
touch "$STREAM_LOG"

# --- 3. VENV SETUP (Eseguito interamente prima di procedere) ---
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
    echo "✅ VENV creation completed."
fi

# --- 4. PROBE CREATION (Output pulito) ---
cat << 'EOF' > "${TMP_DIR}/probe_imports.py"
import time
import sys
import os

# Silencing TensorFlow and Matplotlib noise
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MPLCONFIGDIR'] = '/tmp/matplotlib_cache'

mode = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
modules = ["numpy", "pandas", "h5py", "scipy", "librosa", "soundfile", "transformers", "torch", "msclap"]

results = []
for mod in modules:
    # Print current library being processed to stdout (for real-time tracking)
    print(f"   ⌛ Importing {mod}...", file=sys.stderr, flush=True)
    start = time.perf_counter()
    try:
        __import__(mod)
        end = time.perf_counter()
        results.append(f"🔹 {mod:<15}: {end - start:.4f}s")
    except Exception as e:
        results.append(f"❌ {mod:<15}: ERROR ({e})")

print(f"\n--- Python Results for {mode} ---")
for r in results:
    print(r)
print(f"--- End of {mode} ---\n", flush=True)
EOF

# --- 5. SLURM SCRIPT GENERATION ---
cat << 'EOF' > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=sif_vs_venv_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite

STREAM_LOG=$1; VENV_PATH=$2; TMP_DIR=$3; SIF_FILE=$4; BLACKLIST_FILE=$5; PROJECT_DIR=$6

# Set env vars to keep output clean inside the job
export TF_CPP_MIN_LOG_LEVEL=3
export MPLCONFIGDIR=/tmp/matplotlib_$(hostname)

CURRENT_NODE=$(hostname)
echo -e "\n🚀 NODE START: $(date) on $CURRENT_NODE" >> "$STREAM_LOG"

# --- TEST A: Virtual Environment ---
source "$VENV_PATH/bin/activate"
python3 -u "$TMP_DIR/probe_imports.py" "VENV_COLD" >> "$STREAM_LOG" 2>&1
python3 -u "$TMP_DIR/probe_imports.py" "VENV_WARM" >> "$STREAM_LOG" 2>&1
deactivate

# --- TEST B: Singularity Container ---
singularity exec --nv --no-home \
    --bind "$PROJECT_DIR:/app" \
    --bind "$TMP_DIR:/tmp_bench" \
    "$SIF_FILE" \
    python3 -u /tmp_bench/probe_imports.py "SIF_COLD" >> "$STREAM_LOG" 2>&1

singularity exec --nv --no-home \
    --bind "$PROJECT_DIR:/app" \
    --bind "$TMP_DIR:/tmp_bench" \
    "$SIF_FILE" \
    python3 -u /tmp_bench/probe_imports.py "SIF_WARM" >> "$STREAM_LOG" 2>&1

echo "$CURRENT_NODE" >> "$BLACKLIST_FILE"
echo "🏁 NODE FINISH: $(date)" >> "$STREAM_LOG"
EOF

# --- 6. ORCHESTRATOR LOOP ---
echo "📊 STARTING SCIENTIFIC PROTOCOL (5 NODES)..."
tail -f "$STREAM_LOG" &
TAIL_PID=$!

for i in {1..5}; do
    EXCLUDE_NODES=$(paste -sd "," "$BLACKLIST_FILE" 2>/dev/null)
    
    SBATCH_OPTS="--parsable"
    if [ ! -z "$EXCLUDE_NODES" ]; then
        SBATCH_OPTS="$SBATCH_OPTS --exclude=$EXCLUDE_NODES"
    fi
    
    # Submitting step
    echo -e "\n--- Submitting Step $i/5 ---" >> "$STREAM_LOG"
    JOB_ID=$(sbatch $SBATCH_OPTS "$SLURM_SCRIPT" "$STREAM_LOG" "$VENV_PATH" "$TMP_DIR" "$SIF_FILE" "$BLACKLIST_FILE" "$PROJECT_DIR")
    
    # CRITICAL: Wait for current job to FINISH before loop continues
    while true; do
        STATUS=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
        case "$STATUS" in
            COMPLETED|FAILED|TIMEOUT|CANCELLED) break ;;
            *) sleep 15 ;;
        esac
    done
done

# Cleanup finale manuale
rm -rf "$TMP_DIR"
echo -e "\n✅ Benchmark concluso."
