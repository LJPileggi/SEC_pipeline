#!/bin/bash

# --- 1. CONFIGURATION (Identica all'originale) ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
TMP_DIR="${PROJECT_DIR}/.tmp"
STREAM_LOG="${TMP_DIR}/sif_vs_venv_benchmark_stream.log"
VENV_PATH="${TMP_DIR}/venv_benchmark"
REQ_FILE="${PROJECT_DIR}/requirements.txt"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
SLURM_SCRIPT="${TMP_DIR}/imports_test_slurm.sh"
# Estensione necessaria per la blacklist dei nodi
BLACKLIST_FILE="${TMP_DIR}/node_blacklist.txt"

# --- 2. SETUP CLEANUP TRAP ---
cleanup() {
    if [ ! -z "$TAIL_PID" ]; then kill "$TAIL_PID" 2>/dev/null; fi
    # Non cancelliamo TMP_DIR qui per non perdere i log durante il ciclo dei 5 job
}
trap cleanup EXIT SIGTERM SIGINT

mkdir -p "$TMP_DIR"
touch "$BLACKLIST_FILE"

# --- 3. VENV SETUP (Preso pari pari dal tuo file) ---
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
        # Fallback se requirements.txt manca, ma l'ordine è il tuo
        pip install msclap numpy pandas h5py scipy librosa soundfile transformers torch >> /dev/null 2>&1
    fi
    deactivate
fi

# --- 4. PROBE CREATION (COPIATO E INCOLLATO DAL TUO TESTO) ---
cat << 'EOF' > "${TMP_DIR}/probe_imports.py"
import time
import sys
import os

mode = sys.argv[1] if len(sys.argv) > 1 else "UNKNOWN"
modules = ["numpy", "pandas", "h5py", "scipy", "librosa", "soundfile", "transformers", "torch", "msclap"]

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

# --- 5. SLURM SCRIPT GENERATION (Protocollo Cold/Warm) ---
cat << 'EOF' > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=sif_vs_venv_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite

# Variabili passate dal ciclo orchestratore
STREAM_LOG=$1; VENV_PATH=$2; TMP_DIR=$3; SIF_FILE=$4; BLACKLIST_FILE=$5; PROJECT_DIR=$6

CURRENT_NODE=$(hostname)
echo -e "\n🚀 NODE START: $(date) on $CURRENT_NODE" >> "$STREAM_LOG"

# --- TEST A: Virtual Environment (COLD + WARM) ---
echo "🧪 [VENV] COLD START" >> "$STREAM_LOG"
source "$VENV_PATH/bin/activate"
python3 -u "$TMP_DIR/probe_imports.py" "VENV_COLD" >> "$STREAM_LOG" 2>&1
echo "🧪 [VENV] WARM START" >> "$STREAM_LOG"
python3 -u "$TMP_DIR/probe_imports.py" "VENV_WARM" >> "$STREAM_LOG" 2>&1
deactivate

# --- TEST B: Singularity Container (COLD + WARM) ---
echo "🧪 [SIF] COLD START" >> "$STREAM_LOG"
singularity exec --nv --no-home \
    --bind "$PROJECT_DIR:/app" \
    --bind "$TMP_DIR:/tmp_bench" \
    "$SIF_FILE" \
    python3 -u /tmp_bench/probe_imports.py "SIF_COLD" >> "$STREAM_LOG" 2>&1

echo "🧪 [SIF] WARM START" >> "$STREAM_LOG"
singularity exec --nv --no-home \
    --bind "$PROJECT_DIR:/app" \
    --bind "$TMP_DIR:/tmp_bench" \
    "$SIF_FILE" \
    python3 -u /tmp_bench/probe_imports.py "SIF_WARM" >> "$STREAM_LOG" 2>&1

# Registrazione del nodo nella blacklist per il prossimo job
echo "$CURRENT_NODE" >> "$BLACKLIST_FILE"
echo "🏁 NODE FINISH: $(date)" >> "$STREAM_LOG"
EOF

# --- 6. ORCHESTRATOR LOOP (5 lanci sequenziali su nodi diversi) ---
echo "📊 STARTING SCIENTIFIC PROTOCOL (5 NODES)..."
tail -f "$STREAM_LOG" &
TAIL_PID=$!

for i in {1..5}; do
    # Crea la stringa dei nodi da escludere
    EXCLUDE_NODES=$(paste -sd "," "$BLACKLIST_FILE" 2>/dev/null)
    
    SBATCH_OPTS="--parsable"
    if [ ! -z "$EXCLUDE_NODES" ]; then
        SBATCH_OPTS="$SBATCH_OPTS --exclude=$EXCLUDE_NODES"
    fi
    
    echo -e "\n--- Submitting Step $i/5 ---" >> "$STREAM_LOG"
    JOB_ID=$(sbatch $SBATCH_OPTS "$SLURM_SCRIPT" "$STREAM_LOG" "$VENV_PATH" "$TMP_DIR" "$SIF_FILE" "$BLACKLIST_FILE" "$PROJECT_DIR")
    
    # Aspetta che il job finisca prima di passare al prossimo nodo
    while true; do
        STATUS=$(sacct -j "$JOB_ID" --format=State --noheader | head -n 1 | xargs)
        case "$STATUS" in
            COMPLETED|FAILED|TIMEOUT|CANCELLED) break ;;
            *) sleep 10 ;;
        esac
    done
done

# Pulizia finale della cartella temporanea (una volta finiti tutti i job)
rm -rf "$TMP_DIR"
echo -e "\n✅ Benchmark concluso con successo."
