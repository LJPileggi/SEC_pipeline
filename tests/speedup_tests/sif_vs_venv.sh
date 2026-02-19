#!/bin/bash

# --- 1. CONFIGURATION (Naming conventions from ground truth) ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
# External results directory as per previous agreement
RESULTS_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline_speedup_results"
TMP_DIR="${PROJECT_DIR}/.tmp"

STREAM_LOG="${TMP_DIR}/marathon_benchmark.log"
# Cumulative CSV for all runs
FINAL_CSV="${RESULTS_DIR}/sif_vs_venv_raw_data.csv"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
VENV_PATH="${TMP_DIR}/venv_benchmark"
SLURM_SCRIPT="${TMP_DIR}/single_run_task.sh"

# Marathon parameters
TOTAL_RUNS=20
SLEEP_BETWEEN_RUNS=300 # 5 minutes to mitigate cluster caching

# --- 2. SETUP ---
mkdir -p "$TMP_DIR"
mkdir -p "$RESULTS_DIR"
touch "$STREAM_LOG"

# Initialize CSV with header if it doesn't exist
if [ ! -f "$FINAL_CSV" ]; then
    echo "Run_ID,Node,Mode,Library,Time_s" > "$FINAL_CSV"
fi

# --- 3. VENV SETUP (Only once at the beginning) ---
echo "🔧 Preparing environment..."
module purge
module load profile/base
module load python
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
    source "$VENV_PATH/bin/activate"
    pip install --upgrade pip >> /dev/null 2>&1
    pip install msclap numpy pandas h5py scipy librosa soundfile transformers torch >> /dev/null 2>&1
    deactivate
fi

# --- 4. PROBE CREATION (Outputting CSV-friendly lines) ---
cat << 'EOF' > "${TMP_DIR}/probe_csv.py"
import time, sys, os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['MPLCONFIGDIR'] = f'/tmp/matplotlib_{os.getpid()}'

run_id = sys.argv[1]
node_name = sys.argv[2]
mode = sys.argv[3]
modules = ["numpy", "pandas", "h5py", "scipy", "librosa", "soundfile", "transformers", "torch", "msclap"]

for mod in modules:
    start = time.perf_counter()
    try:
        __import__(mod)
        end = time.perf_counter()
        # Format: Run_ID,Node,Mode,Library,Time_s
        print(f"{run_id},{node_name},{mode},{mod},{end - start:.6f}")
    except Exception:
        print(f"{run_id},{node_name},{mode},{mod},ERROR")
EOF

# --- 5. SLURM TASK GENERATION ---
cat << 'EOF' > "$SLURM_SCRIPT"
#!/bin/bash
#SBATCH --job-name=sif_venv_marathon
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null

RUN_ID=$1; STREAM_LOG=$2; VENV_PATH=$3; TMP_DIR=$4; SIF_FILE=$5; FINAL_CSV=$6; PROJECT_DIR=$7
NODE=$(hostname)
DONE_FILE="${TMP_DIR}/run_${RUN_ID}.done"

echo "🚀 Starting Run $RUN_ID on $NODE" >> "$STREAM_LOG"

# --- VENV COLD TEST ---
source "$VENV_PATH/bin/activate"
export PYTHONPATH="$VENV_PATH/lib/python3.11/site-packages"
python3 -u "$TMP_DIR/probe_csv.py" "$RUN_ID" "$NODE" "VENV_COLD" >> "$FINAL_CSV" 2>&1
deactivate

# --- SIF COLD TEST ---
# -e ensures environment isolation to avoid PYTHONPATH pollution
singularity exec -e --nv --no-home --bind "$PROJECT_DIR:/app" --bind "$TMP_DIR:/tmp_bench" "$SIF_FILE" \
    python3 -u /tmp_bench/probe_csv.py "$RUN_ID" "$NODE" "SIF_COLD" >> "$FINAL_CSV" 2>&1

touch "$DONE_FILE"
EOF

# --- 6. ORCHESTRATOR LOOP ---
echo "🏃 Starting Marathon: $TOTAL_RUNS runs with ${SLEEP_BETWEEN_RUNS}s intervals."
tail -f "$STREAM_LOG" &
TAIL_PID=$!

for i in $(seq 1 $TOTAL_RUNS); do
    RUN_TOKEN=$(date +%H%M%S)
    CURRENT_DONE="${TMP_DIR}/run_${RUN_TOKEN}.done"
    
    echo "📤 Submitting Run $i/$TOTAL_RUNS..."
    sbatch "$SLURM_SCRIPT" "$RUN_TOKEN" "$STREAM_LOG" "$VENV_PATH" "$TMP_DIR" "$SIF_FILE" "$FINAL_CSV" "$PROJECT_DIR"
    
    # Sync via Done-File
    while [ ! -f "$CURRENT_DONE" ]; do sleep 10; done
    
    echo "✅ Run $i complete. Cooling down for ${SLEEP_BETWEEN_RUNS}s..."
    if [ $i -lt $TOTAL_RUNS ]; then sleep $SLEEP_BETWEEN_RUNS; fi
done

# --- 7. FINAL CLEANUP AND ANALYSIS ---
kill -9 "$TAIL_PID" 2>/dev/null
echo -e "\n📊 All runs completed. Data saved in $FINAL_CSV"

# Quick Python summary
singularity exec -e "$SIF_FILE" python3 -u - <<PY_SUM
import pandas as pd
df = pd.read_csv("${FINAL_CSV}")
summary = df[df['Time_s'] != 'ERROR'].copy()
summary['Time_s'] = pd.to_numeric(summary['Time_s'])
stats = summary.groupby(['Mode', 'Library'])['Time_s'].agg(['mean', 'std', 'median']).reset_index()
print("\n--- BENCHMARK SUMMARY ---")
print(stats.to_string())
stats.to_csv("${RESULTS_DIR}/summary_statistics.csv")
PY_SUM

rm -rf "$TMP_DIR"
echo "🧹 Temporary files removed. Process finished."
