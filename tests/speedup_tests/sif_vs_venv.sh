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
#SBATCH --time=05:00:00
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

# Quick Python summary and Plotting (Matplotlib only)
singularity exec -e --bind "$RESULTS_DIR:$RESULTS_DIR" "$SIF_FILE" python3 -u - <<PY_SUM
import pandas as pd
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

# Percorso del file CSV
csv_path = "${FINAL_CSV}"
if not os.path.exists(csv_path):
    print(f"❌ Error: File {csv_path} not found inside the container.")
    sys.exit(1)

df = pd.read_csv(csv_path)

# Pulizia e conversione dati
summary = df[df['Time_s'] != 'ERROR'].copy()
summary['Time_s'] = pd.to_numeric(summary['Time_s'], errors='coerce')
summary = summary.dropna(subset=['Time_s'])

if summary.empty:
    print("⚠️ No valid data found to aggregate.")
else:
    # 1. Generazione Statistiche Testuali
    stats = summary.groupby(['Mode', 'Library'])['Time_s'].agg(['mean', 'std', 'median', 'min', 'max']).reset_index()
    print("\n--- BENCHMARK SUMMARY (Time in Seconds) ---")
    print(stats.to_string(index=False))
    
    # Salvataggio statistiche
    output_stats = os.path.join("${RESULTS_DIR}", "summary_statistics.csv")
    stats.to_csv(output_stats, index=False)
    print(f"\n✅ Summary statistics saved to: {output_stats}")

    # 2. Creazione Boxplot con Matplotlib (Analisi Variabilità)
    try:
        target_libs = ['transformers', 'msclap', 'pandas', 'numpy']
        
        # Prepariamo le liste di dati per il plot
        venv_data = [summary[(summary['Library'] == lib) & (summary['Mode'] == 'VENV_COLD')]['Time_s'].values for lib in target_libs]
        sif_data = [summary[(summary['Library'] == lib) & (summary['Mode'] == 'SIF_COLD')]['Time_s'].values for lib in target_libs]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Posizionamento dei box: uno accanto all'altro per ogni libreria
        positions = np.arange(len(target_libs))
        width = 0.35
        
        # Plot dei due gruppi
        bp_venv = ax.boxplot(venv_data, positions=positions - width/2, widths=width, patch_artist=True)
        bp_sif = ax.boxplot(sif_data, positions=positions + width/2, widths=width, patch_artist=True)
        
        # Estetica: Colori (VENV = Rosso soft, SIF = Verde soft)
        for patch in bp_venv['boxes']:
            patch.set_facecolor('#ff9999')
        for patch in bp_sif['boxes']:
            patch.set_facecolor('#99ff99')
            
        # Configurazione assi
        ax.set_xticks(positions)
        ax.set_xticklabels(target_libs)
        ax.set_yscale('log') # Indispensabile per la scala dei tempi
        
        ax.set_title('Import Times Variability: VENV_COLD vs SIF_COLD', fontsize=15)
        ax.set_ylabel('Time (seconds) - Log Scale', fontsize=12)
        ax.set_xlabel('Library', fontsize=12)
        ax.grid(True, axis='y', linestyle='--', alpha=0.7)
        
        # Legenda manuale
        ax.legend([bp_venv["boxes"][0], bp_sif["boxes"][0]], ['VENV_COLD', 'SIF_COLD'], loc='upper right')
        
        # Salvataggio
        output_plot = os.path.join("${RESULTS_DIR}", "import_variability_boxplot.png")
        plt.savefig(output_plot, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ Boxplot (Matplotlib) visualization saved to: {output_plot}")
        
    except Exception as e:
        print(f"⚠️ Could not generate plot: {e}")
        import traceback
        traceback.print_exc()

PY_SUM

# Cleanup
rm -rf "$TMP_DIR"
echo "🧹 Temporary files removed. Process finished."
