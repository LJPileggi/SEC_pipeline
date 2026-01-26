#!/bin/bash
# ==========================================================
# ADVANCED CAMPAIGN SCHEDULER FOR LEONARDO CLUSTER
# Mantra: "Non toccare quello che già funziona, potenzia quello che serve"
# Updated: Full integration with Signal Traps, NVMe Staging & Modular Consolidation
# ==========================================================

DIRECTIVES_FILE=$1

if [ ! -f "$DIRECTIVES_FILE" ]; then
    echo "❌ Error: Directives file '$DIRECTIVES_FILE' not found."
    exit 1
fi

# --- 1. HEADER PARSING ---
# We extract global parameters once to inject them into individual task scripts
echo "📖 Parsing Global Header..."
SIF_FILE=$(grep "SIF_FILE" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
CLAP_SCRATCH_WEIGHTS=$(grep "CLAP_WEIGHTS" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
ROBERTA_PATH=$(grep "ROBERTA_PATH" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_ACCOUNT=$(grep "SLURM_ACCOUNT" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_PARTITION=$(grep "SLURM_PARTITION" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_TIME=$(grep "SLURM_TIME" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
DATASEC_GLOBAL=$(grep "DATASEC_GLOBAL" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
DATASEC_GLOBAL=${DATASEC_GLOBAL:-"/leonardo_scratch/large/userexternal/$USER/dataSEC"}

SCHEDULING_MODE=$(grep "SCHEDULING_MODE" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
GLOBAL_MODE=${SCHEDULING_MODE:-"sequential"}

echo "🕒 Scheduling Mode: $GLOBAL_MODE"

# --- 2. TASK SUBMISSION FUNCTION ---
submit_task() {
    local c_file=$1; local fmt=$2; local oct=$3; local dep=$4
    local j_name="emb_${fmt}_o${oct}"
    local script="submit_${j_name}.sh"
    
    # 🎯 FIX: Absolute destination resolved at submission time to prevent path errors
    local FINAL_DEST="$DATASEC_GLOBAL/PREPROCESSED_DATASET/$fmt/${oct}_octave"

    cat << EOF > "$script"
#!/bin/bash
#SBATCH --job-name=$j_name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=$SLURM_TIME
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p $SLURM_PARTITION
#SBATCH -A $SLURM_ACCOUNT
#SBATCH --output=%x_%j.out

# 🎯 1. PATH DEFINITION (Absolute scratch paths for NVMe Staging)
TEMP_DIR="/leonardo_scratch/large/userexternal/\$USER/tmp_job_\$SLURM_JOB_ID"
TARGET_GLOBAL="$FINAL_DEST"

mkdir -p "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$fmt"
mkdir -p "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/${oct}_octave"
mkdir -p "\$TEMP_DIR/work_dir/weights"
mkdir -p "\$TEMP_DIR/roberta-base"
mkdir -p "\$TEMP_DIR/numba_cache"

# 🎯 2. MODULAR SHUTDOWN FUNCTION (The Core Fix: Resilience)
finalize_and_cleanup() {
    trap - SIGTERM SIGINT
    echo "⚠️ Signal or End-of-Run caught! Starting consolidation..."
    
    if [ -d "\$TEMP_DIR" ]; then
        # A. LOG MERGING (Uses incremental logic from utils.py)
        echo "🔗 Merging rank-specific logs..."
        singularity exec --no-home --bind "\$TEMP_DIR:/tmp_data" --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
            python3 "/tmp_data/work_dir/join_logs_wrapper.py"

        # B. HDF5 JOINING (Overwrites combined_*.h5 to ensure integrity)
        echo "🔗 Joining HDF5 files..."
        singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "\$TEMP_DIR:/tmp_data" \\
            --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
            python3 scripts/join_hdf5.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt"

        # C. FINAL STAGE-OUT (Using safe rsync flags for Leonardo)
        echo "📦 Consolidating final results to global storage..."
        mkdir -p "\$TARGET_GLOBAL"
        rsync -rlt "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/${oct}_octave/" "\$TARGET_GLOBAL/"
        
        # D. CLEANUP
        rm -rf "\$TEMP_DIR"
    fi
    echo "✅ Task consolidation completed."
    exit 0
}
trap 'finalize_and_cleanup' SIGTERM SIGINT

# 🎯 3. STAGE-IN: Flattening raw dataset and copying assets
echo "📦 Staging-in data and weights..."
cp "$CLAP_SCRATCH_WEIGHTS" "\$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
cp -r "$ROBERTA_PATH/." "\$TEMP_DIR/roberta-base/"

# Sync existing embeddings to local NVMe for O(1) resumability
if [ -d "\$TARGET_GLOBAL" ]; then
    cp -r "\$TARGET_GLOBAL/." "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/${oct}_octave/"
fi

# Copy raw audio (targeted class-based copy)
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_$fmt"/*.h5 "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$fmt/" 2>/dev/null

# 🎯 PRE-CREATE LOG WRAPPER (To avoid here-doc issues during trap)
cat << 'INNER_EOF' > "\$TEMP_DIR/work_dir/join_logs_wrapper.py"
import sys, os
sys.path.append('/app')
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed
def main():
    base_path = os.path.join(basedir_preprocessed, "$fmt", "${oct}_octave")
    if not os.path.exists(base_path): return
    for entry in os.listdir(base_path):
        target_dir = os.path.join(base_path, entry)
        if os.path.isdir(target_dir) and entry.endswith("_secs"):
            join_logs(target_dir)
if __name__ == '__main__': main()
INNER_EOF

# 🎯 4. ENVIRONMENT REDIRECTION (The Mantra)
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=\$(hostname)
export MASTER_PORT=\$(expr 20000 + \${SLURM_JOB_ID} % 10000)

# 🎯 5. PIPELINE EXECUTION
echo "🚀 Starting Parallel Embedding Pipeline..."
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv --no-home \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt"

# 🎯 6. NATURAL END (Same logic as trap)
finalize_and_cleanup
EOF

    chmod +x "$script"
    if [ -z "$dep" ]; then
        job_id=$(sbatch --parsable "$script")
    else
        job_id=$(sbatch --parsable --dependency=afterok:"$dep" "$script")
    fi
    echo "$job_id"
}

# --- 3. TASK DISPATCHER ---
echo "🚀 Dispatching campaign tasks..."
LAST_JOB_ID=""

sed -n '/^[^#]/p' "$DIRECTIVES_FILE" | grep "|" | while IFS='|' read -r cfg fmt oct; do
    cfg=$(echo $cfg | xargs); fmt=$(echo $fmt | xargs); oct=$(echo $oct | xargs)
    
    # Skip headers and empty lines
    [[ "$cfg" == *"SIF_FILE"* || "$cfg" == *"CLAP_WEIGHTS"* || "$cfg" == *"ROBERTA_PATH"* || "$cfg" == *"SLURM_"* || "$cfg" == *"SCHEDULING_MODE"* || "$cfg" == *"DATASEC_GLOBAL"* ]] && continue
    [ -z "$cfg" ] && continue

    echo "Processing: $cfg | Format: $fmt | Octave: $oct"
    
    PREV_ID=""
    if [ "$GLOBAL_MODE" == "sequential" ]; then
        PREV_ID=$LAST_JOB_ID
    fi
    
    LAST_JOB_ID=$(submit_task "$cfg" "$fmt" "$oct" "$PREV_ID")
    echo "   - Submitted Job ID: $LAST_JOB_ID"
done

echo "🏁 All tasks scheduled in $GLOBAL_MODE mode."
