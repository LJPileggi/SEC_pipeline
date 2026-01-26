#!/bin/bash
# ==========================================================
# RIGOROUS SEQUENTIAL SCHEDULER FOR LEONARDO CLUSTER
# Strategy: "Absolute Path Injection & Environment Parity"
# ==========================================================

DIRECTIVES_FILE=$1
CURRENT_USER=$(whoami)

if [ ! -f "$DIRECTIVES_FILE" ]; then
    echo "❌ Error: Directives file '$DIRECTIVES_FILE' not found."
    exit 1
fi

# --- 1. HEADER PARSING (With literal $USER replacement) ---
echo "📖 Parsing Global Header..."

parse_path() {
    local raw_val=$(grep "$1" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
    # Force replacement of literal $USER string with the actual username
    echo "$raw_val" | sed "s/\$USER/$CURRENT_USER/g"
}

SIF_FILE=$(parse_path "SIF_FILE")
CLAP_SCRATCH_WEIGHTS=$(parse_path "CLAP_WEIGHTS")
ROBERTA_PATH=$(parse_path "ROBERTA_PATH")
DATASEC_GLOBAL=$(parse_path "DATASEC_GLOBAL")

SLURM_ACCOUNT=$(grep "SLURM_ACCOUNT" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_PARTITION=$(grep "SLURM_PARTITION" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_TIME=$(grep "SLURM_TIME" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)

# --- 2. TASK SUBMISSION FUNCTION ---
submit_and_wait_task() {
    local c_file=$1; local fmt=$2; local oct=$3
    local j_name="emb_${fmt}_o${oct}"
    local script="submit_${j_name}.sh"
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

# 🎯 PATH DEFINITION
TEMP_DIR="/leonardo_scratch/large/userexternal/$CURRENT_USER/tmp_job_\$SLURM_JOB_ID"
TARGET_GLOBAL="$FINAL_DEST"

# 🛠️ DIRECTORY SETUP (Mirroring run_embedding_pipeline.sh)
mkdir -p "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$fmt"
mkdir -p "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/${oct}_octave"
mkdir -p "\$TEMP_DIR/work_dir/weights"
mkdir -p "\$TEMP_DIR/roberta-base"
mkdir -p "\$TEMP_DIR/numba_cache"

# 🛡️ SIGNAL HANDLING & CONSOLIDATION
finalize_and_cleanup() {
    trap - SIGTERM SIGINT
    echo "⚠️ Signal or End-of-Run caught! Starting consolidation..."
    if [ -d "\$TEMP_DIR" ]; then
        # 🔗 LOG MERGING
        singularity exec --no-home --bind "\$TEMP_DIR:/tmp_data" --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
            python3 "/tmp_data/work_dir/join_logs_wrapper.py"
        
        # 🔗 HDF5 JOINING
        singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "\$TEMP_DIR:/tmp_data" \\
            --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
            python3 scripts/join_hdf5.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt"
        
        # 📦 FINAL STAGE-OUT
        mkdir -p "\$TARGET_GLOBAL"
        rsync -rlt "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/${oct}_octave/" "\$TARGET_GLOBAL/"
        rm -rf "\$TEMP_DIR"
    fi
    exit 0
}
trap 'finalize_and_cleanup' SIGTERM SIGINT

# 📦 STAGE-IN (Restored full logic)
echo "📦 Staging data..."
cp "$CLAP_SCRATCH_WEIGHTS" "\$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"
cp -r "$ROBERTA_PATH/." "\$TEMP_DIR/roberta-base/"
[ -d "\$TARGET_GLOBAL" ] && cp -r "\$TARGET_GLOBAL/." "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/${oct}_octave/"
cp "$DATASEC_GLOBAL/RAW_DATASET/raw_$fmt"/*.h5 "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$fmt/" 2>/dev/null

# 📝 LOG WRAPPER GENERATION
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

# 🚀 ENVIRONMENT & EXECUTION (Full Parity Restored)
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_ALLOC_CONF=expandable_segments:True
export MASTER_ADDR=\$(hostname)
export MASTER_PORT=\$(expr 20000 + \${SLURM_JOB_ID} % 10000)

echo "🚀 Starting Parallel Embedding Pipeline..."
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv --no-home --bind "/leonardo_scratch:/leonardo_scratch" --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt"

finalize_and_cleanup
EOF

    chmod +x "$script"
    echo "⏳ Submitting $j_name and waiting for completion..."
    sbatch --wait "$script"
}

# --- 3. TASK DISPATCHER ---
echo "🚀 Dispatching tasks SEQUENTIALLY..."
sed -n '/^[^#]/p' "$DIRECTIVES_FILE" | grep "|" | while IFS='|' read -r cfg fmt oct; do
    cfg=$(echo $cfg | xargs); fmt=$(echo $fmt | xargs); oct=$(echo $oct | xargs)
    
    # Header check
    [[ "$cfg" == *"SIF_FILE"* || "$cfg" == *"CLAP_WEIGHTS"* || "$cfg" == *"ROBERTA_PATH"* || "$cfg" == *"SLURM_"* || "$cfg" == *"DATASEC_GLOBAL"* ]] && continue
    [ -z "$cfg" ] && continue

    echo "➡️ Current Task: $cfg | $fmt | $oct"
    submit_and_wait_task "$cfg" "$fmt" "$oct"
    echo "✅ Task finished."
done
