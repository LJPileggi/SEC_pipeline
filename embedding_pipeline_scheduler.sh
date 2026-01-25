#!/bin/bash

# ==========================================================
# ADVANCED CAMPAIGN SCHEDULER FOR LEONARDO CLUSTER
# Mantra: "Non toccare quello che già funziona"
# Updated: Integrated NVMe Staging & Memory Stability
# ==========================================

DIRECTIVES_FILE=$1

if [ ! -f "$DIRECTIVES_FILE" ]; then
    echo "❌ Error: Directives file '$DIRECTIVES_FILE' not found."
    exit 1
fi

# --- HEADER PARSING ---
echo "📖 Parsing Global Header..."
SIF_FILE=$(grep "SIF_FILE" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
CLAP_SCRATCH_WEIGHTS=$(grep "CLAP_WEIGHTS" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
ROBERTA_PATH=$(grep "ROBERTA_PATH" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_ACCOUNT=$(grep "SLURM_ACCOUNT" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_PARTITION=$(grep "SLURM_PARTITION" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_TIME=$(grep "SLURM_TIME" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
DATASEC_GLOBAL=$(grep "DATASEC_GLOBAL" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
# Default if not specified: frère of SEC_pipeline
DATASEC_GLOBAL=${DATASEC_GLOBAL:-"/leonardo_scratch/large/userexternal/$USER/dataSEC"}

SCHEDULING_MODE=$(grep "SCHEDULING_MODE" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
GLOBAL_MODE=${SCHEDULING_MODE:-"sequential"}

echo "🕒 Mode: $GLOBAL_MODE"

submit_task() {
    local c_file=$1; local fmt=$2; local oct=$3; local dep=$4
    local j_name="emb_${fmt}_o${oct}"
    local script="submit_${j_name}.sh"

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
#SBATCH -A $SLUR_ACCOUNT
#SBATCH --output=%x_%j.out

# 🎯 1. PATH DEFINITION (Absolute scratch paths for NVMe Staging)
TEMP_DIR="/leonardo_scratch/large/userexternal/\$USER/tmp_job_\$SLURM_JOB_ID"
mkdir -p "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$fmt"
mkdir -p "\$TEMP_DIR/work_dir/weights"
mkdir -p "\$TEMP_DIR/roberta-base"
mkdir -p "\$TEMP_DIR/numba_cache"

# 🎯 2. STAGE-IN: Flattening raw dataset and copying assets
echo "📦 Staging-in data..."
cp "$DATASEC_GLOBAL/RAW_DATASET"/*/*.h5 "\$TEMP_DIR/dataSEC/RAW_DATASET/raw_$fmt/"
cp -r "$ROBERTA_PATH/." "\$TEMP_DIR/roberta-base/"
cp "$CLAP_SCRATCH_WEIGHTS" "\$TEMP_DIR/work_dir/weights/CLAP_weights_2023.pth"

# 🎯 3. ENVIRONMENT REDIRECTION (The Mantra)
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/weights/CLAP_weights_2023.pth"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export MASTER_PORT=\$(expr 20000 + \${SLURM_JOB_ID} % 10000)

# 🎯 4. PIPELINE EXECUTION
echo "🚀 Starting Parallel Embedding Pipeline..."
srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt"

# 🎯 5. LOG MERGING (Essential for coherent join)
echo "🔗 Merging rank-specific logs..."
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

singularity exec --bind "\$TEMP_DIR:/tmp_data" --bind "\$(pwd):/app" --pwd "/app" "$SIF_FILE" \\
    python3 "/tmp_data/work_dir/join_logs_wrapper.py"

# 🎯 6. FINAL JOINING
echo "🔗 Joining HDF5 files..."
singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$TEMP_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/join_hdf5.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt"

# 🎯 7. STAGE-OUT
echo "📦 Staging-out results..."
TARGET_GLOBAL="$DATASEC_GLOBAL/PREPROCESSED_DATASET/$fmt/${oct}_octave"
mkdir -p "\$TARGET_GLOBAL"
cp -r "\$TEMP_DIR/dataSEC/PREPROCESSED_DATASET/$fmt/${oct}_octave/." "\$TARGET_GLOBAL/"

# Cleanup
rm -rf "\$TEMP_DIR"
EOF

    chmod +x "$script"
    if [ -z "$dep" ]; then
        job_id=$(sbatch --parsable "$script")
    else
        job_id=$(sbatch --parsable --dependency=afterok:"$dep" "$script")
    fi
    echo "$job_id"
}

# --- TASK DISPATCHER ---
echo "🚀 Dispatching tasks..."
LAST_JOB_ID=""

sed -n '/^[^#]/p' "$DIRECTIVES_FILE" | grep "|" | while IFS='|' read -r cfg fmt oct; do
    cfg=$(echo $cfg | xargs); fmt=$(echo $fmt | xargs); oct=$(echo $oct | xargs)
    
    # Header check
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
