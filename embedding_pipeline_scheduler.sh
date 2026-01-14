#!/bin/bash

# ==========================================================
# ADVANCED CAMPAIGN SCHEDULER FOR LEONARDO CLUSTER
# Mantra: "Non toccare quello che già funziona"
# ==========================================================

DIRECTIVES_FILE=$1

if [ ! -f "$DIRECTIVES_FILE" ]; then
    echo "❌ Error: Directives file '$DIRECTIVES_FILE' not found."
    exit 1
fi

# --- HEADER PARSING ---
# Estraiamo i parametri globali mantenendo la logica di grep originale
echo "📖 Parsing Global Header..."
SIF_FILE=$(grep "SIF_FILE" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
CLAP_SCRATCH_WEIGHTS=$(grep "CLAP_WEIGHTS" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
ROBERTA_PATH=$(grep "ROBERTA_PATH" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_ACCOUNT=$(grep "SLURM_ACCOUNT" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_PARTITION=$(grep "SLURM_PARTITION" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
SLURM_TIME=$(grep "SLURM_TIME" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)

# Nuova variabile per il controllo dello scheduling
SCHEDULING_MODE=$(grep "SCHEDULING_MODE" "$DIRECTIVES_FILE" | cut -d'|' -f2 | xargs)
GLOBAL_MODE=${SCHEDULING_MODE:-"sequential"}

echo "🕒 Mode: $GLOBAL_MODE"

LAST_JOB_ID=""

submit_task() {
    local c_file=$1; local fmt=$2; local oct=$3; local dep=$4
    local j_name="emb_${fmt}_o${oct}"
    local script="submit_${j_name}.sh"

    # La struttura del sbatch script rimane IDENTICA all'originale per non rompere l'isolamento
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

# --- JOB ISOLATION SETUP ---
JOB_WORK_DIR="/leonardo_scratch/large/userexternal/\$USER/tmp_prod_\${SLURM_JOB_ID}"
mkdir -p "\$JOB_WORK_DIR/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main"
mkdir -p "\$JOB_WORK_DIR/roberta-base"

cp "$CLAP_SCRATCH_WEIGHTS" "\$JOB_WORK_DIR/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main/CLAP_weights_2023.pth"
cp -r "$ROBERTA_PATH/." "\$JOB_WORK_DIR/roberta-base/"

export HF_HOME="\$JOB_WORK_DIR/work_dir/huggingface"
export HF_HUB_OFFLINE=1
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/huggingface/hub/models--microsoft--msclap/snapshots/main/CLAP_weights_2023.pth"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC"
export PYTHONUNBUFFERED=1

srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \\
    singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$JOB_WORK_DIR:/tmp_data" \\
    --bind "\${NODE_TEMP_BASE_DIR:-"../dataSEC"}:/tmp_data/dataSEC" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/get_clap_embeddings.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt"

echo "Joining HDF5 files..."
singularity exec --nv \\
    --bind "/leonardo_scratch:/leonardo_scratch" \\
    --bind "\$JOB_WORK_DIR:/tmp_data" \\
    --bind "\$(pwd):/app" \\
    --pwd "/app" \\
    "$SIF_FILE" \\
    python3 scripts/join_hdf5.py --config_file "$c_file" --n_octave "$oct" --audio_format "$fmt"

rm -rf "\$JOB_WORK_DIR"
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

# Filtriamo le righe: saltiamo i commenti, le righe vuote e le righe dell'header
sed -n '/^[^#]/p' "$DIRECTIVES_FILE" | grep "|" | while IFS='|' read -r cfg fmt oct; do
    # Pulizia input
    cfg=$(echo $cfg | xargs); fmt=$(echo $fmt | xargs); oct=$(echo $oct | xargs)
    
    # Salta se la riga fa parte dell'header globale
    [[ "$cfg" == *"SIF_FILE"* || "$cfg" == *"CLAP_WEIGHTS"* || "$cfg" == *"ROBERTA_PATH"* || "$cfg" == *"SLURM_"* || "$cfg" == *"SCHEDULING_MODE"* ]] && continue
    [ -z "$cfg" ] && continue

    echo "Processing: $cfg | Format: $fmt | Octave: $oct"
    
    PREV_ID=""
    # Se il mantra dice che 'sequential' funziona, lo manteniamo come opzione di default
    if [ "$GLOBAL_MODE" == "sequential" ]; then
        PREV_ID=$LAST_JOB_ID
    fi
    
    LAST_JOB_ID=$(submit_task "$cfg" "$fmt" "$oct" "$PREV_ID")
    echo "   - Submitted Job ID: $LAST_JOB_ID"
done

echo "🏁 All tasks scheduled in $GLOBAL_MODE mode."
