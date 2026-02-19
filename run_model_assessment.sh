#!/bin/bash
#SBATCH --job-name=Assessment_Marathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=boost_usr_prod

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline"
DATASEC_DIR="/leonardo_scratch/large/userexternal/$USER/dataSEC"
EMB_BASE="${DATASEC_DIR}/PREPROCESSED_DATASET"
RESULTS_BASE="${DATASEC_DIR}/results"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
MODEL_WEIGHTS="${PROJECT_DIR}/.models/finetuned_model_Adam_0.01_7_secs.torch"
CONFIG_FILE="${PROJECT_DIR}/configs/config0.yaml"

TMP_DIR="${PROJECT_DIR}/.tmp"
L_TMP="/tmp/assessment_marathon"
MEM_THRESHOLD_MB=5120 

mkdir -p "$TMP_DIR"
mkdir -p "$RESULTS_BASE"
mkdir -p "$L_TMP/embeddings"

# --- 2. PREPARE TASK LIST ---
cd "${EMB_BASE}"
# Cerchiamo tutti i file combined_test.h5 come richiesto
mapfile -t H5_LIST < <(find . -name "combined_test.h5")
cd "$PROJECT_DIR"

# --- 3. CREATE TEMPORARY AGGREGATOR SCRIPT IN .tmp ---
cat << 'EOF' > "${TMP_DIR}/metrics_aggregator.py"
import os, sys, pandas as pd, glob
def aggregate(results_base):
    files = glob.glob(os.path.join(results_base, "**", "assessment_metrics.csv"), recursive=True)
    all_res = []
    for f in files:
        parts = f.replace(results_base + "/", "").split("/")
        # Struttura attesa: fmt/octaves/cut_sec/assessment_metrics.csv
        if len(parts) >= 4:
            df = pd.read_csv(f)
            df['format'] = parts[0]
            df['n_octaves'] = parts[1]
            df['cut_secs'] = parts[2]
            all_res.append(df)
    
    if not all_res: 
        print("⚠️ No metrics files found for aggregation.")
        return
        
    final_df = pd.concat(all_res).sort_values(by='f05', ascending=False)
    final_df.to_csv(os.path.join(results_base, "GLOBAL_ASSESSMENT_REPORT.csv"), index=False)
    print("\n🏆 Top 5 Combinations (Ranked by F0.5):")
    print(final_df.head(5).to_string(index=False))

if __name__ == "__main__":
    aggregate(sys.argv[1])
EOF

# --- 4. EXECUTION LOOP WITH QUEUE LOGIC ---
TOTAL_FILES=${#H5_LIST[@]}
CURRENT_IDX=0

while [ $CURRENT_IDX -lt $TOTAL_FILES ]; do
    BATCH_SIZE_MB=0
    CURRENT_BATCH=()
    
    while [ $CURRENT_IDX -lt $TOTAL_FILES ]; do
        FILE_REL=${H5_LIST[$CURRENT_IDX]}
        FILE_ABS="${EMB_BASE}/${FILE_REL}"
        FILE_SIZE_MB=$(( $(du -m "$FILE_ABS" | cut -f1) ))
        
        if [ $((BATCH_SIZE_MB + FILE_SIZE_MB)) -gt $MEM_THRESHOLD_MB ] && [ ${#CURRENT_BATCH[@]} -gt 0 ]; then
            break
        fi
        
        DIR_REL=$(dirname "$FILE_REL")
        mkdir -p "$L_TMP/embeddings/$DIR_REL"
        cp "$FILE_ABS" "$L_TMP/embeddings/$FILE_REL"
        
        CURRENT_BATCH+=("$FILE_REL")
        BATCH_SIZE_MB=$((BATCH_SIZE_MB + FILE_SIZE_MB))
        CURRENT_IDX=$((CURRENT_IDX + 1))
    done
    
    echo "🚀 Processing batch of ${#CURRENT_BATCH[@]} files..."
    
    singularity exec --nv --no-home \
        --bind "$PROJECT_DIR:/app" \
        --bind "$L_TMP:/tmp_node" \
        --bind "$RESULTS_BASE:$RESULTS_BASE" \
        "$SIF_FILE" \
        python3 /app/scripts/finetuned_model_assessment.py \
            --local_root "/tmp_node/embeddings" \
            --model_path "$MODEL_WEIGHTS" \
            --results_base "$RESULTS_BASE" \
            --config_path "/app/configs/config0.yaml" \
            --batch_list "${CURRENT_BATCH[*]}"

    rm -rf "$L_TMP/embeddings"/*
done

# --- 5. FINAL AGGREGATION ---
singularity exec --no-home --bind "$RESULTS_BASE:$RESULTS_BASE" --bind "$TMP_DIR:/tmp_scripts" "$SIF_FILE" \
    python3 /tmp_scripts/metrics_aggregator.py "$RESULTS_BASE"

# Cleanup
rm -rf "$TMP_DIR"
rm -rf "$L_TMP"
echo "✅ Pipeline terminata."
