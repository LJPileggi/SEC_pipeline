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
# MODEL_WEIGHTS="${PROJECT_DIR}/.models/finetuned_model_Adam_0.01_7_secs.torch"
MODEL_WEIGHTS="${PROJECT_DIR}/.models/finetuned_model_RECOVERY_7_secs.torch"
CONFIG_FILE="${PROJECT_DIR}/configs/config0.yaml"

TMP_DIR="${PROJECT_DIR}/.tmp"
L_TMP="/tmp/assessment_marathon"
# 🎯 FIX: Riduciamo leggermente la soglia per evitare saturazione fisica reale del /tmp del nodo
MEM_THRESHOLD_MB=4000 

# --- 2. ROBUSTNESS & CLEANUP ---
cleanup() {
    echo "🧹 [CLEANUP] Job interrupted or finished. Cleaning directories..."
    [ -d "$L_TMP" ] && rm -rf "$L_TMP"
    [ -d "$TMP_DIR" ] && rm -rf "$TMP_DIR"
}

trap cleanup EXIT SIGTERM SIGINT ERR

mkdir -p "$TMP_DIR"
mkdir -p "$RESULTS_BASE"
mkdir -p "$L_TMP/embeddings"

# --- 3. PREPARE TASK LIST (GROUND TRUTH: Ricerca sull'albero) ---
cd "${EMB_BASE}"
# Troviamo i file relativi partendo dalla root del dataset
mapfile -t H5_TRAIN_LIST < <(find . -path "*/7_secs/combined_train.h5" | sed 's|^\./||')
mapfile -t H5_VALID_LIST < <(find . -path "*/7_secs/combined_valid.h5" | sed 's|^\./||')
mapfile -t H5_ES_LIST < <(find . -path "*/7_secs/combined_es.h5" | sed 's|^\./||')
cd "$PROJECT_DIR"

# --- 4. CREATE TEMPORARY AGGREGATOR SCRIPT IN .tmp ---
cat << 'EOF' > "${TMP_DIR}/metrics_aggregator.py"
import os, sys, pandas as pd, glob
def aggregate(results_base):
    files = glob.glob(os.path.join(results_base, "**", "assessment_metrics.csv"), recursive=True)
    all_res = []
    
    for f in files:
        rel_path = os.path.relpath(f, results_base)
        parts = rel_path.split(os.sep)
        
        if len(parts) >= 4:
            df = pd.read_csv(f)
            cols_to_keep = ['accuracy', 'precision', 'recall', 'f05']
            df = df[[c for c in df.columns if c in cols_to_keep]]
            
            df['format'] = parts[0]
            df['n_octaves'] = parts[1]
            df['cut_secs'] = parts[2]
            all_res.append(df)
    
    if not all_res: 
        print("⚠️ No global metrics files found.")
        return
        
    final_df = pd.concat(all_res).sort_values(by='f05', ascending=False)
    final_df.to_csv(os.path.join(results_base, "GLOBAL_ASSESSMENT_REPORT.csv"), index=False)
    print("\n🏆 GLOBAL REPORT (Ranked by F0.5):")
    print(final_df.to_string(index=False))

if __name__ == "__main__":
    aggregate(sys.argv[1])
EOF
# --- 5. EXECUTION LOOP WITH QUEUE LOGIC ---
TOTAL_FILES=${#H5_TRAIN_LIST[@]}
CURRENT_IDX=0

while [ $CURRENT_IDX -lt $TOTAL_FILES ]; do
    BATCH_SIZE_MB=0
    CURRENT_BATCH=()
    
    # 🎯 FIX: Ad ogni nuovo batch, svuotiamo FISICAMENTE la cartella locale per evitare l'accumulo "No space left"
    rm -rf "$L_TMP/embeddings"/*
    
    while [ $CURRENT_IDX -lt $TOTAL_FILES ]; do
        FILE_REL_TRAIN=${H5_TRAIN_LIST[$CURRENT_IDX]}
        FILE_ABS_TRAIN="${EMB_BASE}/${FILE_REL_TRAIN}"
        FILE_REL_VALID=${H5_VALID_LIST[$CURRENT_IDX]}
        FILE_ABS_VALID="${EMB_BASE}/${FILE_REL_VALID}"
        FILE_REL_ES=${H5_ES_LIST[$CURRENT_IDX]}
        FILE_ABS_ES="${EMB_BASE}/${FILE_REL_ES}"
        
        # Calcolo dimensione tripletta
        S_TR=$(du -m "$FILE_ABS_TRAIN" | cut -f1)
        S_VA=$(du -m "$FILE_ABS_VALID" | cut -f1)
        S_ES=$(du -m "$FILE_ABS_ES" | cut -f1)
        FILE_SIZE_MB=$(( S_TR + S_VA + S_ES ))
        
        if [ $((BATCH_SIZE_MB + FILE_SIZE_MB)) -gt $MEM_THRESHOLD_MB ] && [ ${#CURRENT_BATCH[@]} -gt 0 ]; then
            break
        fi
        
        DIR_REL=$(dirname "$FILE_REL_VALID")
        mkdir -p "$L_TMP/embeddings/$DIR_REL"
        
        # Copia delle triplette (INDISPENSABILE per load_single_cut_secs_dataloaders)
        cp "$FILE_ABS_TRAIN" "$L_TMP/embeddings/$FILE_REL_TRAIN"
        cp "$FILE_ABS_VALID" "$L_TMP/embeddings/$FILE_REL_VALID"
        cp "$FILE_ABS_ES" "$L_TMP/embeddings/$FILE_REL_ES"
        
        # Controllo immediato successo copia
        if [ $? -ne 0 ]; then
            echo "❌ ERROR: Copy failed for $DIR_REL. Space on /tmp is exhausted."
            break
        fi
        
        CURRENT_BATCH+=("$FILE_REL_VALID")
        BATCH_SIZE_MB=$((BATCH_SIZE_MB + FILE_SIZE_MB))
        CURRENT_IDX=$((CURRENT_IDX + 1))
    done
    
    echo "🚀 Processing batch of ${#CURRENT_BATCH[@]} configurations..."
    
    for FILE_REL_VALID in "${CURRENT_BATCH[@]}"; do
        echo "📊 Assessing: $FILE_REL_VALID"
        
        # Estraiamo formato e ottave dal path per passarli correttamente
        # Path atteso: ./format/octave_octave/7_secs/combined_valid.h5
        FMT=$(echo "$FILE_REL_VALID" | cut -d'/' -f1)
        OCT=$(echo "$FILE_REL_VALID" | cut -d'/' -f2 | cut -d'_' -f1)

        singularity exec --nv --no-home \
            --bind "$PROJECT_DIR:/app" \
            --bind "$L_TMP:/tmp_node" \
            --bind "$RESULTS_BASE:$RESULTS_BASE" \
            "$SIF_FILE" \
            # python3 /app/scripts/finetuned_octave_model_selection.py \
            python3 /app/scripts/finetuned_model_assessment.py \
                --local_root "/tmp_node/embeddings" \
                --model_path "$MODEL_WEIGHTS" \
                --results_base "$RESULTS_BASE" \
                --config_path "/app/configs/config0.yaml" \
                --batch_list "$FILE_REL_VALID"
    done
done

# --- 6. FINAL AGGREGATION ---
echo "📈 Aggregating results..."
singularity exec --no-home --bind "$RESULTS_BASE:$RESULTS_BASE" --bind "$TMP_DIR:/tmp_scripts" "$SIF_FILE" \
    python3 /tmp_scripts/metrics_aggregator.py "$RESULTS_BASE"

# Cleanup finale
rm -rf "$TMP_DIR"
rm -rf "$L_TMP"
echo "✅ Pipeline terminata."
