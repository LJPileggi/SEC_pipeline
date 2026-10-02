#!/bin/bash
#SBATCH --job-name=Finetune_and_Assess
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0

# ==============================================================================
# 1. PARAMETRI E CONFIGURAZIONE
# ==============================================================================
MY_USER=$(whoami)
BASEDIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASEDIR}/SEC_pipeline"
DATASEC_DIR="${BASEDIR}/dataSEC"
EMB_BASE="${DATASEC_DIR}/PREPROCESSED_DATASET"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
CONFIG_FILE="${PROJECT_DIR}/configs/config0.yaml"

# Parametri da riga di comando o default
AUDIO_FORMAT=${1:-"wav"}
N_OCTAVE=${2:-"3"}
INJECT_OCTAVE_CMD=${3:-"True"}
CUT_SECS=${4:-"7"}

# Forzatura di coerenza
if [ "$N_OCTAVE" -eq 0 ]; then
    INJECT_OCTAVE_CMD="False"
fi

target_folder="${N_OCTAVE}_octave"
local_suffix=""
if [ "$N_OCTAVE" -ne 0 ] && [ "$INJECT_OCTAVE_CMD" = "False" ]; then
    target_folder="${N_OCTAVE}_octave_no_inject"
    local_suffix="_no_inject"
fi

MODEL_DIR="${PROJECT_DIR}/.models"
PRETRAINED_MODEL="${MODEL_DIR}/finetuned_model_Adam_0.01_7_secs.torch"
FINAL_MODEL_PATH="${MODEL_DIR}/finetuned_model_RECOVERY_${CUT_SECS}_secs_${N_OCTAVE}_octave${local_suffix}.torch"
RESULTS_BASE="${DATASEC_DIR}/results_${target_folder}"

L_TMP="/tmp/assess_single_${SLURM_JOB_ID:-$$}"

cleanup() {
    echo "🧹 [CLEANUP] Pulizia file temporanei su /tmp..."
    [ -d "$L_TMP" ] && rm -rf "$L_TMP"
}
trap cleanup EXIT SIGTERM SIGINT ERR

mkdir -p "$MODEL_DIR"
mkdir -p "$RESULTS_BASE"
mkdir -p "$L_TMP/embeddings"

export INJECT_OCTAVE="$INJECT_OCTAVE_CMD"
export FINAL_MODEL_PATH="$FINAL_MODEL_PATH"

# ==============================================================================
# 2. RUN FINETUNING CLASSIFIER
# ==============================================================================
echo "========================================================================"
echo "🚀 FASE 1: Avvio Fine-Tuning Lineare"
echo "   Target Model: ${FINAL_MODEL_PATH}"
echo "   Format: ${AUDIO_FORMAT} | Octave: ${N_OCTAVE} | Inject: ${INJECT_OCTAVE_CMD} | Cut: ${CUT_SECS}s"
echo "========================================================================"

singularity exec --nv --no-home \
    --bind "${BASEDIR}:/app" \
    --bind "${PROJECT_DIR}:/app/${PROJECT_DIR}" \
    --bind "${DATASEC_DIR}:/app/${DATASEC_DIR}" \
    --pwd "/app/${PROJECT_DIR}" \
    "$SIF_FILE" \
    python3 scripts/train_finetuned_classifier.py \
        --config_file "$CONFIG_FILE" \
        --audio_format "$AUDIO_FORMAT" \
        --n_octave "$N_OCTAVE" \
        --cut_secs "$CUT_SECS" \
        --pretrained_path "$PRETRAINED_MODEL"

if [ ! -f "$FINAL_MODEL_PATH" ]; then
    echo "❌ [ERROR] File modello non trovato dopo il training: $FINAL_MODEL_PATH"
    exit 1
fi
echo "✅ Fine-Tuning completato con successo!"

# ==============================================================================
# 3. PREPARAZIONE DATASET LOCALE PER ASSESSMENT
# ==============================================================================
echo ""
echo "========================================================================"
echo "📊 FASE 2: Preparazione e Assessment per la Singola Configurazione"
echo "========================================================================"

# Percorso relativo atteso: format/octave_folder/cut_secs/combined_*.h5
REL_DIR="${AUDIO_FORMAT}/${target_folder}/${CUT_SECS}_secs"
SRC_DIR="${EMB_BASE}/${REL_DIR}"

if [ ! -d "$SRC_DIR" ]; then
    echo "❌ [ERROR] Directory embeddings non trovata: $SRC_DIR"
    exit 1
fi

DEST_DIR="$L_TMP/embeddings/${REL_DIR}"
mkdir -p "$DEST_DIR"

echo "📦 Copia file H5 su disco locale (/tmp)..."
cp "${SRC_DIR}/combined_train.h5" "${DEST_DIR}/"
cp "${SRC_DIR}/combined_valid.h5" "${DEST_DIR}/"
cp "${SRC_DIR}/combined_es.h5"    "${DEST_DIR}/"

REL_VALID_FILE="${REL_DIR}/combined_valid.h5"

# ==============================================================================
# 4. RUN MODEL ASSESSMENT
# ==============================================================================
echo "📈 Calcolo metriche di classificazione e matrici di confusione..."

singularity exec --nv --no-home \
    --bind "$PROJECT_DIR:/app" \
    --bind "$L_TMP:/tmp_node" \
    --bind "$RESULTS_BASE:$RESULTS_BASE" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 /app/scripts/finetuned_model_assessment.py \
        --local_root "/tmp_node/embeddings" \
        --model_path "$FINAL_MODEL_PATH" \
        --results_base "$RESULTS_BASE" \
        --config_path "/app/configs/config0.yaml" \
        --batch_list "$REL_VALID_FILE"

echo ""
echo "========================================================================"
echo "✅ Pipeline completata con successo!"
echo "📁 Risultati e matrici salvati in: $RESULTS_BASE"
echo "========================================================================"
