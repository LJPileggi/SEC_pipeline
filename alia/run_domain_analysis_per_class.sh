#!/bin/bash

# --- 1. CONFIGURAZIONE DEI PATH BASE ---
MY_USER=$(whoami)
BASEDIR="/leonardo_scratch/large/userexternal/${MY_USER}"
PROJECT_DIR="${BASEDIR}/SEC_pipeline"
DATASEC_DIR="${BASEDIR}/dataSEC"
SIF_FILE="${PROJECT_DIR}/.containers/clap_pipeline.sif"
CONFIG_FILE="${PROJECT_DIR}/configs/config0.yaml"

# Esportazione esplicita per consentire la corretta interpretazione dei percorsi spaccettati nel container
export BASEDIR="$BASEDIR"

# 🎯 MACRO VARIABILI PER LE MATRICI DI CONFUSIONE REALI (Passate come richiesto)
CONF_MATRIX_0_OCTAVE="${DATASEC_DIR}/results_0_octave/wav/0_octave/7_secs/misclassified_keys.csv"
CONF_MATRIX_3_OCTAVE="${DATASEC_DIR}/results_3_octave/wav/3_octave/7_secs/misclassified_keys.csv"

# PARAMETRI COMPILATI SUI PESI DI FABBRICA
CLAP_SCRATCH_WEIGHTS="${PROJECT_DIR}/.clap_weights/CLAP_weights_2023.pth"
ROBERTA_PATH="${PROJECT_DIR}/.clap_weights/roberta-base"

# --- 2. VINCOLI ATOMICI PER IL NODO DI LOGIN (Risorse Safe) ---
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES="" 

export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
export INJECT_OCTAVE="False" 

export NUMBA_CACHE_DIR="${PROJECT_DIR}/.numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"

export LOCAL_CLAP_WEIGHTS_PATH="$CLAP_SCRATCH_WEIGHTS"
export CLAP_TEXT_ENCODER_PATH="$ROBERTA_PATH"

echo "⏳ Esecuzione Analisi di Dominio dataSEC Online (Modalità Interattiva Offline a Tranche)..."

# --- 3. ESTRAZIONE DINAMICA DELLE CLASSI DAL CONFIG YAML ---
CLASSES_LIST=$(singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 -c "import sys; sys.path.insert(0, '/app'); from src.utils import get_config_from_yaml; print(' '.join(get_config_from_yaml('$CONFIG_FILE')[0]))")

if [ -z "$CLASSES_LIST" ]; then
    echo "❌ Errore: Impossibile estrarre le classi dal file config o ambiente non configurato."
    exit 1
fi

# --- 4. RUN PIPELINE ONLINE SEQUENZIALE (Classe per Classe) ---
for CLASSE in $CLASSES_LIST; do
    echo "============================================================"
    echo "⏳ AVVIO TRANCHE dataSEC PER LA CLASSE ISOLATA: $CLASSE"
    echo "============================================================"
    
    singularity exec --no-home \
        --bind "/leonardo_scratch:/leonardo_scratch" \
        --bind "$(pwd):/app" \
        --pwd "/app" \
        "$SIF_FILE" \
        python3 alia/evaluate_domain_distance_per_class.py \
            --config_file "$CONFIG_FILE" \
            --n_octave "3" \
            --audio_format "wav" \
            --cut_secs 7 \
            --samples_per_class 50 \
            --class_to_process "$CLASSE"

    echo "✅ Tranche classe $CLASSE conclusa. Risorse RAM interamente riacquisite."
done

# --- 5. AGGREGAZIONE FINALE DEI RISULTATI NUMERICI ---
echo "============================================================"
echo "📊 AVVIO SCRIPT DI CONSOLIDAMENTO E ANALISI GLOBALE dataSEC"
echo "============================================================"

singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 alia/aggregate_domain_results.py \
        --results_dir "results/domain_analysis_online" \
        --output_dir "results/domain_analysis_final"

# --- 6. GENERAZIONE DEI BOXPLOT SPETTRALI PER CANALE MEL ---
echo "============================================================"
echo "📈 AVVIO GENERATORE AUTOMATICO DI BOXPLOT PER CANALE MEL"
echo "============================================================"

singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 alia/plot_mel_boxplots.py \
        --results_dir "results/domain_analysis_online" \
        --output_dir "results/domain_analysis_plots"

# --- 7. NEW: CALCOLO DELLE DISTANZE INTERCLASSE E CORRELAZIONE MATRICE DI CONFUSIONE ---
echo "============================================================"
echo "🌐 AVVIO GEOMETRIA INTERCLASSE INTERATTIVA ED INCROCIO CONFUSIONE"
echo "============================================================"

singularity exec --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 alia/compute_interclass_distances.py \
        --results_dir "results/domain_analysis_online" \
        --output_dir "results/domain_analysis_interclass" \
        --conf_0 "$CONF_MATRIX_0_OCTAVE" \
        --conf_3 "$CONF_MATRIX_3_OCTAVE"

echo "🎉 Pipeline dataSEC completata con successo in totale sicurezza hardware."
