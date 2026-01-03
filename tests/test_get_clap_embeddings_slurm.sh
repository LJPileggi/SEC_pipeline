#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:4
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_Pb-skite

# --- 1. VARIABILI GLOBALI E PERCORSI ---

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"

TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_job_$SLURM_JOB_ID"
PERSISTENT_DESTINATION="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/benchmark_logs/$SLURM_JOB_ID"

BENCHMARK_CONFIG_FILE="test_config.yaml" 
BENCHMARK_AUDIO_FORMAT="wav"
BENCHMARK_N_OCTAVE="1"

# --- 2. PREPARAZIONE AMBIENTE E "TRUFFA" CACHE CLAP ---

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET"
mkdir -p "$TEMP_DIR/work_dir"

# 🎯 CREAZIONE STRUTTURA CACHE HUGGINGFACE (Cruciale per msclap)
# msclap cerca i file in questa specifica gerarchia di sottocartelle
CACHE_MODELS_DIR="$TEMP_DIR/work_dir/hub/models--microsoft--msclap/snapshots/main"
mkdir -p "$CACHE_MODELS_DIR"

echo "Preparazione cache CLAP locale in $CACHE_MODELS_DIR..."
# Copiamo il file rinominandolo come si aspetta la libreria
cp "$CLAP_SCRATCH_WEIGHTS" "$CACHE_MODELS_DIR/CLAP_weights_2023.pth"

# --- 3. GENERAZIONE DINAMICA DEL DATASET HDF5 ---

echo "Generazione dataset HDF5 di test..."
cat << EOF > "$TEMP_DIR/work_dir/create_h5_data.py"
import sys, os
# 🎯 FORZA PATH ASSOLUTO INTERNO
sys.path.append('/app')
from tests.utils.create_fake_raw_audio_h5 import create_fake_raw_audio_h5 
TARGET_DIR = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'RAW_DATASET') 
if __name__ == '__main__':
    create_fake_raw_audio_h5(TARGET_DIR)
EOF

# 🎯 RIMOSSO -C E AGGIUNTO BIND PROGETTO PER EVITARE MODULENOTFOUND
singularity exec --bind "$TEMP_DIR:/tmp_data" --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
    python3 "/tmp_data/work_dir/create_h5_data.py"

# --- 4. CONFIGURAZIONE AMBIENTE ---

# 🎯 Diciamo a HuggingFace di usare la nostra "finta" cartella home/cache
export HF_HOME="$TEMP_DIR/work_dir"
export HF_HUB_OFFLINE=1 

export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/hub/models--microsoft--msclap/snapshots/main/CLAP_weights_2023.pth"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC" 
export NO_EMBEDDING_SAVE="True" 

export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

# --- 5. ESECUZIONE PIPELINE ---
echo "🚀 Avvio Multi-Processo..."

srun --unbuffered -l -n 4 --export=ALL --cpu-bind=none \
    singularity exec \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" \
    --pwd "/app" \
    "$SIF_FILE" \
    python3 scripts/get_clap_embeddings.py \
        --config_file "$BENCHMARK_CONFIG_FILE" \
        --n_octave "$BENCHMARK_N_OCTAVE" \
        --audio_format "$BENCHMARK_AUDIO_FORMAT"

# --- 6. UNIONE SEQUENZIALE DEI LOG ---

echo "🔗 Unione sequenziale dei log..."
cat << EOF > "$TEMP_DIR/work_dir/join_logs_wrapper.py"
import sys, os
sys.path.append('/app')
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed
def main():
    base_path = os.path.join(basedir_preprocessed, "$BENCHMARK_AUDIO_FORMAT", "${BENCHMARK_N_OCTAVE}_octave")
    if not os.path.exists(base_path): return
    for entry in os.listdir(base_path):
        target_dir = os.path.join(base_path, entry)
        if os.path.isdir(target_dir) and entry.endswith("_secs"):
            print(f"Merging: {entry}")
            join_logs(target_dir)
if __name__ == '__main__': main()
EOF

singularity exec --bind "$TEMP_DIR:/tmp_data" --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
    python3 "/tmp_data/work_dir/join_logs_wrapper.py"

# --- 7. ANALISI FINALE ---

echo "📊 Avvio Analisi Tempi..."
cat << EOF > "$TEMP_DIR/work_dir/analysis_wrapper.py"
import os, sys
sys.path.append('/app') 
import tests.utils.analyse_test_execution_times as analysis_module
analysis_module.config_test_folder = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'PREPROCESSED_DATASET')
if __name__ == '__main__':
    results = analysis_module.analyze_execution_times("$BENCHMARK_AUDIO_FORMAT", "$BENCHMARK_N_OCTAVE", "$BENCHMARK_CONFIG_FILE")
    analysis_module.print_analysis_results(results)
EOF

singularity exec --bind "$TEMP_DIR:/tmp_data" --bind "$(pwd):/app" --pwd "/app" "$SIF_FILE" \
    python3 "/tmp_data/work_dir/analysis_wrapper.py"

# --- 8. SALVATAGGIO RISULTATI E PULIZIA ---

mkdir -p "$PERSISTENT_DESTINATION"
cp -r "$TEMP_DIR/dataSEC" "$PERSISTENT_DESTINATION/"
# rm -rf "$TEMP_DIR"
echo "✅ Job Completato. Risultati in $PERSISTENT_DESTINATION"
