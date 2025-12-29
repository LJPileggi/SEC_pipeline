#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           
#SBATCH --cpus-per-task=8              # 🎯 Aumentato per velocizzare l'inferenza su CPU
#SBATCH --time=04:00:00               
#SBATCH --mem=128G                     # 🎯 RAM abbondante per evitare OOM con segmenti lunghi
#SBATCH --exclusive                   
#SBATCH --gres=gpu:4                   
#SBATCH -A IscrC_Pb-skite
#SBATCH -p boost_usr_prod

# --- 1. VARIABILI GLOBALI E PERCORSI ---

SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
TEMP_DIR="/tmp/$SLURM_JOB_ID"
PERSISTENT_DESTINATION="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/benchmark_logs/$SLURM_JOB_ID"

# Variabili di configurazione
BENCHMARK_CONFIG_FILE="test_config.yaml" 
BENCHMARK_AUDIO_FORMAT="wav"
BENCHMARK_N_OCTAVE="1"

# --- 2. PREPARAZIONE AMBIENTE SU /tmp (Disco Locale Veloce) ---

mkdir -p "$TEMP_DIR/dataSEC/RAW_DATASET"
mkdir -p "$TEMP_DIR/work_dir"

echo "Copia dei pesi CLAP su /tmp..."
CLAP_LOCAL_WEIGHTS="$TEMP_DIR/work_dir/CLAP_weights_2023.pth"
cp "$CLAP_SCRATCH_WEIGHTS" "$CLAP_LOCAL_WEIGHTS"

# --- 3. GENERAZIONE DINAMICA DEL DATASET HDF5 ---

echo "Generazione dataset HDF5 di test..."
cat << EOF > "$TEMP_DIR/work_dir/create_h5_data.py"
import sys, os
sys.path.append('.')
from tests.utils.create_fake_raw_audio_h5 import create_fake_raw_audio_h5 
TARGET_DIR = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'RAW_DATASET') 
if __name__ == '__main__':
    create_fake_raw_audio_h5(TARGET_DIR)
EOF

singularity exec --bind "$TEMP_DIR:/tmp_data" "$SIF_FILE" \
    python3 "/tmp_data/work_dir/create_h5_data.py"

# --- 4. CONFIGURAZIONE AMBIENTE ---

export CLAP_TEXT_ENCODER_PATH="/usr/local/clap_cache/tokenizer_model/" 
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/work_dir/CLAP_weights_2023.pth"
export NODE_TEMP_BASE_DIR="/tmp_data/dataSEC" 
export NO_EMBEDDING_SAVE="True" 

# --- 5. ESECUZIONE PIPELINE DISTRIBUITA (srun) ---

# --- DEBUG RETE PYTORCH ---
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export TORCH_CPP_LOG_LEVEL=INFO
export NCCL_DEBUG=INFO

# 🎯 FORZA L'INTERFACCIA DI RETE (Spesso risolve su Leonardo)
# Se Leonardo usa InfiniBand, a volte forzare lo stack socket aiuta il debugging
export NCCL_IB_DISABLE=1 
export NCCL_P2P_DISABLE=1

# 🎯 FORZA IL MASTER_ADDR DINAMICO (Se non lo fa utils.py)
# export MASTER_ADDR=$(hostname)
export MASTER_PORT=29500

# Forza PyTorch e NCCL a usare solo l'interfaccia di loopback locale
export NCCL_SOCKET_IFNAME=lo
export GLOO_SOCKET_IFNAME=lo
export MASTER_ADDR=127.0.0.1

echo "🚀 Avvio Pipeline CLAP su SLURM..."
srun singularity exec \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd)/configs:/app/configs" \
    "$SIF_FILE" \
    python3 scripts/get_clap_embeddings.py \
        --config_file "$BENCHMARK_CONFIG_FILE" \
        --n_octave "$BENCHMARK_N_OCTAVE" \
        --audio_format "$BENCHMARK_AUDIO_FORMAT"

# --- 6. UNIONE SEQUENZIALE DEI LOG (Post-Processing) ---

echo "🔗 Unione sequenziale dei log..."
cat << EOF > "$TEMP_DIR/work_dir/join_logs_wrapper.py"
import sys, os
sys.path.append('.')
from src.utils import join_logs
from src.dirs_config import basedir_preprocessed
def main():
    base_path = os.path.join(basedir_preprocessed, "$BENCHMARK_AUDIO_FORMAT", "${BENCHMARK_N_OCTAVE}_octave")
    for entry in os.listdir(base_path):
        target_dir = os.path.join(base_path, entry)
        if os.path.isdir(target_dir) and entry.endswith("_secs"):
            print(f"Merging: {entry}")
            join_logs(target_dir)
if __name__ == '__main__': main()
EOF

singularity exec --bind "$TEMP_DIR:/tmp_data" "$SIF_FILE" \
    python3 "/tmp_data/work_dir/join_logs_wrapper.py"

# --- 7. ANALISI FINALE ---

echo "📊 Avvio Analisi Tempi..."
cat << EOF > "$TEMP_DIR/work_dir/analysis_wrapper.py"
import os, sys
sys.path.append('.') 
import tests.utils.analyse_test_execution_times as analysis_module
analysis_module.config_test_folder = os.path.join(os.getenv('NODE_TEMP_BASE_DIR'), 'PREPROCESSED_DATASET')
if __name__ == '__main__':
    results = analysis_module.analyze_execution_times("$BENCHMARK_AUDIO_FORMAT", "$BENCHMARK_N_OCTAVE", "$BENCHMARK_CONFIG_FILE")
    analysis_module.print_analysis_results(results)
EOF

singularity exec --bind "$TEMP_DIR:/tmp_data" "$SIF_FILE" \
    python3 "/tmp_data/work_dir/analysis_wrapper.py"

# --- 8. SALVATAGGIO RISULTATI E PULIZIA ---

mkdir -p "$PERSISTENT_DESTINATION"
cp -r "$TEMP_DIR/dataSEC" "$PERSISTENT_DESTINATION/"
rm -rf "$TEMP_DIR"
echo "✅ Job Completato. Risultati in $PERSISTENT_DESTINATION"
