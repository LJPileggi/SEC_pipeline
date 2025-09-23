#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4           # <--- Esegui 4 processi per nodo
#SBATCH --cpus-per-task=1             # Un CPU core per processo (controlla i requisiti CLAP)
#SBATCH --time=00:30:00
#SBATCH --exclusive                   # O --exclusive, o --gpus-per-task=1
#SBATCH --gres=gpu:4                   # Richiedi 4 GPU per il nodo
#SBATCH -A IscrC_Pb-skite
#SBATCH -p boost_usr_prod

ROOT_SOURCE_PATH="../dataSEC/testing/RAW_DATASET"
FINAL_TARGET_PATH="../dataSEC/testing/PREPROCESSED_DATASET"

export OMP_NUM_THREADS=1
export OMP_PLACES=cores
export OMP_PROC_BIND=close

module purge

source .venv/bin/activate

# Setta MASTER_ADDR e MASTER_PORT per la comunicazione DDP
# SLURM_JOB_NODELIST ti darà l'hostname del nodo
# Questa è una pratica comune per il master_addr
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
# Scegli una porta alta e non in uso
export MASTER_PORT=29500

embed_test="./tests/test_embeddings.py"

cp -r $ROOT_SOURCE_PATH $TMPDIR/RAW_DATASET
cp -r $FINAL_TARGET_PATH $TMPDIR/PREPROCESSED_DATASET

# Avvia ogni processo con srun, che imposterà RANK e WORLD_SIZE
srun python3 "$embed_test"

cp -r $TMPDIR/PREPROCESSED_DATASET* $FINAL_TARGET_PATH
