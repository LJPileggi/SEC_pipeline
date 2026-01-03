#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --time=00:10:00
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_Pb-skite

# Configura qui i tuoi percorsi
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
# Usiamo lo scratch per il file di sync
SYNC_FILE="/leonardo_scratch/large/userexternal/$USER/torch_sync_test"

sleep 10

echo "--- 1. TEST SLURM (FUORI DAL CONTAINER) ---"
srun -l -n 4 echo "SLURM OK: Sono il processo $SLURM_PROCID di $SLURM_NTASKS"

echo -e "\n--- 2. TEST SINGULARITY + VARIABILI ---"
srun -l -n 4 singularity exec "$SIF_FILE" bash -c "echo SINGULARITY OK: PROCID=\$SLURM_PROCID"

echo -e "\n--- 3. TEST PYTORCH SYNC (GENERAZIONE SCRIPT) ---"
cat << EOF > test_sync_minimal.py
import os
import torch.distributed as dist
import datetime
import torch

def main():
    # Recupero variabili
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 4))
    
    print(f"[RANK {rank}] Avvio test su device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    sync_method = "file://$SYNC_FILE"
    
    try:
        # Pulizia file iniziale (solo Rank 0)
        if rank == 0 and os.path.exists("$SYNC_FILE"):
            os.remove("$SYNC_FILE")
            print(f"[RANK {rank}] File di sync rimosso.")

        dist.init_process_group(
            backend="gloo", 
            init_method=sync_method,
            rank=rank,
            world_size=world_size,
            timeout=datetime.timedelta(seconds=30)
        )
        print(f"[RANK {rank}] ✅ SINCRONIZZAZIONE RIUSCITA!")
        
        dist.barrier()
        print(f"[RANK {rank}] ✅ BARRIERA SUPERATA!")
        
    except Exception as e:
        print(f"[RANK {rank}] ❌ FALLIMENTO: {str(e)}")

if __name__ == "__main__":
    main()
EOF

echo -e "\n--- 4. ESECUZIONE TEST PYTORCH SYNC ---"
srun -l -n 4 --export=ALL singularity exec --bind /leonardo_scratch:/leonardo_scratch "$SIF_FILE" python3 test_sync_minimal.py

# Pulizia finale
# rm test_sync_minimal.py "$SYNC_FILE"
