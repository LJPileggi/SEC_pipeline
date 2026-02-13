#!/bin/bash

# --- 1. CONFIGURATION ---
PROJECT_DIR="/leonardo_scratch/large/userexternal/${USER}/SEC_pipeline"
L_TMP="${PROJECT_DIR}/.tmp_io_final"
STREAM_LOG="${L_TMP}/benchmark_stats.log"
SIF="/leonardo_scratch/large/userexternal/${USER}/SEC_pipeline/.containers/clap_pipeline.sif"

# Parametri per la media
ITERATIONS=10
BATCH_SIZE=500
TOTAL_FILES=$((ITERATIONS * BATCH_SIZE)) # 5000 file

mkdir -p "${L_TMP}/wav_files"
touch "$STREAM_LOG"

# --- 2. GENERATION (5000 file) ---
echo "🔨 Generazione di $TOTAL_FILES file per test statistico..."
singularity exec --no-home --bind "${L_TMP}:/mnt_lustre" "$SIF" \
    python3 -u - <<'PY'
import numpy as np
import soundfile as sf
import h5py
import os
n, sr, d = 5000, 44100, 1
wav_dir, h5_p = "/mnt_lustre/wav_files", "/mnt_lustre/dataset.h5"
data = np.random.uniform(-1, 1, sr * d).astype(np.float32)
with h5py.File(h5_p, 'w') as h5:
    ds = h5.create_dataset('audio', (n, sr*d), dtype='f4')
    for i in range(n):
        sf.write(f"{wav_dir}/track_{i}.wav", data, sr)
        ds[i] = data
PY

# --- 3. SLURM BATCH SCRIPT ---
cat << 'EOF' > "${L_TMP}/run_stats.sh"
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite

L_TMP="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final"
LOG="${L_TMP}/benchmark_stats.log"
SIF="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.containers/clap_pipeline.sif"
SSD="/tmp/bench_${SLURM_JOB_ID}"
mkdir -p "$SSD/wavs"

# --- Python Probe con distinzione System/User ---
cat << 'PY_STATS' > "/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final/reader_stats.py"
import time, h5py, soundfile as sf, sys, os

def benchmark(wav_p, h5_p, label, start_idx, count):
    # os.times() restituisce (user, system, children_user, children_system, elapsed)
    t_start = os.times()
    wall_start = time.perf_counter()
    
    # Test POSIX
    for i in range(start_idx, start_idx + count):
        with sf.SoundFile(os.path.join(wav_p, f"track_{i}.wav")) as f: _ = f.read()
    
    t_end = os.times()
    wall_end = time.perf_counter()
    
    posix_wall = wall_end - wall_start
    posix_sys = t_end[1] - t_start[1]
    posix_user = t_end[0] - t_start[0]

    # Test HDF5
    t_start = os.times()
    wall_start = time.perf_counter()
    with h5py.File(h5_p, 'r') as h5:
        ds = h5['audio']
        for i in range(start_idx, start_idx + count): _ = ds[i]
    
    t_end = os.times()
    wall_end = time.perf_counter()
    
    h5_wall = wall_end - wall_start
    h5_sys = t_end[1] - t_start[1]
    h5_user = t_end[0] - t_start[0]

    print(f"{label}|{posix_wall:.4f}|{posix_sys:.4f}|{h5_wall:.4f}|{h5_sys:.4f}")

benchmark(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), int(sys.argv[5]))
PY_STATS

echo "Label|POSIX_Wall|POSIX_Sys|HDF5_Wall|HDF5_Sys" >> "$LOG"

# Esecuzione Media (10 iterazioni)
for i in {0..9}; do
    OFFSET=$((i * 500))
    echo "🏃 Iterazione $i (Offset $OFFSET)..."
    
    # Test su Lustre
    singularity exec --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "$L_TMP/reader_stats.py" "$L_TMP/wav_files" "$L_TMP/dataset.h5" "LUSTRE_IT$i" "$OFFSET" 500 >> "$LOG" 2>&1
done

# Copia per test SSD (Fase 2)
cp "$L_TMP/wav_files/"*.wav "$SSD/wavs/"
cp "$L_TMP/dataset.h5" "$SSD/"

for i in {0..9}; do
    OFFSET=$((i * 500))
    # Test su SSD
    singularity exec --no-home --bind "/tmp:/tmp" --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "$L_TMP/reader_stats.py" "$SSD/wavs" "$SSD/dataset.h5" "SSD_IT$i" "$OFFSET" 500 >> "$LOG" 2>&1
done
EOF

# --- 4. INVIO E MONITORAGGIO ---
JOB_ID=$(sbatch --parsable "${L_TMP}/run_stats.sh")
echo "📊 Benchmark in corso (Job $JOB_ID)..."
# Aspetta la fine del job
while squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do sleep 10; done

echo "📈 ANALISI FINALE DEI RISULTATI:"
column -t -s '|' "$STREAM_LOG"
