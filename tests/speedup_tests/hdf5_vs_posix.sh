#!/bin/bash

# --- 1. CONFIGURATION ---
MY_USER="lpilegg1"
PROJECT_DIR="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline"
L_TMP="${PROJECT_DIR}/.tmp_io_final"
SIF="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline/.containers/clap_pipeline.sif"

# Parametri statistici
ITERATIONS=10
BATCH_SIZE=500
TOTAL_FILES=$((ITERATIONS * BATCH_SIZE))

rm -rf "$L_TMP"
mkdir -p "${L_TMP}/wav_files"

# --- 2. GENERATION (5000 file) ---
echo "🔨 Generazione dataset di test ($TOTAL_FILES tracce)..."
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
cat << 'EOF' > "${L_TMP}/run_ultimate.sh"
#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:40:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null

L_TMP="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final"
SIF="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.containers/clap_pipeline.sif"
SSD_PATH="/tmp/bench_${SLURM_JOB_ID}"
RAW_RESULTS="${L_TMP}/raw_data.csv"

echo "Phase,Iteration,POSIX_Wall,POSIX_Sys,HDF5_Wall,HDF5_Sys" > "$RAW_RESULTS"

# --- Python Probe ---
cat << 'PY_PROBE' > "${L_TMP}/probe.py"
import time, h5py, soundfile as sf, sys, os
def run_test(wav_p, h5_p, start, count):
    # POSIX
    ts = os.times(); ws = time.perf_counter()
    for i in range(start, start + count):
        with sf.SoundFile(os.path.join(wav_p, f"track_{i}.wav")) as f: _ = f.read()
    te = os.times(); we = time.perf_counter()
    p_wall, p_sys = we - ws, te[1] - ts[1]
    # HDF5
    ts = os.times(); ws = time.perf_counter()
    with h5py.File(h5_p, 'r') as h5:
        ds = h5['audio']
        for i in range(start, start + count): _ = ds[i]
    te = os.times(); we = time.perf_counter()
    h_wall, h_sys = we - ws, te[1] - ts[1]
    return p_wall, p_sys, h_wall, h_sys

p_w, p_s, h_w, h_s = run_test(sys.argv[1], sys.argv[2], int(sys.argv[4]), 500)
print(f"{sys.argv[3]},{sys.argv[5]},{p_w:.6f},{p_s:.6f},{h_w:.6f},{h_s:.6f}")
PY_PROBE

# 1. TEST LUSTRE (10 iterazioni con rotazione)
for i in {0..9}; do
    OFFSET=$((i * 500))
    singularity exec --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "$L_TMP/probe.py" "$L_TMP/wav_files" "$L_TMP/dataset.h5" "LUSTRE" "$OFFSET" "$i" >> "$RAW_RESULTS"
done

# 2. TEST STAGING (Misurato 10 volte)
# Nota: Per lo staging leggiamo gruppi di 500 file diversi per ogni iterazione
for i in {0..9}; do
    OFFSET=$((i * 500))
    ITER_SSD="${SSD_PATH}_$i"
    mkdir -p "$ITER_SSD/wavs"
    
    # Misura Staging POSIX
    t_s=$(date +%s.%N)
    cp "$L_TMP/wav_files/track_"{$OFFSET..$((OFFSET+499))}.wav "$ITER_SSD/wavs/"
    t_e=$(date +%s.%N)
    st_posix=$(python3 -c "print($t_e - $t_s)")
    
    # Misura Staging HDF5 (essendo un file solo, lo copiamo e basta, ma 10 volte)
    t_s=$(date +%s.%N)
    cp "$L_TMP/dataset.h5" "$ITER_SSD/data.h5"
    t_e=$(date +%s.%N)
    st_h5=$(python3 -c "print($t_e - $t_s)")
    
    echo "STAGING,$i,$st_posix,0,$st_h5,0" >> "$RAW_RESULTS"

    # 3. TEST LOCAL SSD
    singularity exec --no-home --bind "/tmp:/tmp" --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "$L_TMP/probe.py" "$ITER_SSD/wavs" "$ITER_SSD/data.h5" "SSD" "$OFFSET" "$i" >> "$RAW_RESULTS"
    
    rm -rf "$ITER_SSD"
done

# --- Analisi Statistica Finale ---
singularity exec --no-home --bind "${L_TMP}:${L_TMP}" "$SIF" python3 -u - <<'PY_STATS'
import pandas as pd
import numpy as np
df = pd.read_csv("/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final/raw_data.csv")
print("\n" + "="*80)
print(f"{'ANALISI I/O PERFORMANCE (Media su 10 iterazioni)':^80}")
print("="*80)
for phase in ['LUSTRE', 'STAGING', 'SSD']:
    sub = df[df['Phase'] == phase]
    print(f"\n>>> FASE: {phase}")
    for fmt in ['POSIX', 'HDF5']:
        wall = sub[f'{fmt}_Wall']
        sys_t = sub[f'{fmt}_Sys']
        overhead = (sys_t / wall * 100).mean() if wall.mean() > 0 else 0
        print(f"  {fmt:<6} | Wall Time: {wall.mean():.4f}s ± {wall.std():.4f} | Sys Time: {sys_t.mean():.4f}s | Overhead: {overhead:.1f}%")
print("="*80)
PY_STATS
EOF

# --- 4. EXECUTION ---
echo "📤 Invio Job di Benchmark..."
JOB_ID=$(sbatch --parsable "${L_TMP}/run_ultimate.sh")
while squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do sleep 10; done
