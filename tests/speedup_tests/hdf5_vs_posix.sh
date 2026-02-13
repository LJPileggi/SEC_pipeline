#!/bin/bash

# --- 1. CONFIGURATION ---
MY_USER="lpilegg1"
PROJECT_DIR="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline"
L_TMP="${PROJECT_DIR}/.tmp_io_final"
RAW_LOG="${L_TMP}/raw_bench.log"
SIF="/leonardo_scratch/large/userexternal/${MY_USER}/SEC_pipeline/.containers/clap_pipeline.sif"

ITERATIONS=10
BATCH_SIZE=500

rm -rf "$L_TMP"
mkdir -p "${L_TMP}/wav_files"

# --- 2. GENERATION (5000 file) ---
echo "🔨 Generazione dataset di test (5000 tracce)..."
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

# --- 3. GENERATE SLURM BATCH SCRIPT ---
cat << 'EOF' > "${L_TMP}/run_ultimate.sh"
#!/bin/bash
#SBATCH --job-name=io_bench
#SBATCH --partition=boost_usr_prod
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --gres=gpu:1
#SBATCH -A IscrC_Pb-skite
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

L_TMP="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final"
RAW_LOG="${L_TMP}/raw_bench.log"
SIF="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.containers/clap_pipeline.sif"
SSD_NODE="/tmp/bench_data_${SLURM_JOB_ID}"

# --- Python Probe (Output CSV-like per il parsing) ---
cat << 'PY_INNER' > "${L_TMP}/probe.py"
import time, h5py, soundfile as sf, sys, os
def run(wav_p, h5_p, label, start, count):
    # POSIX
    ts = os.times(); ws = time.perf_counter()
    for i in range(start, start + count):
        with sf.SoundFile(os.path.join(wav_p, f"track_{i}.wav")) as f: _ = f.read()
    te = os.times(); we = time.perf_counter()
    p_w, p_s = we - ws, te[1] - ts[1]
    # HDF5
    ts = os.times(); ws = time.perf_counter()
    with h5py.File(h5_p, 'r') as h5:
        ds = h5['audio']
        for i in range(start, start + count): _ = ds[i]
    te = os.times(); we = time.perf_counter()
    h_w, h_s = we - ws, te[1] - ts[1]
    print(f"{label},{p_w:.6f},{p_s:.6f},{h_w:.6f},{h_s:.6f}")

run(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), 500)
PY_INNER

echo "Phase,Wall_P,Sys_P,Wall_H,Sys_H" > "$RAW_LOG"

# PHASE 1: LUSTRE
for i in {0..9}; do
    OFFSET=$((i * 500))
    singularity exec --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "$L_TMP/probe.py" "$L_TMP/wav_files" "$L_TMP/dataset.h5" "LUSTRE" "$OFFSET" >> "$RAW_LOG" 2>&1
done

# PHASE 2: STAGING (10 iterazioni di staging-in)
mkdir -p "${SSD_NODE}/wavs"
for i in {0..9}; do
    OFFSET=$((i * 500))
    t1=$(date +%s.%N)
    cp "$L_TMP/wav_files/track_"{$OFFSET..$((OFFSET+499))}.wav "${SSD_NODE}/wavs/"
    cp "$L_TMP/dataset.h5" "${SSD_NODE}/dataset.h5"
    t2=$(date +%s.%N)
    echo "STAGING,$(python3 -c "print($t2 - $t1)"),0,0,0" >> "$RAW_LOG"
    
    # PHASE 3: SSD (Immediatamente dopo ogni staging per mediare)
    singularity exec --no-home --bind "/tmp:/tmp" --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "$L_TMP/probe.py" "${SSD_NODE}/wavs" "${SSD_NODE}/dataset.h5" "SSD" "$OFFSET" >> "$RAW_LOG" 2>&1
    
    rm -f ${SSD_NODE}/wavs/* ${SSD_NODE}/dataset.h5
done

# --- ANALISI FINALE (Python) ---
singularity exec --no-home --bind "$L_TMP:$L_TMP" "$SIF" python3 -u - <<'PY_STATS'
import pandas as pd
import numpy as np
df = pd.read_csv("/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final/raw_bench.log")

def get_stats(data):
    return f"{data.mean():.4f}s (±{data.std():.4f})"

print("\n" + "="*90)
print(f"{'REPORT STATISTICO I/O (N=10 iterazioni, 500 file/iter)':^90}")
print("="*90)
print(f"{'Fase':<12} | {'POSIX (Wall)':<22} | {'HDF5 (Wall)':<22} | {'Overhead Sys':<12}")
print("-" * 90)

for phase in ['LUSTRE', 'SSD']:
    sub = df[df['Phase'] == phase]
    p_wall = sub['Wall_P']
    h_wall = sub['Wall_H']
    # Overhead calcolato come SysTime / WallTime
    p_ov = (sub['Sys_P'].mean() / p_wall.mean() * 100)
    h_ov = (sub['Sys_H'].mean() / h_wall.mean() * 100)
    
    print(f"{phase:<12} | {get_stats(p_wall):<22} | {get_stats(h_wall):<22} | P:{p_ov:>4.1f}% H:{h_ov:>4.1f}%")

st_data = df[df['Phase'] == 'STAGING']['Wall_P']
print("-" * 90)
print(f"STAGING IN   | Media: {st_data.mean():.4f}s | Dev.Std: {st_data.std():.4f} | (Lustre -> SSD)")
print("="*90 + "\n")
PY_STATS
EOF

# --- 4. EXECUTION ---
echo "📤 Invio Job di Benchmark (10 iterazioni)..."
JOB_ID=$(sbatch --parsable "${L_TMP}/run_ultimate.sh")

# Invece di tail -f, aspettiamo e basta
while squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
    echo -n "."
    sleep 10
done

echo -e "\n✅ Job completato. Risultati analizzati:\n"
# Stampiamo il report finale che il nodo ha scritto nel log (catturandolo dal file generato)
# Cerchiamo l'inizio della tabella nel log raw o facciamo girare l'analisi qui
singularity exec --no-home --bind "$L_TMP:$L_TMP" "$SIF" python3 -u - <<'PY_FINAL'
import pandas as pd
df = pd.read_csv("/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final/raw_bench.log")
for phase in ['LUSTRE', 'SSD']:
    sub = df[df['Phase'] == phase]
    print(f"{phase}: POSIX {sub['Wall_P'].mean():.4f}±{sub['Wall_P'].std():.4f} | HDF5 {sub['Wall_H'].mean():.4f}±{sub['Wall_H'].std():.4f}")
print(f"STAGING: {df[df['Phase'] == 'STAGING']['Wall_P'].mean():.4f}s")
PY_FINAL
