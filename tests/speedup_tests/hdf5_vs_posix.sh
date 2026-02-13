#!/bin/bash

# --- 1. CONFIGURATION (PERCORSI ASSOLUTI) ---
USER_NAME=$(whoami)
PROJECT_DIR="/leonardo_scratch/large/userexternal/${USER_NAME}/SEC_pipeline"
L_TMP="${PROJECT_DIR}/.tmp_io_final"
RAW_CSV="${L_TMP}/raw_data.csv"
SIF="${PROJECT_DIR}/.containers/clap_pipeline.sif"

# Cleanup e setup iniziale
rm -rf "$L_TMP"
mkdir -p "${L_TMP}/wav_files"
echo "Phase,Iteration,POSIX_Wall,POSIX_Sys,HDF5_Wall,HDF5_Sys" > "$RAW_CSV"

# --- 2. GENERATION PHASE (5000 file) ---
echo "🔨 Phase 0: Generazione Dataset di test (5000 file)..."
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
echo "✅ Generazione completata."

# --- 3. GENERATE SLURM BATCH SCRIPT ---
# Usiamo 'EOF' tra apici per evitare interferenze del login node
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

# Percorsi interni al nodo
NODE_TMP="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.tmp_io_final"
CSV="${NODE_TMP}/raw_data.csv"
SIF="/leonardo_scratch/large/userexternal/lpilegg1/SEC_pipeline/.containers/clap_pipeline.sif"
SSD="/tmp/bench_${SLURM_JOB_ID}"
mkdir -p "${SSD}/wavs"

# --- Probe Python ---
cat << 'PY_PROBE' > "${NODE_TMP}/probe.py"
import time, h5py, soundfile as sf, sys, os
wav_p, h5_p, label, start, count = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]), 500
# Test POSIX
ts = os.times(); ws = time.perf_counter()
for i in range(start, start + count):
    with sf.SoundFile(os.path.join(wav_p, f"track_{i}.wav")) as f: _ = f.read()
te = os.times(); we = time.perf_counter()
p_w, p_s = we - ws, te[1] - ts[1]
# Test HDF5
ts = os.times(); ws = time.perf_counter()
with h5py.File(h5_p, 'r') as h5:
    ds = h5['audio']
    for i in range(start, start + count): _ = ds[i]
te = os.times(); we = time.perf_counter()
h_w, h_s = we - ws, te[1] - ts[1]
print(f"{label},{start//500},{p_w:.6f},{p_s:.6f},{h_w:.6f},{h_s:.6f}")
PY_PROBE

# PHASE 1: LUSTRE
for i in {0..9}; do
    OFFSET=$((i * 500))
    singularity exec --no-home --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "${NODE_TMP}/probe.py" "${NODE_TMP}/wav_files" "${NODE_TMP}/dataset.h5" "LUSTRE" "$OFFSET" >> "$CSV"
done

# PHASE 2: STAGING (Misura media su 10 blocchi)
for i in {0..9}; do
    OFFSET=$((i * 500))
    t1=$(date +%s.%N)
    cp "${NODE_TMP}/wav_files/track_"{$OFFSET..$((OFFSET+499))}.wav "${SSD}/wavs/"
    cp "${NODE_TMP}/dataset.h5" "${SSD}/dataset.h5"
    t2=$(date +%s.%N)
    echo "STAGING,$i,$(python3 -c "print($t2 - $t1)"),0,0,0" >> "$CSV"
    
    # PHASE 3: SSD (Immediato dopo staging)
    singularity exec --no-home --bind "/tmp:/tmp" --bind "/leonardo_scratch:/leonardo_scratch" "$SIF" \
        python3 -u "${NODE_TMP}/probe.py" "${SSD}/wavs" "${SSD}/dataset.h5" "SSD" "$OFFSET" >> "$CSV"
    
    rm -f ${SSD}/wavs/* ${SSD}/dataset.h5
done
EOF

# --- 4. SUBMISSION & ANALYSIS ---
echo "📤 Invio Job Slurm (10 iterazioni)..."
JOB_ID=$(sbatch --parsable "${L_TMP}/run_ultimate.sh")

# Loop di attesa con progresso visibile
while squeue -j "$JOB_ID" | grep -q "$JOB_ID"; do
    echo -n "."
    sleep 5
done
echo -e "\n✅ Job terminato. Elaborazione statistiche..."

# Analisi Finale in Python (direttamente qui nel login node)
python3 -u - <<PY_STATS
import pandas as pd
import numpy as np
try:
    df = pd.read_csv("${RAW_CSV}")
    print("\n" + "="*95)
    print(f"{'REPORT FINALE I/O BENCHMARK (N=10)':^95}")
    print("="*95)
    print(f"{'FASE':<10} | {'POSIX Wall (mean ± std)':<25} | {'HDF5 Wall (mean ± std)':<25} | {'Sys Overhead'}")
    print("-" * 95)
    
    for phase in ['LUSTRE', 'SSD']:
        sub = df[df['Phase'] == phase]
        pw, ps = sub['POSIX_Wall'], sub['POSIX_Sys']
        hw, hs = sub['HDF5_Wall'], sub['HDF5_Sys']
        
        p_str = f"{pw.mean():.4f}s ± {pw.std():.4f}"
        h_str = f"{hw.mean():.4f}s ± {hw.std():.4f}"
        ov = f"P:{(ps.mean()/pw.mean()*100):.1f}% | H:{(hs.mean()/hw.mean()*100):.1f}%"
        
        print(f"{phase:<10} | {p_str:<25} | {h_str:<25} | {ov}")
    
    st = df[df['Phase'] == 'STAGING']['POSIX_Wall']
    print("-" * 95)
    print(f"STAGING IN | Media: {st.mean():.4f}s | Dev.Std: {st.std():.4f} (Lustre -> Local SSD)")
    print("="*95 + "\n")
except Exception as e:
    print(f"Errore nell'analisi: {e}")
PY_STATS
