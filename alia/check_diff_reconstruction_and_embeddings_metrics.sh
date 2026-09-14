#!/bin/bash
#SBATCH --job-name=eval_heatmaps
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:40:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -p boost_usr_prod
#SBATCH -A IscrC_BrISkite_0
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

TEMP_DIR="/leonardo_scratch/large/userexternal/$USER/tmp_eval_$SLURM_JOB_ID"
SIF_FILE="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.containers/clap_pipeline.sif"
CLAP_SCRATCH_WEIGHTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/CLAP_weights_2023.pth"
CLAP_BN0_CONSTANTS="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/clap_bn0_constants.npz"
ROBERTA_PATH="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.clap_weights/roberta-base"
DATASEC_GLOBAL="/leonardo_scratch/large/userexternal/$USER/dataSEC"
MODELS_GLOBAL="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/.models/diff_model"
RESULTS_DIR="/leonardo_scratch/large/userexternal/$USER/SEC_pipeline/results/supervisor_report"

mkdir -p "$TEMP_DIR/weights"
mkdir -p "$TEMP_DIR/roberta-base"
mkdir -p "$TEMP_DIR/numba_cache"
mkdir -p "$TEMP_DIR/data"
mkdir -p "$TEMP_DIR/models"
mkdir -p "$RESULTS_DIR"

echo "📦 Setup directory e pesi..."
cp "$CLAP_SCRATCH_WEIGHTS" "$TEMP_DIR/weights/CLAP_weights_2023.pth" 2>/dev/null
[ -f "$CLAP_BN0_CONSTANTS" ] && cp "$CLAP_BN0_CONSTANTS" "$TEMP_DIR/weights/clap_bn0_constants.npz" 2>/dev/null
cp -r "$ROBERTA_PATH/." "$TEMP_DIR/roberta-base/" 2>/dev/null
cp -r "$DATASEC_GLOBAL/RAW_DATASET/raw_wav"/*.h5 "$TEMP_DIR/data/" 2>/dev/null
cp "$MODELS_GLOBAL"/*.pt "$TEMP_DIR/models/" 2>/dev/null

# ==============================================================================
# SCRIPT 1: METRICHE SPETTRALI PER CLASSE + HEATMAP MATPLOTLIB
# ==============================================================================
cat << 'EOF' > "$TEMP_DIR/eval_spectral_per_class.py"
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import scipy.stats

sys.path.insert(0, "/app")

from src.utils import get_config_from_yaml
from src.filterbank_diffusion.models.unet import SpectrogramUNet
from src.filterbank_diffusion.models.diffusion import ConditionalGaussianDiffusion
from src.filterbank_diffusion.data.dataset import DistributedAudioRAWDataset
from src.filterbank_diffusion.pipeline.spectral import OnlineSpectrogramPipeline

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🔧 Device selezionato: {device}")

classes_list, _, _, _, sampling_rate, _, _, seed, _, _, _ = get_config_from_yaml("config0.yaml")
print(f"📋 Classi totali registrate ({len(classes_list)}): {classes_list}")

weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
spectral_pipeline = OnlineSpectrogramPipeline(weights_path=weights_path, sample_rate=sampling_rate, device=device).to(device)

model_dir = "/tmp_data/models"
pts = [f for f in os.listdir(model_dir) if f.endswith(".pt")]
pts_sorted = sorted(pts, key=lambda x: int(x.replace("unet_epoch_", "").replace(".pt", "")))
latest_ckpt = os.path.join(model_dir, pts_sorted[-1])
print(f"📦 Checkpoint U-Net caricato: {latest_ckpt}")

unet = SpectrogramUNet(base_channels=64, emb_dim=256).to(device)
ckpt = torch.load(latest_ckpt, map_location=device)
unet.load_state_dict(ckpt['model_state_dict'])
diffusion_scheduler = ConditionalGaussianDiffusion(unet_model=unet, timesteps=1000).to(device)

raw_dataset_root = "/tmp_data/data"
test_dataset = DistributedAudioRAWDataset(base_dir=raw_dataset_root, split="test", target_samples_per_class=30)
test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=2)

out_dir = os.getenv("RESULTS_DIR", "/app/results")
os.makedirs(out_dir, exist_ok=True)

def compute_track_metrics(p_clean, q_clean):
    frob = torch.norm(p_clean - q_clean, p='fro').item()
    p_energy = torch.exp(p_clean).flatten().double().cpu().numpy()
    q_energy = torch.exp(q_clean).flatten().double().cpu().numpy()

    p_sum, q_sum = np.sum(p_energy), np.sum(q_energy)
    p_energy = p_energy / p_sum if p_sum > 0 else np.ones_like(p_energy) / len(p_energy)
    q_energy = q_energy / q_sum if q_sum > 0 else np.ones_like(q_energy) / len(q_energy)

    eps = 1e-12
    p_energy = np.clip(p_energy, eps, 1.0)
    q_energy = np.clip(q_energy, eps, 1.0)

    kl = scipy.stats.entropy(p_energy, q_energy)
    wass = scipy.stats.wasserstein_distance(p_energy, q_energy)
    return frob, float(kl), float(wass)

print("\n" + "="*80)
print("📊 CALCOLO METRICHE SPETTRALI PER CLASSE (1/3 d'ottava)")
print("="*80)

records = []
native_centroids = {c: [] for c in classes_list}
rec_centroids = {c: [] for c in classes_list}

with torch.no_grad():
    for raw_audio, class_labels in test_dataloader:
        raw_audio = raw_audio.to(device, non_blocking=True)
        frac_tensor = torch.full((raw_audio.shape[0],), fill_value=3.0, device=device)

        x_0_pristine, x_cond = spectral_pipeline(raw_audio, format_id=1, fraction_id=3, device=device)
        x_rec = diffusion_scheduler.sample_ddim(x_cond, fraction_id=frac_tensor, ddim_steps=25)

        x_0_clean = torch.nan_to_num(x_0_pristine, nan=0.0)[:, :, :, :1140]
        x_rec_clean = torch.nan_to_num(x_rec, nan=0.0)[:, :, :, :1140]

        for b in range(x_0_clean.shape[0]):
            c_idx = class_labels[b].item()
            c_name = classes_list[c_idx] if c_idx < len(classes_list) else f"cls_{c_idx}"

            frob, kl, wass = compute_track_metrics(x_0_clean[b], x_rec_clean[b])
            records.append({
                'class': c_name,
                'frobenius': frob,
                'kl_divergence': kl,
                'wasserstein': wass
            })

            if c_name in native_centroids:
                native_centroids[c_name].append(x_0_clean[b].squeeze().cpu().numpy())
                rec_centroids[c_name].append(x_rec_clean[b].squeeze().cpu().numpy())

df = pd.DataFrame(records)
class_summary = df.groupby('class')[['frobenius', 'kl_divergence', 'wasserstein']].agg(['mean', 'std'])
class_summary = class_summary.reindex(classes_list)
class_summary.to_csv(os.path.join(out_dir, "spectral_metrics_per_class.csv"))
print("\n" + class_summary.to_string())

# MATRICI DI DISTANZA CENTROIDI
n_cls = len(classes_list)
frob_matrix = np.full((n_cls, n_cls), np.nan)
wass_matrix = np.full((n_cls, n_cls), np.nan)

for i, c_rec in enumerate(classes_list):
    if len(rec_centroids[c_rec]) == 0:
        continue
    rec_c = np.mean(rec_centroids[c_rec], axis=0)
    for j, c_nat in enumerate(classes_list):
        if len(native_centroids[c_nat]) == 0:
            continue
        nat_c = np.mean(native_centroids[c_nat], axis=0)

        frob_matrix[i, j] = np.linalg.norm(rec_c - nat_c, 'fro')

        p_prof = np.mean(rec_c, axis=1)
        q_prof = np.mean(nat_c, axis=1)
        p_prob = np.exp(p_prof) / np.sum(np.exp(p_prof))
        q_prob = np.exp(q_prof) / np.sum(np.exp(q_prof))
        wass_matrix[i, j] = scipy.stats.wasserstein_distance(p_prob, q_prob)

df_frob_mat = pd.DataFrame(frob_matrix, index=classes_list, columns=classes_list)
df_frob_mat.to_csv(os.path.join(out_dir, "centroid_frobenius_distance_matrix.csv"))

df_wass_mat = pd.DataFrame(wass_matrix, index=classes_list, columns=classes_list)
df_wass_mat.to_csv(os.path.join(out_dir, "centroid_wasserstein_distance_matrix.csv"))

# FUNZIONE PLOT HEATMAP NATIVA MATPLOTLIB
def plot_matrix_heatmap(mat, labels, title, save_path, cmap="viridis", fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(mat, cmap=cmap, aspect='auto')

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Classi Native", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel("Classi Ricostruite (DDIM)", fontsize=12, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=15)

    # Annotazioni numeriche all'interno delle celle
    valid_vals = mat[~np.isnan(mat)]
    thresh = (np.nanmax(valid_vals) + np.nanmin(valid_vals)) / 2.0 if len(valid_vals) > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if val < thresh else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center", color=text_color, fontsize=7)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"🖼️ Heatmap salvata con successo in: {save_path}")

plot_matrix_heatmap(
    frob_matrix, classes_list, 
    "Matrice di Confusione Geometrica: Distanza di Frobenius tra Centroidi",
    os.path.join(out_dir, "centroid_frobenius_heatmap.png"),
    cmap="magma_r", fmt="{:.1f}"
)

plot_matrix_heatmap(
    wass_matrix, classes_list, 
    "Matrice di Confusione Spettrale: Distanza 1D Wasserstein tra Profili",
    os.path.join(out_dir, "centroid_wasserstein_heatmap.png"),
    cmap="viridis_r", fmt="{:.3f}"
)

test_dataset.close()
EOF

# ==============================================================================
# SCRIPT 2: COSINE SIMILARITY EMBEDDINGS HDF5 + HEATMAP DI SEPARABILITÀ
# ==============================================================================
cat << 'EOF' > "$TEMP_DIR/eval_hdf5_cosine_similarity.py"
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import h5py
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

sys.path.insert(0, "/app")
from src.utils import get_config_from_yaml

classes_list, _, _, _, _, _, _, _, _, _, _ = get_config_from_yaml("config0.yaml")

h5_octave_path = "/leonardo_scratch/large/userexternal/" + os.environ["USER"] + "/dataSEC/PREPROCESSED_DATASET/wav/3_octave/7_secs/combined_test.h5"
h5_raw_path = "/leonardo_scratch/large/userexternal/" + os.environ["USER"] + "/dataSEC/PREPROCESSED_DATASET/wav/0_octave/7_secs/combined_test.h5"
out_dir = os.getenv("RESULTS_DIR", "/app/results")

print("\n" + "="*80)
print("🔍 CALCOLO COSINE SIMILARITY EMBEDDINGS SU TUTTE LE CLASSI")
print(f"   • Octave (Rec): {h5_octave_path}")
print(f"   • Raw (Native): {h5_raw_path}")
print("="*80)

if not os.path.exists(h5_octave_path) or not os.path.exists(h5_raw_path):
    print("❌ Errore: File HDF5 non trovati!")
    sys.exit(1)

with h5py.File(h5_octave_path, 'r') as hf_oct, h5py.File(h5_raw_path, 'r') as hf_raw:
    oct_dset = hf_oct['embedding_dataset']
    raw_dset = hf_raw['embedding_dataset']

    oct_ids = [k.decode('utf-8') if isinstance(k, bytes) else str(k) for k in oct_dset['ID'][:]]
    raw_ids = [k.decode('utf-8') if isinstance(k, bytes) else str(k) for k in raw_dset['ID'][:]]

    oct_classes = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in oct_dset['classes'][:]]
    raw_classes = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in raw_dset['classes'][:]]

    oct_embs = oct_dset['embeddings'][:]
    raw_embs = raw_dset['embeddings'][:]

raw_lookup = {raw_ids[i]: (raw_embs[i], raw_classes[i]) for i in range(len(raw_ids))}

paired_oct_embs = []
paired_raw_embs = []
paired_classes = []
paired_keys = []

for i, key in enumerate(oct_ids):
    if key in raw_lookup:
        paired_oct_embs.append(oct_embs[i])
        paired_raw_embs.append(raw_lookup[key][0])
        paired_classes.append(oct_classes[i])
        paired_keys.append(key)

if not paired_oct_embs:
    print("⚠️ Fallback allineamento sequenziale...")
    min_len = min(len(oct_embs), len(raw_embs))
    paired_oct_embs = oct_embs[:min_len]
    paired_raw_embs = raw_embs[:min_len]
    paired_classes = oct_classes[:min_len]
    paired_keys = oct_ids[:min_len]

t_oct = F.normalize(torch.from_numpy(np.array(paired_oct_embs)).float(), p=2, dim=-1)
t_raw = F.normalize(torch.from_numpy(np.array(paired_raw_embs)).float(), p=2, dim=-1)

cosine_sims = F.cosine_similarity(t_raw, t_oct, dim=-1).cpu().numpy()

df_pairs = pd.DataFrame({
    'ID': paired_keys,
    'class': paired_classes,
    'cosine_similarity': cosine_sims
})
df_pairs.to_csv(os.path.join(out_dir, "pairwise_embedding_similarities.csv"), index=False)

cls_stats = df_pairs.groupby('class')['cosine_similarity'].agg(['count', 'mean', 'std', 'min', 'max'])
cls_stats = cls_stats.reindex(classes_list)
cls_stats.to_csv(os.path.join(out_dir, "cosine_similarity_per_class.csv"))
print("\n📊 COSINE SIMILARITY PER CLASSE:")
print(cls_stats.to_string())

# CENTROIDI LATENTI E MATRICE DI SIMILARITÀ N x N
raw_centroids = {}
oct_centroids = {}
for c in classes_list:
    mask = [cls == c for cls in paired_classes]
    if any(mask):
        raw_centroids[c] = F.normalize(t_raw[mask].mean(dim=0, keepdim=True), p=2, dim=-1)
        oct_centroids[c] = F.normalize(t_oct[mask].mean(dim=0, keepdim=True), p=2, dim=-1)

n_cls = len(classes_list)
cos_matrix = np.full((n_cls, n_cls), np.nan)

for i, c_rec in enumerate(classes_list):
    if c_rec not in oct_centroids:
        continue
    for j, c_nat in enumerate(classes_list):
        if c_nat not in raw_centroids:
            continue
        cos_matrix[i, j] = F.cosine_similarity(oct_centroids[c_rec], raw_centroids[c_nat], dim=-1).item()

df_cos_mat = pd.DataFrame(cos_matrix, index=classes_list, columns=classes_list)
df_cos_mat.to_csv(os.path.join(out_dir, "embedding_cosine_similarity_matrix.csv"))

# HEATMAP MATPLOTLIB COSINE SIMILARITY
fig, ax = plt.subplots(figsize=(12, 10))
im = ax.imshow(cos_matrix, cmap='coolwarm', vmin=-1.0, vmax=1.0, aspect='auto')

cbar = ax.figure.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=10)

ax.set_xticks(np.arange(n_cls))
ax.set_yticks(np.arange(n_cls))
ax.set_xticklabels(classes_list, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(classes_list, fontsize=9)

ax.set_xlabel("Centroidi CLAP Nativi", fontsize=12, fontweight='bold', labelpad=10)
ax.set_ylabel("Centroidi CLAP Ricostruiti", fontsize=12, fontweight='bold', labelpad=10)
ax.set_title("Matrice di Similarità Coseno tra Centroidi Latenti (CLAP Space)", fontsize=14, fontweight='bold', pad=15)

for i in range(n_cls):
    for j in range(n_cls):
        val = cos_matrix[i, j]
        if not np.isnan(val):
            text_color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=7)

fig.tight_layout()
cos_plot_path = os.path.join(out_dir, "embedding_cosine_similarity_heatmap.png")
plt.savefig(cos_plot_path, dpi=300)
plt.close(fig)
print(f"🖼️ Heatmap Similarità Coseno salvata in: {cos_plot_path}")

# CALCOLO MARGINI DI SEPARABILITÀ
delta_records = []
for i in range(len(paired_classes)):
    c_true = paired_classes[i]
    if c_true not in raw_centroids:
        continue
    emb_rec = t_oct[i:i+1]

    sim_intra = F.cosine_similarity(emb_rec, raw_centroids[c_true], dim=-1).item()
    other_sims = [F.cosine_similarity(emb_rec, raw_centroids[c_other], dim=-1).item() 
                  for c_other in raw_centroids if c_other != c_true]
    sim_inter = max(other_sims) if other_sims else np.nan

    delta_records.append({
        'class': c_true,
        'sim_intra': sim_intra,
        'sim_inter_max': sim_inter,
        'margin_delta': sim_intra - sim_inter if not np.isnan(sim_inter) else np.nan
    })

df_delta = pd.DataFrame(delta_records)
df_delta.to_csv(os.path.join(out_dir, "class_separability_margins.csv"), index=False)

margin_summary = df_delta.groupby('class')[['sim_intra', 'sim_inter_max', 'margin_delta']].mean()
margin_summary = margin_summary.reindex(classes_list)
margin_summary.to_csv(os.path.join(out_dir, "class_separability_margins_summary.csv"))

print("\n" + "="*80)
print("🎯 MARGINI DI SEPARABILITÀ SULLE CLASSI TOTALI:")
print("="*80)
print(margin_summary.to_string())
EOF

export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/weights/clap_bn0_constants.npz"
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export RESULTS_DIR="$RESULTS_DIR"
export HF_HUB_OFFLINE=1

echo "🚀 Esecuzione Analisi 1 (Spettrale + Heatmap)..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 /tmp_data/eval_spectral_per_class.py

echo "🚀 Esecuzione Analisi 2 (Embedding Coseno + Heatmap)..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 /tmp_data/eval_hdf5_cosine_similarity.py

rm -rf "$TEMP_DIR"
echo "✅ Job terminato! Immagini e tabelle generate in: $RESULTS_DIR"
