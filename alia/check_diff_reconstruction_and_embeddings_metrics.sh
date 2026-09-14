#!/bin/bash
#SBATCH --job-name=eval_metrics_all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=00:45:00
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
print("📊 PARTE 1: CALCOLO METRICHE SPETTRALI PER CLASSE (1/3 d'ottava)")
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

def plot_matrix_heatmap(mat, labels, title, save_path, cmap="viridis", fmt="{:.1f}"):
    fig, ax = plt.subplots(figsize=(13, 11))
    im = ax.imshow(mat, cmap=cmap, aspect='auto')

    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=10)

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)

    ax.set_xlabel("Classi Native", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_ylabel("Classi Ricostruite (DDIM)", fontsize=11, fontweight='bold', labelpad=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=15)

    valid_vals = mat[~np.isnan(mat)]
    thresh = (np.nanmax(valid_vals) + np.nanmin(valid_vals)) / 2.0 if len(valid_vals) > 0 else 0
    for i in range(len(labels)):
        for j in range(len(labels)):
            val = mat[i, j]
            if not np.isnan(val):
                text_color = "white" if val < thresh else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center", color=text_color, fontsize=6.5)

    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"🖼️ Heatmap salvata in: {save_path}")

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
# SCRIPT 2: SIMILARITÀ CENTROIDI EMBEDDINGS HDF5 (NESSUN VINCOLO SU ID)
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

h5_octave_path = f"/leonardo_scratch/large/userexternal/{os.environ['USER']}/dataSEC/PREPROCESSED_DATASET/wav/3_octave/7_secs/combined_test.h5"
h5_raw_path = f"/leonardo_scratch/large/userexternal/{os.environ['USER']}/dataSEC/PREPROCESSED_DATASET/wav/0_octave/7_secs/combined_test.h5"
out_dir = os.getenv("RESULTS_DIR", "/app/results")

print("\n" + "="*80)
print("🔍 PARTE 2: SIMILARITÀ TRA CENTROIDI EMBEDDINGS (3_octave vs 0_octave)")
print("="*80)

if not os.path.exists(h5_octave_path) or not os.path.exists(h5_raw_path):
    print(f"❌ Errore: File HDF5 non trovati!\n   • Octave: {h5_octave_path}\n   • Raw: {h5_raw_path}")
    sys.exit(1)

with h5py.File(h5_octave_path, 'r') as hf_oct, h5py.File(h5_raw_path, 'r') as hf_raw:
    oct_dset = hf_oct['embedding_dataset']
    raw_dset = hf_raw['embedding_dataset']

    oct_classes = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in oct_dset['classes'][:]]
    raw_classes = [c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in raw_dset['classes'][:]]

    oct_embs = torch.from_numpy(oct_dset['embeddings'][:]).float()
    raw_embs = torch.from_numpy(raw_dset['embeddings'][:]).float()

oct_embs = F.normalize(oct_embs, p=2, dim=-1)
raw_embs = F.normalize(raw_embs, p=2, dim=-1)

raw_centroids = {}
oct_centroids = {}
oct_embs_by_class = {}

for c in classes_list:
    mask_raw = [cls == c for cls in raw_classes]
    mask_oct = [cls == c for cls in oct_classes]

    if any(mask_raw):
        raw_c = raw_embs[mask_raw].mean(dim=0, keepdim=True)
        raw_centroids[c] = F.normalize(raw_c, p=2, dim=-1)

    if any(mask_oct):
        oct_subset = oct_embs[mask_oct]
        oct_embs_by_class[c] = oct_subset
        oct_c = oct_subset.mean(dim=0, keepdim=True)
        oct_centroids[c] = F.normalize(oct_c, p=2, dim=-1)

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
df_cos_mat.to_csv(os.path.join(out_dir, "centroid_embedding_cosine_matrix.csv"))

class_summary_rows = []

for c in classes_list:
    has_raw = c in raw_centroids
    has_oct = c in oct_centroids

    if not has_raw or not has_oct:
        class_summary_rows.append({
            'class': c,
            'n_samples_oct': len(oct_embs_by_class.get(c, [])),
            'centroid_cosine_intra': np.nan,
            'mean_sample_to_native_centroid': np.nan,
            'max_inter_centroid_cosine': np.nan,
            'centroid_margin_delta': np.nan
        })
        continue

    c_intra = F.cosine_similarity(oct_centroids[c], raw_centroids[c], dim=-1).item()
    sample_sims = F.cosine_similarity(oct_embs_by_class[c], raw_centroids[c], dim=-1).cpu().numpy()
    mean_sample_sim = sample_sims.mean()

    other_sims = [F.cosine_similarity(oct_centroids[c], raw_centroids[other_c], dim=-1).item()
                  for other_c in raw_centroids if other_c != c]
    max_inter = max(other_sims) if other_sims else np.nan

    class_summary_rows.append({
        'class': c,
        'n_samples_oct': len(oct_embs_by_class[c]),
        'centroid_cosine_intra': c_intra,
        'mean_sample_to_native_centroid': mean_sample_sim,
        'max_inter_centroid_cosine': max_inter,
        'centroid_margin_delta': c_intra - max_inter if not np.isnan(max_inter) else np.nan
    })

df_summary = pd.DataFrame(class_summary_rows)
df_summary.to_csv(os.path.join(out_dir, "class_embedding_similarity_summary.csv"), index=False)

print("\n" + "="*80)
print("📊 TABELLA SIMILARITÀ EMBEDDING PER CLASSE (22 CLASSI)")
print("="*80)
print(df_summary.to_string())

fig, ax = plt.subplots(figsize=(13, 11))
im = ax.imshow(cos_matrix, cmap='coolwarm', vmin=-0.2, vmax=1.0, aspect='auto')

cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.ax.tick_params(labelsize=10)
cbar.set_label("Cosine Similarity", fontsize=11, fontweight='bold')

ax.set_xticks(np.arange(n_cls))
ax.set_yticks(np.arange(n_cls))
ax.set_xticklabels(classes_list, rotation=45, ha="right", fontsize=8)
ax.set_yticklabels(classes_list, fontsize=8)

ax.set_xlabel("Centroidi Audio Grezzo Nativo (0_octave)", fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel("Centroidi Ricostruiti da Terze (3_octave)", fontsize=11, fontweight='bold', labelpad=10)
ax.set_title("Matrice di Similarità Coseno tra Centroidi di Classe (CLAP Space)", fontsize=13, fontweight='bold', pad=15)

for i in range(n_cls):
    for j in range(n_cls):
        val = cos_matrix[i, j]
        if not np.isnan(val):
            text_color = "white" if abs(val) > 0.65 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=6.5)

fig.tight_layout()
heatmap_path = os.path.join(out_dir, "centroid_embedding_cosine_heatmap.png")
plt.savefig(heatmap_path, dpi=300)
plt.close(fig)
print(f"\n🖼️ Heatmap salvata in: {heatmap_path}")
print("="*80)
EOF

# Esportazione variabili d'ambiente per il container
export LOCAL_CLAP_WEIGHTS_PATH="/tmp_data/weights/CLAP_weights_2023.pth"
export LOCAL_CLAP_BN0_CONSTANTS_PATH="/tmp_data/weights/clap_bn0_constants.npz"
export CLAP_TEXT_ENCODER_PATH="/tmp_data/roberta-base"
export NUMBA_CACHE_DIR="/tmp_data/numba_cache"
export RESULTS_DIR="$RESULTS_DIR"
export HF_HUB_OFFLINE=1

echo "🚀 Esecuzione Parte 1: Metriche Spettrali e Centroidi per Classe..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 /tmp_data/eval_spectral_per_class.py

echo "🚀 Esecuzione Parte 2: Similarità Coseno tra Centroidi HDF5 e Heatmap..."
singularity exec --nv --no-home \
    --bind "/leonardo_scratch:/leonardo_scratch" \
    --bind "$TEMP_DIR:/tmp_data" \
    --bind "$(pwd):/app" --pwd "/app" \
    "$SIF_FILE" \
    python3 /tmp_data/eval_hdf5_cosine_similarity.py

rm -rf "$TEMP_DIR"
echo "✅ Analisi unificata completata con successo! Tutti i file e i grafici sono in: $RESULTS_DIR"
