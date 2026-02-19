import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import librosa
import h5py
import scipy.fftpack as fftpack
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, calinski_harabasz_score,
    davies_bouldin_score, fowlkes_mallows_score
)

# Priority to modules in /app
sys.path.insert(0, '/app')

# Production pipeline patches and redirects
import huggingface_hub
import transformers
import msclap

def universal_path_redirect(*args, **kwargs):
    weights_path = os.getenv("LOCAL_CLAP_WEIGHTS_PATH")
    text_path = os.getenv("CLAP_TEXT_ENCODER_PATH")
    if any(x for x in args if 'msclap' in str(x)) or 'CLAP_weights' in str(kwargs):
        return weights_path
    filename = kwargs.get('filename') or (args[1] if len(args) > 1 else None)
    if filename and text_path:
        return os.path.join(text_path, str(filename))
    return text_path

huggingface_hub.hf_hub_download = universal_path_redirect
transformers.utils.hub.cached_file = universal_path_redirect
transformers.utils.hub.hf_hub_download = universal_path_redirect
msclap.CLAPWrapper.hf_hub_download = universal_path_redirect

from src.utils import HDF5DatasetManager
from src.models import CLAP_initializer

# Patch MSCLAP to bypass problematic internal audio loading
def patched_read_audio(self, audio_path, resample=True): pass 
msclap.CLAP.read_audio = patched_read_audio

start_time = time.time()
def log_step(msg):
    print(f"[{time.time() - start_time:7.2f} sec] {msg}", flush=True)

# --- FEATURE EXTRACTION FUNCTIONS ---

def extract_mfcc(sig, sr, n=13):
    return np.mean(librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n).T, axis=0)

def extract_gfcc(sig, sr, n=13):
    """
    Patched implementation à la librosa for GFCC extraction.
    Replaces Mel-filterbank with Gammatone/ERB and Log with Power compression (0.25).
    """
    # 1. Filterbank parameters (40 filters to match librosa's melspectrogram default)
    n_filters = 40 
    f_min = 20
    f_max = sr / 2
    
    # 2. Compute center frequencies on the ERB (Equivalent Rectangular Bandwidth) scale
    erb_min = 21.4 * np.log10(4.37e-3 * f_min + 1)
    erb_max = 21.4 * np.log10(4.37e-3 * f_max + 1)
    erb_centers = np.linspace(erb_min, erb_max, n_filters + 2)
    freq_centers = (10**(erb_centers / 21.4) - 1) / 4.37e-3
    
    # 3. Compute Power Spectrogram (identical STFT parameters for comparability)
    stft = np.abs(librosa.stft(sig))**2
    freqs = librosa.fft_frequencies(sr=sr)
    
    # 4. Construct the Gammatone Filterbank (Triangular approximation in ERB scale)
    filter_bank = np.zeros((n_filters, stft.shape[0]))
    for i in range(1, n_filters + 1):
        lower = freq_centers[i-1]
        center = freq_centers[i]
        upper = freq_centers[i+1]
        
        # Frequency response of the filter
        filter_bank[i-1] = np.maximum(0, np.minimum((freqs - lower) / (center - lower), 
                                                   (upper - freqs) / (upper - center)))
    
    # 5. Apply filterbank and Power Compression (0.25 exponent)
    raw_gfcc = np.dot(filter_bank, stft)
    compressed_gfcc = np.power(raw_gfcc + 1e-10, 0.25) 
    
    # 6. Apply Discrete Cosine Transform (DCT Type-II)
    gfcc = fftpack.dct(compressed_gfcc, axis=0, type=2, norm='ortho')[:n]
    
    return np.mean(gfcc.T, axis=0)

def extract_cqcc(sig, sr, n=13):
    cq = librosa.feature.chroma_cqt(y=sig, sr=sr).T
    return np.mean(cq, axis=0)[:n]

def compute_metrics(X, y_true, y_pred):
    return {
        "silhouette": silhouette_score(X, y_pred),
        "rand": adjusted_rand_score(y_true, y_pred),
        "fowlkes_mallows": fowlkes_mallows_score(y_true, y_pred),
        "calinski_harabasz": calinski_harabasz_score(X, y_pred),
        "davies_bouldin": davies_bouldin_score(X, y_pred),
    }

def main():
    input_dir = os.getenv("INPUT_HDF5_DIR")
    output_folder = os.getenv("OUTPUT_RESULTS_PATH")
    audio_format = os.getenv("AUDIO_FORMAT", "wav")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not input_dir or not os.path.exists(input_dir):
        print(f"❌ ERROR: Invalid input directory: {input_dir}")
        sys.exit(1)

    log_step(f"🎸 Initializing CLAP (Offline Mode)")
    clap_model, get_audio_embeddings, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())

    feats = {"emb": [], "mfcc": [], "gfcc": [], "cqcc": []}
    all_filenames, all_labels = [], []

    # Pattern: ClassName_format_dataset.h5
    h5_files = sorted([f for f in os.listdir(input_dir) if f.endswith(f'_{audio_format}_dataset.h5')])
    
    if not h5_files:
        print(f"❌ ERROR: No .h5 files found in {input_dir}")
        sys.exit(1)

    classes = [f.replace(f'_{audio_format}_dataset.h5', '') for f in h5_files]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for h5_file in h5_files:
        label = h5_file.replace(f'_{audio_format}_dataset.h5', '')
        h5_path = os.path.join(input_dir, h5_file)
        
        # Initialize manager (opens h5 file automatically)
        manager = HDF5DatasetManager(h5_path, audio_format=audio_format)
        
        log_step(f"Processing class: {label}")
        audio_ds = manager.hf[f'audio_{audio_format}']
        meta_ds = manager.hf[f'metadata_{audio_format}']
        sr_h5 = manager.hf.attrs.get('sample_rate', 51200)
        
        for i in range(len(audio_ds)):
            sig = audio_ds[i].astype('float32')
            fname = meta_ds[i]['track_name'].decode('utf-8')
            
            # Standardization before extraction
            sig_norm = StandardScaler().fit_transform(sig.reshape(-1, 1)).flatten()
            feats["mfcc"].append(extract_mfcc(sig_norm, sr_h5))
            feats["gfcc"].append(extract_gfcc(sig_norm, sr_h5))
            feats["cqcc"].append(extract_cqcc(sig_norm, sr_h5))

            track_t = torch.from_numpy(sig).float().unsqueeze(0).to(device)
            with torch.no_grad():
                emb = get_audio_embeddings(track_t)[0][0]
                feats["emb"].append(emb.cpu().numpy().squeeze())
            
            all_labels.append(class_to_idx[label])
            all_filenames.append(fname)
        
        manager.close()

    # Clustering logic
    features = {name: np.array(v) for name, v in feats.items()}
    y_true = np.array(all_labels)
    results = {}

    log_step("PCA Analysis and Clustering...")
    for name, X in features.items():
        pca_full = PCA().fit(X)
        n_comp = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.9) + 1
        X_pca = PCA(n_components=n_comp).fit_transform(X)

        km = KMeans(n_clusters=len(classes), random_state=42, n_init=50, max_iter=1000).fit(X_pca)
        results[(name, "kmeans")] = {"model": km, "labels": km.labels_, "pca_data": X_pca, **compute_metrics(X, y_true, km.labels_)}
        
        bkm = BisectingKMeans(n_clusters=len(classes), random_state=42, n_init=10, max_iter=1000).fit(X_pca)
        results[(name, "bisecting")] = {"model": bkm, "labels": bkm.labels_, "pca_data": X_pca, **compute_metrics(X, y_true, bkm.labels_)}

    # Save Output
    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    metrics_df.to_csv(os.path.join(output_folder, f"metrics_{audio_format}.csv"))

    # Plotting Convex Hulls
    color_list = [cm.get_cmap("tab20" if len(classes) <= 20 else "hsv", len(classes))(i) for i in range(len(classes))]
    for (name, algo), res in results.items():
        X2 = PCA(n_components=2).fit_transform(res["pca_data"])
        plt.figure(figsize=(10, 8))
        
        for c in range(len(classes)):
            col = color_list[c % len(color_list)]
            pts_cluster = X2[res["labels"] == c]
            pts_true = X2[y_true == c]
            
            if len(pts_cluster) >= 3:
                try:
                    jitter = np.random.normal(0, 1e-9, pts_cluster.shape)
                    pts_jittered = pts_cluster + jitter
                    hull = ConvexHull(pts_jittered)
                    hp = pts_jittered[hull.vertices]
                    plt.plot(np.r_[hp[:, 0], hp[0, 0]], np.r_[hp[:, 1], hp[0, 1]], "--", color=col, lw=1.5, alpha=0.6)
                    plt.fill(hp[:, 0], hp[:, 1], color=col, alpha=0.05)
                except Exception as e:
                    print(f"⚠️ Warning: Hull failed for cluster {c}: {e}")

            if len(pts_true) > 0:
                gt_mean_x = np.mean(pts_true[:, 0])
                gt_mean_y = np.mean(pts_true[:, 1])
                plt.text(gt_mean_x, gt_mean_y, str(c+1), 
                         fontsize=14, fontweight='extra bold', color='black',
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor=col, boxstyle='round,pad=0.2'))
                plt.plot(gt_mean_x, gt_mean_y, "o", c="black", markersize=4)

        plt.scatter(res["model"].cluster_centers_[:, 0], 
                    res["model"].cluster_centers_[:, 1], 
                    c="red", s=100, marker="X", label="Cluster Centroids", alpha=0.8)
        plt.title(f"{name.upper()} – {algo.capitalize()} ({audio_format})")
        plt.savefig(os.path.join(output_folder, f"plot_{name}_{algo}_{audio_format}.png"), dpi=600)
        plt.close()
    
    log_step(f"✅ Completed. Results in {output_folder}")

if __name__ == "__main__":
    main()
