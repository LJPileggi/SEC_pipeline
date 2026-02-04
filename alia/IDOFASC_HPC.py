import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.spatial import ConvexHull
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import (
    silhouette_score, rand_score, calinski_harabasz_score,
    davies_bouldin_score, fowlkes_mallows_score
)
sys.path.insert(0, '/app')

# Moduli della tua pipeline
from src.utils import HDF5DatasetManager
from src.models import CLAP_initializer
import msclap

# --- PATCH OBBLIGATORIA PER MSCLAP ---
def patched_read_audio(self, audio_path, resample=True): pass 
msclap.CLAP.read_audio = patched_read_audio

#%% Timing
start_time = time.time()

def log_step(msg):
    elapsed = time.time() - start_time
    print(f"[{elapsed:7.2f} sec] {msg}")

# --- REPLICA FUNZIONI ESTRAZIONE ORIGINALI IDOFASC_v7 ---
def extract_mfcc(sig, sr, n=13):
    return np.mean(librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n).T, axis=0)

def extract_gfcc(sig, sr, n=13):
    return np.mean(librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n).T, axis=0)

def extract_cqcc(sig, sr, n=13):
    cq = librosa.feature.chroma_cqt(y=sig, sr=sr).T
    return np.mean(cq, axis=0)[:n]

def compute_metrics(X, y_true, y_pred):
    return {
        "silhouette": silhouette_score(X, y_pred),
        "rand": rand_score(y_true, y_pred),
        "fowlkes_mallows": fowlkes_mallows_score(y_true, y_pred),
        "calinski_harabasz": calinski_harabasz_score(X, y_pred),
        "davies_bouldin": davies_bouldin_score(X, y_pred),
    }

def main():
    # Lettura variabili d'ambiente dinamiche
    input_dir = os.getenv("INPUT_HDF5_DIR")
    output_folder = os.getenv("OUTPUT_RESULTS_PATH")
    audio_format = os.getenv("AUDIO_FORMAT", "wav") # Dinamico!
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    os.makedirs(output_folder, exist_ok=True)

    # Parametri originali IDOFASC_v7
    expl_var = 0.9
    
    # Inizializzazione CLAP patchata
    clap_model = CLAP_initializer(device=device)

    feats = {"emb": [], "mfcc": [], "gfcc": [], "cqcc": []}
    file_paths_basenames = []
    labels_list = []

    # Filtraggio file HDF5 basato sul formato dinamico
    h5_files = sorted([f for f in os.listdir(input_dir) if f.endswith(f'_{audio_format}_dataset.h5')])
    classes = [f.split('_')[0] for f in h5_files]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    k = len(classes)

    log_step(f"Starting preprocessing from HDF5 (Format: {audio_format})")

    for h5_file in h5_files:
        label = h5_file.split('_')[0]
        manager = HDF5DatasetManager(os.path.join(input_dir, h5_file))
        
        # Accesso ai dataset dinamici basati sul formato
        with manager.get_h5file() as f:
            audio_ds = f[f'audio_{audio_format}']
            meta_ds = f[f'metadata_{audio_format}']
            sr_h5 = f.attrs.get('sample_rate', 51200)
            
            log_step(f"Processing class: {label}")
            
            for i in range(len(audio_ds)):
                sig = audio_ds[i].astype('float32')
                fname = meta_ds[i]['track_name'].decode('utf-8')
                
                # Preprocessing originale: Scaling prima delle feature
                sig_norm = StandardScaler().fit_transform(sig.reshape(-1, 1)).flatten()
                feats["mfcc"].append(extract_mfcc(sig_norm, sr_h5))
                feats["gfcc"].append(extract_gfcc(sig_norm, sr_h5))
                feats["cqcc"].append(extract_cqcc(sig_norm, sr_h5))

                # Estrazione CLAP embedding via GPU
                track_t = torch.from_numpy(sig).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = clap_model.clap.get_audio_embeddings(track_t)
                    feats["emb"].append(emb.cpu().numpy().squeeze())
                
                labels_list.append(class_to_idx[label])
                file_paths_basenames.append(fname)

    features = {name: np.array(v) for name, v in feats.items()}
    y_true = np.array(labels_list)
    log_step("Finished preprocessing")

    # --- PCA (Logica originale) ---
    reduced = {}
    for name, X in features.items():
        pca_temp = PCA().fit(X)
        n_comp = np.argmax(np.cumsum(pca_temp.explained_variance_ratio_) >= expl_var) + 1
        reduced[name] = PCA(n_components=n_comp).fit_transform(X)
        log_step(f"{name.upper()} PCA → {n_comp} components")

    # --- CLUSTERING (KMeans + Bisecting originali) ---
    results = {}
    for name, X in reduced.items():
        log_step(f"Clustering {name.upper()}")
        km = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=1000).fit(X)
        results[(name, "kmeans")] = {"model": km, "labels": km.labels_, **compute_metrics(X, y_true, km.labels_)}
        bkm = BisectingKMeans(n_clusters=k, random_state=42, n_init=10, max_iter=1000, bisecting_strategy="biggest_inertia").fit(X)
        results[(name, "bisecting")] = {"model": bkm, "labels": bkm.labels_, **compute_metrics(X, y_true, bkm.labels_)}

    # --- SALVATAGGIO ---
    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    metrics_df.to_excel(os.path.join(output_folder, f"clustering_metrics_{audio_format}.xlsx"))
    log_step("Metrics and plots saved")

    # --- PLOT (Poligoni / Convex Hull originali) ---
    color_list = [cm.get_cmap("tab20" if k <= 20 else "hsv", k)(i) for i in range(k)]
    for (name, algo), res in results.items():
        X2 = PCA(n_components=2).fit_transform(reduced[name])
        plt.figure(figsize=(8, 6))
        for c in range(k):
            pts = X2[res["labels"] == c]
            col = color_list[c % len(color_list)]
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                hp = pts[hull.vertices]
                plt.plot(np.r_[hp[:, 0], hp[0, 0]], np.r_[hp[:, 1], hp[0, 1]], "--", color=col, lw=2)
                plt.fill(hp[:, 0], hp[:, 1], color=col, alpha=0.1)
            plt.plot(np.mean(pts[:, 0]), np.mean(pts[:, 1]), "X", c="black")
        plt.scatter(res["model"].cluster_centers_[:, 0], res["model"].cluster_centers_[:, 1], c="red", s=80, marker="X")
        plt.title(f"{name.upper()} – {algo.capitalize()} Polygon Clusters ({audio_format})")
        plt.savefig(os.path.join(output_folder, f"{name}_{algo}_{audio_format}_polygon.png"), dpi=600)
        plt.close()

    total = time.time() - start_time
    print(f"\n✅ Completed in {total/60:.2f} minutes")

if __name__ == "__main__":
    main()
