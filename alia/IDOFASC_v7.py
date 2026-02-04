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

# Moduli della pipeline
from src.utils import HDF5DatasetManager
from src.models import CLAP_initializer
import msclap

# --- PATCH DI EMERGENZA ---
def patched_read_audio(self, audio_path, resample=True): pass 
msclap.CLAP.read_audio = patched_read_audio

# --- REPLICA FUNZIONI ORIGINALI IDOFASC_v7 ---
def extract_mfcc(sig, sr, n=13):
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
    start_time = time.time()
    input_dir = os.getenv("INPUT_HDF5_DIR")
    output_folder = os.getenv("OUTPUT_RESULTS_PATH")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(output_folder, exist_ok=True)

    # Parametri originali
    expl_var = 0.9
    
    # Inizializzazione CLAP (usa la tua patch offline)
    clap_model = CLAP_initializer(device=device)

    feats = {"emb": [], "mfcc": [], "cqcc": []}
    all_labels = []
    all_filenames = []

    h5_files = sorted([f for f in os.listdir(input_dir) if f.endswith('.h5')])
    classes = [f.replace('.h5', '') for f in h5_files]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    # --- LOOP DI ESTRAZIONE DA HDF5 ---
    for h5_file in h5_files:
        label = h5_file.replace('.h5', '')
        manager = HDF5DatasetManager(os.path.join(input_dir, h5_file))
        
        with manager.get_h5file() as f:
            audio_ds = f['audio']
            # Leonardo usa 48kHz o 44.1kHz negli HDF5, gestiamo sr
            current_sr = f['audio'].attrs.get('sample_rate', 44100) 
            
            print(f"[{time.time()-start_time:.2f}s] Processing class: {label}")
            
            for i in range(len(audio_ds)):
                sig = audio_ds[i]
                
                # 1. Feature Artigianali (su CPU come nell'originale)
                sig_norm = StandardScaler().fit_transform(sig.reshape(-1, 1)).flatten()
                feats["mfcc"].append(extract_mfcc(sig_norm, current_sr))
                feats["cqcc"].append(extract_cqcc(sig_norm, current_sr))

                # 2. CLAP Embeddings (su GPU)
                track_t = torch.from_numpy(sig).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = clap_model.clap.get_audio_embeddings(track_t)
                    feats["emb"].append(emb.cpu().numpy().squeeze())
                
                all_labels.append(class_to_idx[label])

    # Conversione in array numpy
    features = {k: np.array(v) for k, v in feats.items()}
    y_true = np.array(all_labels)

    # --- PCA E CLUSTERING (Logica originale intatta) ---
    reduced = {}
    results = {}
    k_clusters = len(classes)

    for name, X in features.items():
        # Riduzione PCA
        pca_obj = PCA().fit(X)
        n_comp = np.argmax(np.cumsum(pca_obj.explained_variance_ratio_) >= expl_var) + 1
        X_pca = PCA(n_components=n_comp).fit_transform(X)
        reduced[name] = X_pca

        # KMeans
        km = KMeans(n_clusters=k_clusters, random_state=42, n_init=50).fit(X_pca)
        results[(name, "kmeans")] = {"labels": km.labels_, "model": km, **compute_metrics(X_pca, y_true, km.labels_)}

    # --- SALVATAGGIO E PLOT (Convex Hull incluso) ---
    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    metrics_df.to_excel(os.path.join(output_folder, "clustering_metrics_HPC.xlsx"))

    # Plotting Polygons (Solo per i primi 2 componenti PCA di ogni feature)
    for (name, algo), res in results.items():
        X2 = PCA(n_components=2).fit_transform(reduced[name])
        plt.figure(figsize=(8, 6))
        for c in range(k_clusters):
            pts = X2[res["labels"] == c]
            if len(pts) >= 3:
                hull = ConvexHull(pts)
                plt.fill(pts[hull.vertices, 0], pts[hull.vertices, 1], alpha=0.1)
        plt.scatter(X2[:, 0], X2[:, 1], c=res["labels"], cmap='tab10', s=10)
        plt.title(f"{name.upper()} - {algo}")
        plt.savefig(os.path.join(output_folder, f"plot_{name}_{algo}.png"))

    print(f"✅ Completato in {(time.time()-start_time)/60:.2f} minuti")

if __name__ == "__main__":
    main()
