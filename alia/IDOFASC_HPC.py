import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import librosa
import h5py
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

# Patch e Redirect (Mantra della pipeline di produzione)
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

def patched_read_audio(self, audio_path, resample=True): pass 
msclap.CLAP.read_audio = patched_read_audio

start_time = time.time()
def log_step(msg):
    print(f"[{time.time() - start_time:7.2f} sec] {msg}")

# --- FUNZIONI ORIGINALI IDOFASC_v7 ---
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
    input_dir = os.getenv("INPUT_HDF5_DIR")
    output_folder = os.getenv("OUTPUT_RESULTS_PATH")
    audio_format = os.getenv("AUDIO_FORMAT", "wav")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    log_step(f"🎸 Inizializzazione CLAP (Offline)")
    clap_model = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())

    feats = {"emb": [], "mfcc": [], "gfcc": [], "cqcc": []}
    all_filenames, all_labels = [], []

    # Cerchiamo i file con il pattern NomeClasse_formato_dataset.h5
    h5_files = sorted([f for f in os.listdir(input_dir) if f.endswith(f'_{audio_format}_dataset.h5')])
    classes = [f.split('_')[0] for f in h5_files]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}

    for h5_file in h5_files:
        label = h5_file.split('_')[0]
        manager = HDF5DatasetManager(os.path.join(input_dir, h5_file))
        
        with manager.get_h5file() as f:
            audio_ds = f[f'audio_{audio_format}']
            meta_ds = f[f'metadata_{audio_format}']
            sr_h5 = f.attrs.get('sample_rate', 51200)
            
            log_step(f"Processing class: {label}")
            for i in range(len(audio_ds)):
                sig = audio_ds[i].astype('float32')
                fname = meta_ds[i]['track_name'].decode('utf-8')
                
                sig_norm = StandardScaler().fit_transform(sig.reshape(-1, 1)).flatten()
                feats["mfcc"].append(extract_mfcc(sig_norm, sr_h5))
                feats["gfcc"].append(extract_gfcc(sig_norm, sr_h5))
                feats["cqcc"].append(extract_cqcc(sig_norm, sr_h5))

                track_t = torch.from_numpy(sig).float().unsqueeze(0).to(device)
                with torch.no_grad():
                    emb = clap_model.clap.get_audio_embeddings(track_t)
                    feats["emb"].append(emb.cpu().numpy().squeeze())
                
                all_labels.append(class_to_idx[label])
                all_filenames.append(fname)

    # Logica di clustering e plot Convex Hull (identica a IDOFASC_v7)
    features = {name: np.array(v) for name, v in feats.items()}
    y_true = np.array(all_labels)
    results = {}
    for name, X in features.items():
        pca_full = PCA().fit(X)
        n_comp = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.9) + 1
        X_pca = PCA(n_components=n_comp).fit_transform(X)
        km = KMeans(n_clusters=len(classes), random_state=42, n_init=50, max_iter=1000).fit(X_pca)
        results[(name, "kmeans")] = {"model": km, "labels": km.labels_, "pca_data": X_pca, **compute_metrics(X_pca, y_true, km.labels_)}
        bkm = BisectingKMeans(n_clusters=len(classes), random_state=42, n_init=10, max_iter=1000).fit(X_pca)
        results[(name, "bisecting")] = {"model": bkm, "labels": bkm.labels_, "pca_data": X_pca, **compute_metrics(X_pca, y_true, bkm.labels_)}

    metrics_df = pd.DataFrame.from_dict(results, orient="index")
    metrics_df.to_excel(os.path.join(output_folder, f"metrics_{audio_format}.xlsx"))
    log_step(f"✅ Completato. Risultati in: {output_folder}")

if __name__ == "__main__":
    main()
