#%% Imports
import os
import time
import numpy as np
import pandas as pd
import librosa
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torchaudio

# Forza torchaudio a non usare torchcodec se non presente
# e a ripiegare sui backend standard (ffmpeg/sox)
try:
    torchaudio.set_audio_backend("ffmpeg") # o "sox" se preferisci
except:
    pass

from collections import Counter
# from tkinter import Tk, filedialog
from scipy.spatial import ConvexHull

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import (
    silhouette_score,
    rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    fowlkes_mallows_score
)

from msclap import CLAP

#%% Timing
start_time = time.time()

def log_step(msg):
    elapsed = time.time() - start_time
    print(f"[{elapsed:7.2f} sec] {msg}")

#%% Select dataset
# Tk().withdraw()
# dataset_path = filedialog.askdirectory(title="Select dataset folder")
# output_folder = filedialog.askdirectory(title="Select output folder")

# Legge i percorsi dalle variabili d'ambiente impostate dallo script shell
dataset_path = os.getenv("INPUT_DATASET_PATH")
output_folder = os.getenv("OUTPUT_RESULTS_PATH")

if not dataset_path or not output_folder:
    raise ValueError("INPUT_DATASET_PATH or OUTPUT_RESULTS_PATH not set in environment.")

print(f"Dataset path: {dataset_path}")
print(f"Output folder: {output_folder}")

if not os.path.exists(output_folder):
    os.makedirs(output_folder, exist_ok=True)

#%% Load audio files
classes = [
    d for d in sorted(os.listdir(dataset_path))
    if os.path.isdir(os.path.join(dataset_path, d)) and not d.startswith(".")
]

if not classes:
    classes = ["__root__"]

class_to_index = {cls: idx for idx, cls in enumerate(classes)}

audio_exts = (".wav", ".flac", ".mp3", ".ogg", ".m4a", ".aiff", ".aif")
file_paths, labels = [], []

for cls, idx in class_to_index.items():
    cls_path = os.path.join(dataset_path, cls)
    for root, _, files in os.walk(cls_path):
        for f in files:
            if f.lower().endswith(audio_exts):
                file_paths.append(os.path.join(root, f))
                labels.append(idx)

counts = Counter([classes[i] for i in labels])
print("Loaded files per class:")
for cls in classes:
    print(f"  {cls}: {counts.get(cls, 0)}")

#%% Parameters
expl_var = 0.9
sr = 22050
k = len(classes)
clap_model = CLAP(version="2023", use_cuda=False)

#%% Feature extraction
def extract_embeddings(fp):
    return clap_model.get_audio_embeddings([fp]).squeeze().numpy()

def extract_mfcc(sig, sr, n=13):
    return np.mean(librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n).T, axis=0)

def extract_gfcc(sig, sr, n=13):
    return np.mean(librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n).T, axis=0)

def extract_cqcc(sig, sr, n=13):
    cq = librosa.feature.chroma_cqt(y=sig, sr=sr).T
    return np.mean(cq, axis=0)[:n]

#%% Preprocessing
def preprocess(paths, labels, class_names):
    feats = {"emb": [], "mfcc": [], "gfcc": [], "cqcc": []}
    last_cls = None

    for fp, lbl in zip(paths, labels):
        cls = class_names[lbl]
        if cls != last_cls:
            log_step(f"Processing class: {cls}")
            last_cls = cls

        sig, sr_ = librosa.load(fp, sr=None, mono=True)
        sig = StandardScaler().fit_transform(sig.reshape(-1, 1)).flatten()

        feats["emb"].append(extract_embeddings(fp))
        feats["mfcc"].append(extract_mfcc(sig, sr_))
        feats["gfcc"].append(extract_gfcc(sig, sr_))
        feats["cqcc"].append(extract_cqcc(sig, sr_))

    return {k: np.array(v) for k, v in feats.items()}

log_step("Starting preprocessing")
features = preprocess(file_paths, labels, classes)
log_step("Finished preprocessing")

#%% PCA
def reduce_pca(X, thr=expl_var):
    pca = PCA().fit(X)
    n = np.argmax(np.cumsum(pca.explained_variance_ratio_) >= thr) + 1
    return PCA(n_components=n).fit_transform(X), n

reduced = {}
for name, X in features.items():
    reduced[name], n = reduce_pca(X)
    log_step(f"{name.upper()} PCA → {n} components")

#%% Metrics
def compute_metrics(X, y_true, y_pred):
    return {
        "silhouette": silhouette_score(X, y_pred),
        "rand": rand_score(y_true, y_pred),
        "fowlkes_mallows": fowlkes_mallows_score(y_true, y_pred),
        "calinski_harabasz": calinski_harabasz_score(X, y_pred),
        "davies_bouldin": davies_bouldin_score(X, y_pred),
    }

#%% KMeans
def kmeans_fixed(X):
    model = KMeans(n_clusters=k, random_state=42, n_init=50, max_iter=1000)
    y = model.fit_predict(X)
    return model, y

#%% Bisecting KMeans
def bisecting_kmeans_fixed(X):
    model = BisectingKMeans(
        n_clusters=k,
        random_state=42,
        n_init=10,
        max_iter=1000,
        bisecting_strategy="biggest_inertia"
    )
    y = model.fit_predict(X)
    return model, y

#%% Clustering
results = {}

for name, X in reduced.items():
    log_step(f"Clustering {name.upper()}")

    km_model, y_km = kmeans_fixed(X)
    results[(name, "kmeans")] = {
        "model": km_model,
        "labels": y_km,
        "k": k,
        **compute_metrics(X, labels, y_km)
    }

    bkm_model, y_bkm = bisecting_kmeans_fixed(X)
    results[(name, "bisecting")] = {
        "model": bkm_model,
        "labels": y_bkm,
        "k": k,
        **compute_metrics(X, labels, y_bkm)
    }

#%% Save metrics
metrics_df = pd.DataFrame.from_dict(results, orient="index")
metrics_df.index = pd.MultiIndex.from_tuples(
    metrics_df.index, names=["feature", "algorithm"]
)

print("\nClustering Metrics:\n", metrics_df)

if output_folder:
    metrics_df.to_excel(os.path.join(output_folder, "clustering_metrics_all.xlsx"))
    log_step("Metrics saved")

#%% Save CLAP embeddings
if output_folder:
    emb_df = pd.DataFrame(features["emb"])
    emb_df.insert(0, "filename", [os.path.basename(fp) for fp in file_paths])
    emb_df.insert(1, "label", [classes[i] for i in labels])
    emb_df.to_excel(os.path.join(output_folder, "CLAP_embeddings.xlsx"), index=False)
    log_step("CLAP embeddings saved")

#%% Colors
def get_colors(n):
    cmap = cm.get_cmap("tab20" if n <= 20 else "hsv", n)
    return [cmap(i) for i in range(n)]

color_map = {i: c for i, c in enumerate(get_colors(len(classes)))}

#%% Polygon plot
def plot_polygon_clusters(X, labels, centroids, title, fname):
    plt.figure(figsize=(8, 6))

    for c in sorted(set(labels)):
        pts = X[labels == c]
        col = color_map[c % len(color_map)]

        if len(pts) >= 3:
            hull = ConvexHull(pts)
            hp = pts[hull.vertices]
            plt.plot(
                np.r_[hp[:, 0], hp[0, 0]],
                np.r_[hp[:, 1], hp[0, 1]],
                "--", color=col, lw=2
            )
            plt.fill(hp[:, 0], hp[:, 1], color=col, alpha=0.1)

        plt.plot(np.mean(pts[:, 0]), np.mean(pts[:, 1]), "X", c="black")

    plt.scatter(centroids[:, 0], centroids[:, 1], c="red", s=80, marker="X")
    plt.title(title)
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()

    if output_folder:
        plt.savefig(os.path.join(output_folder, fname), dpi=600, bbox_inches="tight")

    plt.show()

#%% Polygon plots
for (name, algo), res in results.items():
    X2 = PCA(n_components=2).fit_transform(reduced[name])
    plot_polygon_clusters(
        X2,
        res["labels"],
        res["model"].cluster_centers_[:, :2],
        f"{name.upper()} – {algo.capitalize()} Polygon Clusters",
        f"{name}_{algo}_polygon.png"
    )

#%% Done
total = time.time() - start_time
print(f"\n✅ Completed in {total/60:.2f} minutes ({total:.1f} seconds)")
