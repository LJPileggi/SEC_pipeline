import os
import sys
import time
import torch
import numpy as np
import pandas as pd
import librosa
import h5py
import scipy.fftpack as fftpack
import matplotlib
matplotlib.use('Agg')  # Headless mode per cluster
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, BisectingKMeans
from sklearn.metrics import (
    silhouette_score, silhouette_samples, adjusted_rand_score,
    calinski_harabasz_score, davies_bouldin_score, fowlkes_mallows_score
)

# Priorità ai moduli in /app
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

def patched_read_audio(self, audio_input, resample=True):
    if isinstance(audio_input, tuple):
        sig, sr = audio_input
        t = torch.from_numpy(sig).float() if isinstance(sig, np.ndarray) else sig.float()
        return t.squeeze().cpu(), sr
    elif torch.is_tensor(audio_input):
        return audio_input.squeeze().cpu().float(), 44100
    elif isinstance(audio_input, np.ndarray):
        return torch.from_numpy(audio_input).squeeze().float(), 44100
    data, sr = librosa.load(audio_input, sr=44100, mono=True)
    return torch.from_numpy(data).float(), sr

msclap.CLAP.read_audio = patched_read_audio
msclap.CLAPWrapper.read_audio = patched_read_audio

start_time = time.time()
def log_step(msg):
    print(f"[{time.time() - start_time:7.2f} sec] {msg}", flush=True)

# --- FEATURE EXTRACTION FUNCTIONS ---

def extract_mfcc(sig, sr, n=13):
    return np.mean(librosa.feature.mfcc(y=sig, sr=sr, n_mfcc=n).T, axis=0)

def extract_gfcc(sig, sr, n=13):
    n_filters = 40 
    f_min = 20
    f_max = sr / 2
    
    erb_min = 21.4 * np.log10(4.37e-3 * f_min + 1)
    erb_max = 21.4 * np.log10(4.37e-3 * f_max + 1)
    erb_centers = np.linspace(erb_min, erb_max, n_filters + 2)
    freq_centers = (10**(erb_centers / 21.4) - 1) / 4.37e-3
    
    stft = np.abs(librosa.stft(sig))**2
    freqs = librosa.fft_frequencies(sr=sr)
    
    filter_bank = np.zeros((n_filters, stft.shape[0]))
    for i in range(1, n_filters + 1):
        lower = freq_centers[i-1]
        center = freq_centers[i]
        upper = freq_centers[i+1]
        filter_bank[i-1] = np.maximum(0, np.minimum((freqs - lower) / (center - lower), 
                                                   (upper - freqs) / (upper - center)))
    
    raw_gfcc = np.dot(filter_bank, stft)
    compressed_gfcc = np.power(raw_gfcc + 1e-10, 0.25) 
    gfcc = fftpack.dct(compressed_gfcc, axis=0, type=2, norm='ortho')[:n]
    return np.mean(gfcc.T, axis=0)

def extract_cqcc(sig, sr, n=13):
  cqt = np.abs(librosa.cqt(sig, sr=sr))
  log_cqt = librosa.amplitude_to_db(cqt)
  cqcc = fftpack.dct(log_cqt, axis=0, type=2, norm='ortho')[:n]
  return np.mean(cqcc.T, axis=0)

def compute_metrics(X, y_true, y_pred):
    return {
        "silhouette": silhouette_score(X, y_pred),
        "rand": adjusted_rand_score(y_true, y_pred),
        "fowlkes_mallows": fowlkes_mallows_score(y_true, y_pred),
        "calinski_harabasz": calinski_harabasz_score(X, y_pred),
        "davies_bouldin": davies_bouldin_score(X, y_pred),
    }

# ==============================================================================
# ANALISI CLASS-WISE & HUNGARIAN MATCHING
# ==============================================================================

def compute_semantic_silhouette(X, y_true, classes):
    sil_samples = silhouette_samples(X, y_true)
    results = {}
    for c_idx, c_name in enumerate(classes):
        mask = (y_true == c_idx)
        c_sil = sil_samples[mask]
        results[c_name] = {
            "n_samples": int(np.sum(mask)),
            "mean_semantic_silhouette": float(np.mean(c_sil)) if len(c_sil) > 0 else np.nan,
            "std_semantic_silhouette": float(np.std(c_sil)) if len(c_sil) > 0 else np.nan
        }
    return results

def compute_contingency_and_matching(y_true, y_pred, n_classes):
    N = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < n_classes and 0 <= p < n_classes:
            N[t, p] += 1

    row_ind, col_ind = linear_sum_assignment(-N)
    cluster_to_class = {c: r for r, c in zip(row_ind, col_ind)}
    
    y_matched = np.array([cluster_to_class.get(p, -1) for p in y_pred])
    
    matched_matrix = np.zeros((n_classes, n_classes), dtype=int)
    for t, m in zip(y_true, y_matched):
        if 0 <= t < n_classes and 0 <= m < n_classes:
            matched_matrix[t, m] += 1
            
    return N, matched_matrix, y_matched

def compute_class_metrics_single_run(N, matched_matrix, classes):
    n_classes = len(classes)
    run_metrics = {}
    K = n_classes
    
    for i, c_name in enumerate(classes):
        row_n = N[i, :]
        N_i = np.sum(row_n)
        if N_i == 0:
            continue
            
        C_i = np.max(row_n) / N_i
        
        p_ij = row_n / N_i
        p_ij_nonzero = p_ij[p_ij > 0]
        H_i = -np.sum(p_ij_nonzero * np.log(p_ij_nonzero)) / np.log(K) if K > 1 else 0.0
        
        TP = matched_matrix[i, i]
        FN = N_i - TP
        FP = np.sum(matched_matrix[:, i]) - TP
        
        precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        confused_row = matched_matrix[i, :].copy()
        confused_row[i] = -1
        
        run_metrics[c_name] = {
            "n_samples": N_i,
            "dominant_cluster_concentration": C_i,
            "normalized_cluster_entropy": H_i,
            "matched_precision": precision,
            "matched_recall": recall,
            "matched_f1": f1,
            "confused_row_counts": confused_row  # vettore di conteggi di confusione verso altre classi
        }
        
    return run_metrics

def plot_and_save_heatmap(matrix, row_labels, col_labels, title, save_path, fmt="{:.2f}", cmap="viridis"):
    fig, ax = plt.subplots(figsize=(max(10, len(col_labels) * 0.45), max(8, len(row_labels) * 0.45)))
    im = ax.imshow(matrix, cmap=cmap, aspect='auto')
    
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)
    
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(row_labels, fontsize=8)
    
    ax.set_title(title, fontsize=12, fontweight='bold', pad=12)
    ax.set_ylabel("True Ground-Truth Class", fontsize=10, fontweight='bold')
    ax.set_xlabel("Assigned Cluster / Matched Class", fontsize=10, fontweight='bold')
    
    if len(row_labels) <= 25:
        thresh = (np.nanmax(matrix) + np.nanmin(matrix)) / 2.0
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                val = matrix[i, j]
                text_color = "white" if val > thresh else "black"
                ax.text(j, i, fmt.format(val), ha="center", va="center", color=text_color, fontsize=7)
                
    fig.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close(fig)

# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

def main():
    input_dir = os.getenv("INPUT_HDF5_DIR")
    output_folder = os.getenv("OUTPUT_RESULTS_PATH")
    audio_format = os.getenv("AUDIO_FORMAT", "wav")
    dataset_name = os.getenv("DATASET", "dataSEC")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not input_dir or not os.path.exists(input_dir):
        print(f"❌ ERROR: Invalid input directory: {input_dir}")
        sys.exit(1)

    log_step(f"🎸 Initializing CLAP (Offline Mode) for dataset: {dataset_name}")
    clap_model, get_audio_embeddings, _ = CLAP_initializer(device=device, use_cuda=torch.cuda.is_available())

    feats = {"mfcc": [], "gfcc": [], "cqcc": [], "clap": []}
    all_filenames, all_labels = [], []

    h5_files = sorted([f for f in os.listdir(input_dir) if f.endswith(f'_{audio_format}_dataset.h5')])
    
    if not h5_files:
        print(f"❌ ERROR: No .h5 files found in {input_dir}")
        sys.exit(1)

    classes = [f.replace(f'_{audio_format}_dataset.h5', '') for f in h5_files]
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    n_classes = len(classes)

    for h5_file in h5_files:
        label = h5_file.replace(f'_{audio_format}_dataset.h5', '')
        h5_path = os.path.join(input_dir, h5_file)
        
        manager = HDF5DatasetManager(h5_path, audio_format=audio_format)
        log_step(f"Processing class: {label}")
        
        audio_ds = manager.hf[f'audio_{audio_format}']
        meta_ds = manager.hf[f'metadata_{audio_format}']
        sr_h5 = manager.hf.attrs.get('sample_rate', 51200)
        
        for i in range(len(audio_ds)):
            sig = audio_ds[i].astype('float32')
            fname = meta_ds[i]['track_name'].decode('utf-8')
            
            sig_norm = StandardScaler().fit_transform(sig.reshape(-1, 1)).flatten()
            feats["mfcc"].append(extract_mfcc(sig_norm, sr_h5))
            feats["gfcc"].append(extract_gfcc(sig_norm, sr_h5))
            feats["cqcc"].append(extract_cqcc(sig_norm, sr_h5))

            with torch.no_grad():
                emb = get_audio_embeddings([(sig, sr_h5)])[0]
                feats["clap"].append(emb.cpu().numpy().squeeze())

            all_labels.append(class_to_idx[label])
            all_filenames.append(fname)
        
        manager.close()

    features = {name: np.array(v) for name, v in feats.items()}
    y_true = np.array(all_labels)

    matrices_dir = os.path.join(output_folder, "matrices")
    heatmaps_dir = os.path.join(output_folder, "heatmaps")
    sample_level_dir = os.path.join(output_folder, "sample_level")
    os.makedirs(matrices_dir, exist_ok=True)
    os.makedirs(heatmaps_dir, exist_ok=True)
    os.makedirs(sample_level_dir, exist_ok=True)

    # 🎯 20 Random Initializations esplicitamente tracciate
    N_RUNS = 20
    SEEDS = [42 + i for i in range(N_RUNS)]
    log_step(f"📋 Semi stocastici utilizzati per le {N_RUNS} random initializations: {SEEDS}")

    master_class_records = []
    global_results = []

    log_step(f"Inizio PCA, Silhouette Semantica e Clustering su {dataset_name} ({N_RUNS} Runs)...")

    for feat_name, X in features.items():
        log_step(f"⚡ Elaborazione Feature: {feat_name.upper()}")
        
        pca_full = PCA().fit(X)
        n_comp = np.argmax(np.cumsum(pca_full.explained_variance_ratio_) >= 0.9) + 1
        X_pca = PCA(n_components=n_comp).fit_transform(X)

        sem_sil_dict = compute_semantic_silhouette(X_pca, y_true, classes)

        for algo_name in ["kmeans", "bisecting"]:
            log_step(f"  • Clustering: {algo_name} (20 Runs con Hungarian Matching)...")
            
            runs_metrics_collector = {c: {
                "dominant_cluster_concentration": [],
                "normalized_cluster_entropy": [],
                "matched_precision": [],
                "matched_recall": [],
                "matched_f1": [],
                "confused_counts_list": []
            } for c in classes}

            # Collezionatori di matrici sulle 20 run
            contingency_mats = []
            matched_mats = []
            matched_norm_mats = []
            run_global_metrics = []

            for seed_idx, curr_seed in enumerate(SEEDS):
                if algo_name == "kmeans":
                    model = KMeans(n_clusters=n_classes, random_state=curr_seed, n_init=10, max_iter=500).fit(X_pca)
                else:
                    model = BisectingKMeans(n_clusters=n_classes, random_state=curr_seed, n_init=5, max_iter=500).fit(X_pca)

                y_pred = model.labels_
                N, matched_mat, y_matched = compute_contingency_and_matching(y_true, y_pred, n_classes)

                # 🎯 Salvataggio CSV sample-level per ciascuna run
                matched_class_names = [classes[idx] if idx >= 0 else "Unmatched" for idx in y_matched]
                true_class_names = [classes[idx] for idx in y_true]
                df_sample_level = pd.DataFrame({
                    "sample_id": all_filenames,
                    "true_class": true_class_names,
                    "cluster_id": y_pred,
                    "matched_class": matched_class_names
                })
                df_sample_level.to_csv(
                    os.path.join(sample_level_dir, f"sample_level_{dataset_name}_{feat_name}_{algo_name}_run_{seed_idx}.csv"),
                    index=False
                )

                # Metriche globali per questa run
                m_single = compute_metrics(X, y_true, y_pred)
                run_global_metrics.append(m_single)

                # Matrice di confusione normalizzata per riga (recall-based)
                m_row_sums = matched_mat.sum(axis=1, keepdims=True)
                m_norm = np.divide(matched_mat, m_row_sums, out=np.zeros_like(matched_mat, dtype=float), where=m_row_sums != 0)

                contingency_mats.append(N)
                matched_mats.append(matched_mat)
                matched_norm_mats.append(m_norm)

                run_m = compute_class_metrics_single_run(N, matched_mat, classes)
                for c_name, m_vals in run_m.items():
                    runs_metrics_collector[c_name]["dominant_cluster_concentration"].append(m_vals["dominant_cluster_concentration"])
                    runs_metrics_collector[c_name]["normalized_cluster_entropy"].append(m_vals["normalized_cluster_entropy"])
                    runs_metrics_collector[c_name]["matched_precision"].append(m_vals["matched_precision"])
                    runs_metrics_collector[c_name]["matched_recall"].append(m_vals["matched_recall"])
                    runs_metrics_collector[c_name]["matched_f1"].append(m_vals["matched_f1"])
                    runs_metrics_collector[c_name]["confused_counts_list"].append(m_vals["confused_row_counts"])

            # 🎯 AGGREGAZIONE METRICHE GLOBALI SULLE 20 RUNS
            df_rgm = pd.DataFrame(run_global_metrics)
            global_record = {
                "dataset": dataset_name,
                "feature": feat_name,
                "clustering_method": algo_name,
                "n_components": n_comp,
                "n_runs": N_RUNS,
                "silhouette_mean": df_rgm["silhouette"].mean(),
                "silhouette_std": df_rgm["silhouette"].std(),
                "rand_mean": df_rgm["rand"].mean(),
                "rand_std": df_rgm["rand"].std(),
                "fowlkes_mallows_mean": df_rgm["fowlkes_mallows"].mean(),
                "fowlkes_mallows_std": df_rgm["fowlkes_mallows"].std(),
                "calinski_harabasz_mean": df_rgm["calinski_harabasz"].mean(),
                "calinski_harabasz_std": df_rgm["calinski_harabasz"].std(),
                "davies_bouldin_mean": df_rgm["davies_bouldin"].mean(),
                "davies_bouldin_std": df_rgm["davies_bouldin"].std()
            }
            global_results.append(global_record)

            # 🎯 MATRICI MEDIE E DEVIAZIONE STANDARD SULLE 20 RUNS
            matched_mean = np.mean(matched_norm_mats, axis=0)
            matched_std = np.std(matched_norm_mats, axis=0)

            df_matched_mean = pd.DataFrame(matched_mean, index=classes, columns=classes)
            df_matched_mean.to_csv(os.path.join(matrices_dir, f"matched_confusion_mean_{dataset_name}_{feat_name}_{algo_name}.csv"))

            df_matched_std = pd.DataFrame(matched_std, index=classes, columns=classes)
            df_matched_std.to_csv(os.path.join(matrices_dir, f"matched_confusion_std_{dataset_name}_{feat_name}_{algo_name}.csv"))

            # Salva anche la media assoluta di contingency
            contingency_mean = np.mean(contingency_mats, axis=0)
            cluster_cols = [f"Cluster_{j}" for j in range(n_classes)]
            df_cont_mean = pd.DataFrame(contingency_mean, index=classes, columns=cluster_cols)
            df_cont_mean.to_csv(os.path.join(matrices_dir, f"contingency_abs_mean_{dataset_name}_{feat_name}_{algo_name}.csv"))

            # Generazione Heatmaps Medie
            plot_and_save_heatmap(
                matched_mean, classes, classes,
                f"Matched Confusion Matrix (Mean 20 Runs): {feat_name.upper()} - {algo_name.capitalize()}",
                os.path.join(heatmaps_dir, f"heatmap_matched_confusion_mean_{dataset_name}_{feat_name}_{algo_name}.png"),
                fmt="{:.2f}", cmap="magma"
            )
            plot_and_save_heatmap(
                matched_std, classes, classes,
                f"Matched Confusion Matrix (Std 20 Runs): {feat_name.upper()} - {algo_name.capitalize()}",
                os.path.join(heatmaps_dir, f"heatmap_matched_confusion_std_{dataset_name}_{feat_name}_{algo_name}.png"),
                fmt="{:.2f}", cmap="viridis"
            )

            # 🎯 AGGREGAZIONE METRICHE CLASS-WISE SULLE 20 RUNS
            for i, c_name in enumerate(classes):
                c_data = runs_metrics_collector[c_name]
                sil_info = sem_sil_dict[c_name]
                n_samples_class = sil_info["n_samples"]
                total_samples_evaluated = n_samples_class * N_RUNS

                # Somma dei conteggi di confusione verso ciascuna altra classe su tutte le 20 run
                sum_confused_counts = np.sum(np.array(c_data["confused_counts_list"]), axis=0)
                sum_confused_counts[i] = -1  # escludiamo la vera classe

                max_conf_idx = np.argmax(sum_confused_counts)
                total_conf_n = sum_confused_counts[max_conf_idx]

                if total_conf_n > 0:
                    top_other = classes[max_conf_idx]
                    mean_conf_n_per_run = total_conf_n / N_RUNS
                    # 🎯 Calcolato sull'intero campione delle 20 run:
                    main_conf_pct_overall = (total_conf_n / total_samples_evaluated) * 100.0
                else:
                    top_other = "None"
                    mean_conf_n_per_run = 0.0
                    main_conf_pct_overall = 0.0

                master_class_records.append({
                    "dataset": dataset_name,
                    "feature": feat_name,
                    "clustering_method": algo_name,
                    "class": c_name,
                    "n_samples": n_samples_class,
                    "dominant_cluster_concentration_mean": np.mean(c_data["dominant_cluster_concentration"]),
                    "dominant_cluster_concentration_std": np.std(c_data["dominant_cluster_concentration"]),
                    "normalized_cluster_entropy_mean": np.mean(c_data["normalized_cluster_entropy"]),
                    "normalized_cluster_entropy_std": np.std(c_data["normalized_cluster_entropy"]),
                    "matched_precision_mean": np.mean(c_data["matched_precision"]),
                    "matched_precision_std": np.std(c_data["matched_precision"]),
                    "matched_recall_mean": np.mean(c_data["matched_recall"]),
                    "matched_recall_std": np.std(c_data["matched_recall"]),
                    "matched_f1_mean": np.mean(c_data["matched_f1"]),
                    "matched_f1_std": np.std(c_data["matched_f1"]),
                    "main_confused_class": top_other,
                    "main_confusion_mean_n": mean_conf_n_per_run,
                    "main_confusion_percentage": main_conf_pct_overall,
                    "mean_semantic_silhouette": sil_info["mean_semantic_silhouette"],
                    "std_semantic_silhouette": sil_info["std_semantic_silhouette"]
                })

    # --- SALVATAGGIO DEI RISULTATI FINALI GLOBALI E CLASS-WISE ---
    df_global = pd.DataFrame(global_results)
    global_csv_path = os.path.join(output_folder, f"metrics_{dataset_name}_{audio_format}.csv")
    df_global.to_csv(global_csv_path, index=False)

    df_master = pd.DataFrame(master_class_records)
    master_csv_path = os.path.join(output_folder, f"class_wise_analysis_{dataset_name}_{audio_format}.csv")
    df_master.to_csv(master_csv_path, index=False)
    
    log_step(f"✅ Analisi completata! Risultati esportati in:")
    log_step(f"   • Metriche globali: {global_csv_path}")
    log_step(f"   • Metriche class-wise: {master_csv_path}")
    log_step(f"   • Matrici (mean/std): {matrices_dir}")
    log_step(f"   • Heatmaps: {heatmaps_dir}")
    log_step(f"   • Sample-level CSVs (20 runs): {sample_level_dir}")

if __name__ == "__main__":
    main()
