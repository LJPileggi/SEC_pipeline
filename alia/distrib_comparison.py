import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json
import torch

sys.path.insert(0, '/app')

# Import dal tuo src.utils (Ground Truth)
from src.utils import HDF5EmbeddingDatasetsManager

def kl_divergence_gaussians(mu_p, sigma_p, mu_q, sigma_q):
    """
    Versione robusta della KL Divergence per alta dimensionalità.
    Usa slogdet per evitare l'underflow del determinante.
    """
    k = mu_p.shape[0]
    
    # Aumentiamo leggermente la regolarizzazione basandoci sulla scala dei dati
    # Se gli embedding sono normalizzati, 1e-5 è un buon compromesso
    eps = 1e-5 * np.eye(k)
    sigma_p_reg = sigma_p + eps
    sigma_q_reg = sigma_q + eps
    
    try:
        # Inversa robusta
        inv_sigma_q = np.linalg.inv(sigma_q_reg)
        
        # Calcolo del log-determinante stabile: det(Sigma) = exp(slogdet)
        # ln(det_q / det_p) = ln(det_q) - ln(det_p)
        sign_p, logdet_p = np.linalg.slogdet(sigma_p_reg)
        sign_q, logdet_q = np.linalg.slogdet(sigma_q_reg)
        
        if sign_p <= 0 or sign_q <= 0:
            print("Warning: Matrice di covarianza non definita positiva.")
            return np.nan

        term1 = np.trace(inv_sigma_q @ sigma_p_reg)
        term2 = (mu_q - mu_p).T @ inv_sigma_q @ (mu_q - mu_p)
        term3 = -k
        term4 = logdet_q - logdet_p
        
        return 0.5 * (term1 + term2 + term3 + term4)
    
    except np.linalg.LinAlgError:
        print("Error: Matrice di covarianza singolare anche dopo regolarizzazione.")
        return np.nan

def main(args):
    # 1. Caricamento Embeddings tramite il tuo Manager
    print(f"Loading Audio embeddings: {args.audio_h5}")
    manager_a = HDF5EmbeddingDatasetsManager(args.audio_h5, mode='r')
    emb_a = manager_a.hf['embedding_dataset']['embeddings'][:]
    # FIX: Decodifica bytes e gestione stringhe per le label
    raw_lab_a = [l.decode('utf-8') if isinstance(l, bytes) else l for l in manager_a.hf['embedding_dataset']['classes'][:]]
    
    print(f"Loading Octave embeddings: {args.octave_h5}")
    manager_o = HDF5EmbeddingDatasetsManager(args.octave_h5, mode='r')
    emb_o = manager_o.hf['embedding_dataset']['embeddings'][:]
    raw_lab_o = [l.decode('utf-8') if isinstance(l, bytes) else l for l in manager_o.hf['embedding_dataset']['classes'][:]]

    # Mappatura etichette testuali in indici numerici per il plot
    unique_labels = sorted(list(set(raw_lab_a + raw_lab_o)))
    label_to_id = {label: i for i, label in enumerate(unique_labels)}
    num_lab_a = np.array([label_to_id[l] for l in raw_lab_a])
    num_lab_o = np.array([label_to_id[l] for l in raw_lab_o])

    # 2. Riduzione dimensionalità (t-SNE) per confronto spaziale
    print("Running t-SNE reduction...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    
    # Calcoliamo le proiezioni
    proj_a = tsne.fit_transform(emb_a)
    proj_o = tsne.fit_transform(emb_o)
    
    # Plotting side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Pannello Audio
    scatter1 = axes[0].scatter(proj_a[:, 0], proj_a[:, 1], c=num_lab_a, cmap='tab10', alpha=0.6, s=15)
    axes[0].set_title("Audio Domain Embeddings (Source)")
    
    # Pannello Ottave
    scatter2 = axes[1].scatter(proj_o[:, 0], proj_o[:, 1], c=num_lab_o, cmap='tab10', alpha=0.6, s=15)
    axes[1].set_title("Octave Domain Embeddings (Target)")

    # Legenda con i nomi reali delle classi
    handles, _ = scatter2.legend_elements()
    axes[1].legend(handles, unique_labels, loc="best", title="Classes", markerscale=1.2)
    
    plot_path = os.path.join(args.output_dir, "tsne_comparison.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {plot_path}")
    
    # 3. Confronto statistico (KL Divergence)
    print("Computing KL Divergence...")
    mu_a, sigma_a = np.mean(emb_a, axis=0), np.cov(emb_a, rowvar=False)
    mu_o, sigma_o = np.mean(emb_o, axis=0), np.cov(emb_o, rowvar=False)
    
    kl_val = kl_divergence_gaussians(mu_a, sigma_a, mu_o, sigma_o)
    
    # Salvataggio statistiche in JSON
    stats = {
        "kl_divergence": float(kl_val),
        "audio_samples": int(emb_a.shape[0]),
        "octave_samples": int(emb_o.shape[0]),
        "dim": int(emb_a.shape[1])
    }
    
    stats_path = os.path.join(args.output_dir, "comparison_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=4)
    
    print(f"KL Divergence: {kl_val:.4f}")
    manager_a.close()
    manager_o.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Embedding Distribution Comparison')
    parser.add_argument("--audio_h5", type=str, required=True, help="Path to source H5")
    parser.add_argument("--octave_h5", type=str, required=True, help="Path to target H5")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for plots and stats")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
