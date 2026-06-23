import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, '/app')

# --- NON-LINEAR METRICS FOR GLOBAL SPEC DISCREPANCY ---
def compute_agnostic_mmd(x, y, alphas=[0.1, 1.0, 10.0]):
    """
    Compute Maximum Mean Discrepancy (MMD) between two matrices using a multi-scale RBF kernel.
    """
    import torch
    if not isinstance(x, torch.Tensor):
        x = torch.from_numpy(x).float()
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(y).float()
        
    if x.ndim > 2:
        x = x.squeeze()
    if y.ndim > 2:
        y = y.squeeze()
        
    x_size = x.size(0)
    y_size = y.size(0)
    
    xx = torch.pow(torch.norm(x, dim=1, keepdim=True), 2)
    yy = torch.pow(torch.norm(y, dim=1, keepdim=True), 2)
    
    dist_xx = xx + xx.t() - 2 * torch.mm(x, x.t())
    dist_yy = yy + yy.t() - 2 * torch.mm(y, y.t())
    dist_xy = xx + yy.t() - 2 * torch.mm(x, y.t())
    
    kernel_xx, kernel_yy, kernel_xy = 0.0, 0.0, 0.0
    for alpha in alphas:
        kernel_xx += torch.exp(-dist_xx / (2 * alpha))
        kernel_yy += torch.exp(-dist_yy / (2 * alpha))
        kernel_xy += torch.exp(-dist_xy / (2 * alpha))
        
    mmd = (kernel_xx.sum() / (x_size * x_size) + 
           kernel_yy.sum() / (y_size * y_size) - 
           2 * kernel_xy.sum() / (x_size * y_size))
           
    return torch.clamp(mmd, min=0.0).item()

def compute_agnostic_wasserstein(p_mat, q_mat, epsilon=0.01, max_iter=100):
    """
    Compute 2D Wasserstein distance using the Sinkhorn algorithm with entropic regularization.
    """
    import torch
    import torch.nn.functional as F
    if not isinstance(p_mat, torch.Tensor):
        p_mat = torch.from_numpy(p_mat).float()
    if not isinstance(q_mat, torch.Tensor):
        q_mat = torch.from_numpy(q_mat).float()
        
    if p_mat.ndim > 2:
        p_mat = p_mat.squeeze()
    if q_mat.ndim > 2:
        q_mat = q_mat.squeeze()

    a = F.softmax(p_mat.flatten(), dim=0).unsqueeze(1) 
    b = F.softmax(q_mat.flatten(), dim=0).unsqueeze(1) 
    
    dim = a.size(0)
    grid = torch.arange(dim, dtype=torch.float32).unsqueeze(1)
    C = torch.pow(grid - grid.t(), 2)
    C = C / C.max() 
    
    K = torch.exp(-C / epsilon)
    u = torch.ones((dim, 1), dtype=torch.float32) / dim
    
    for _ in range(max_iter):
        v = b / (torch.mm(K.t(), u) + 1e-8)
        u = a / (torch.mm(K, v) + 1e-8)
        
    transport_plan = u * K * v.t()
    return torch.sum(transport_plan * C).item()

def parsing():
    parser = argparse.ArgumentParser(description='Aggregate Tranched Domain Distance Results')
    parser.add_argument('--results_dir', default='results/domain_analysis_online', help='Directory containing partial CSVs')
    parser.add_argument('--output_dir', default='results/domain_analysis_final', help='Directory for consolidated reports')
    return parser.parse_args()

def main():
    args = parsing()
    logging.basicConfig(level=logging.INFO)
    print("📊 STARTING GLOBAL DOMAIN DISTANCE AGGREGATION")
    print(f"   • Scanning directory: {args.results_dir}")
    
    if not os.path.exists(args.results_dir):
        raise FileNotFoundError(f"Results directory not found: {args.results_dir}")
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Gathering all partial class-specific audio distance CSV files
    all_files = [f for f in os.listdir(args.results_dir) if f.startswith("online_per_audio_distances_") and f.endswith(".csv")]
    
    if not all_files:
        print("⚠️ No partial tranched CSV files found. Ensure the Bash loop ran successfully.")
        return
        
    print(f"   • Found {len(all_files)} partial class files to aggregate.")
    
    list_dfs = []
    for filename in all_files:
        file_path = os.path.join(args.results_dir, filename)
        try:
            temp_df = pd.read_csv(file_path)
            if not temp_df.empty:
                list_dfs.append(temp_df)
        except Exception as e:
            print(f"   ⚠️ Warning: Could not read {filename}: {e}")
            continue
            
    if not list_dfs:
        print("❌ Error: All found partial CSVs are empty or corrupted.")
        return
        
    # 1. Consolidating the Master DataFrame (Granular track-by-track evaluation)
    master_df = pd.concat(list_dfs, ignore_index=True)
    
    # Enforcing column structure alignment based on evaluate_domain_distance_per_class.py keys
    required_cols = ['track_id', 'class', 'frobenius', 'kl_divergence', 'wasserstein']
    for col in required_cols:
        if col not in master_df.columns:
            raise KeyError(f"Expected column '{col}' missing from aggregated data framework.")
            
    # Saving full consolidated track-by-track data
    master_output_path = os.path.join(args.output_dir, "consolidated_track_distances.csv")
    master_df.to_csv(master_output_path, index=False)
    
    # 2. EVALUATION LEVEL A: Global Mean Distance across the entire dataset
    print("\n" + "="*60)
    print(" 📈 GLOBAL DISTRIBUTION ACCUMULATED DISTANCES (METRICS OVERVIEW)")
    print(f"   • Total Tracks Analyzed:   {len(master_df)}")
    print(f"   • Frobenius Distance Mean: {master_df['frobenius'].mean():.4f} (± {master_df['frobenius'].std():.4f})")
    print(f"   • KL Divergence Mean:      {master_df['kl_divergence'].mean():.4f} (± {master_df['kl_divergence'].std():.4f})")
    print(f"   • Wasserstein Distance Mean: {master_df['wasserstein'].mean():.4f} (± {master_df['wasserstein'].std():.4f})")
    print("="*60)
    
    # 3. EVALUATION LEVEL B: Disaggregated Class-Level Evaluation
    # Grouping by class and computing descriptive distribution averages
    class_summary_df = master_df.groupby('class')[['frobenius', 'kl_divergence', 'wasserstein']].agg([
        ('mean', 'mean'),
        ('std', 'std')
    ]).reset_index()
    
    # Flattening MultiIndex columns resulting from the double aggregation (.agg)
    class_summary_df.columns = [
        'class', 
        'frobenius_mean', 'frobenius_std', 
        'kl_mean', 'kl_std', 
        'wasserstein_mean', 'wasserstein_std'
    ]
    
    # Sorting from largest domain discrepancy to smallest based on Wasserstein metric
    class_summary_df = class_summary_df.sort_values(by='wasserstein_mean', ascending=False)
    
    # Saving class summary breakdown report
    class_output_path = os.path.join(args.output_dir, "final_class_distance_breakdown.csv")
    class_summary_df.to_csv(class_output_path, index=False)
    
    print("\n📊 TOP 5 DOMAINS WITH HIGHEST DISCREPANCY (CRITICAL Adaptation Targets):")
    print(class_summary_df[['class', 'wasserstein_mean', 'kl_mean', 'frobenius_mean']].head(5).to_string(index=False))
    
    print("\n📉 TOP 5 DOMAINS WITH LOWEST DISCREPANCY (STABLE Targets):")
    print(class_summary_df[['class', 'wasserstein_mean', 'kl_mean', 'frobenius_mean']].tail(5).to_string(index=False))
    
    # 4. EVALUATION LEVEL C: Global Non-linear Metrics via Accumulated Class Centroids
    print("\n🌐 COMPLIANCE VERIFICATION VIA SPECTRAL CENTROIDS (NON-LINEAR METRICS)")
    native_files = [f for f in os.listdir(args.results_dir) if f.startswith("spectral_centroid_native_") and f.endswith(".npy")]
    injected_files = [f for f in os.listdir(args.results_dir) if f.startswith("spectral_centroid_injected_") and f.endswith(".npy")]
    
    native_centroids = []
    injected_centroids = []
    
    for f in native_files:
        try:
            native_centroids.append(np.load(os.path.join(args.results_dir, f)))
        except Exception:
            continue
            
    for f in injected_files:
        try:
            injected_centroids.append(np.load(os.path.join(args.results_dir, f)))
        except Exception:
            continue
            
    if native_centroids and injected_centroids:
        # Linear aggregation of spectral matrices across distributed chunks
        global_native_centroid = np.mean(native_centroids, axis=0)
        global_injected_centroid = np.mean(injected_centroids, axis=0)
        
        # MMD expects [Samples, Features], transpose time axis to treat frames as samples [Time, 64]
        global_mmd = compute_agnostic_mmd(global_native_centroid.T, global_injected_centroid.T)
        global_wasserstein_2d = compute_agnostic_wasserstein(global_native_centroid, global_injected_centroid)
        
        print(f"   • GLOBAL DATASET SHIFT - Maximum Mean Discrepancy (MMD): {global_mmd:.6f}")
        print(f"   • GLOBAL DATASET SHIFT - 2D Optimal Transport (Wasserstein): {global_wasserstein_2d:.6f}")
        
        # 🎯 1. CALCOLO DELLA SOGLIA DI VARIANZA INTRINSECA (BASELINE H0)
        # Dividiamo il centroide nativo a metà lungo l'asse temporale per simulare due sotto-porzioni dello stesso dominio
        time_steps = global_native_centroid.shape[1]
        half_time = time_steps // 2
        native_part_A = global_native_centroid[:, :half_time]
        native_part_B = global_native_centroid[:, half_time:(half_time * 2)]
        
        # Calcolo del rumore di fondo statistico della metrica
        h0_wasserstein = compute_agnostic_wasserstein(native_part_A, native_part_B)
        print(f"   • METRIC BACKGROUND NOISE (H0 Baseline): {h0_wasserstein:.6f}")

        # 🎯 2.ESTRAZIONE AUTOMATICA SOGLIA INTERCLASSE NATIVA (SEPARABILITÀ)
        # Cerchiamo i file generati da compute_interclass_distances.py per estrarre la distanza minima tra classi native
        interclass_dir = os.path.join(os.path.dirname(args.output_dir), "domain_analysis_interclass")
        threshold_separability = None
        
        if os.path.exists(interclass_dir):
            interclass_files = [f for f in os.listdir(interclass_dir) if f.startswith("interclass_NATIVE_mean_matrix_")]
            frob_values = []
            for f in interclass_files:
                try:
                    ic_df = pd.read_csv(os.path.join(interclass_dir, f), index_col=0)
                    # Estraiamo i valori della riga di riepilogo globale escludendo gli zeri sulla diagonale della stessa classe
                    global_row = ic_df.loc['GLOBAL_CENTROID_DISTANCE']
                    valid_distances = [float(v) for v in global_row.values if float(v) > 1e-5]
                    if valid_distances:
                        frob_values.append(min(valid_distances))
                except Exception:
                    continue
            if frob_values:
                # La soglia è definita dal margine di separazione minimo riscontrato tra due classi sani
                threshold_separability = min(frob_values)
                print(f"   • NATIVE INTERCLASS SEPARABILITY THRESHOLD: {threshold_separability:.6f}")

        # 🎯 3. VERIFICA MATEMATICA DEI BOUND E REGIME DI ADATTAMENTO
        print("\n⚖️ DOMAIN SHIFT REASONING OVER BEN-DAVID THEOREMS:")
        if global_wasserstein_2d <= h0_wasserstein:
            print("   👉 STATUS: NEGligible ShIFT. Target domain matches Source distribution statistics.")
            print("      Action: Standard training or simple linear alignment is sufficient.")
        elif threshold_separability and global_wasserstein_2d >= threshold_separability:
            print("   🚨 STATUS: THEORETICAL IMPOSSIBILITY TRIGGERED.")
            print("      Reason: Tool-induced shift exceeds the geometric separation between native classes.")
            print("      Effect: Joint hypothesis error (lambda) explodes due to density overlap.")
            print("      Action: Classical Domain Adaptation is ineffective. Generative Feature Restoration (Diffusion) is required.")
        else:
            print("   ⚠️ STATUS: MODERATE NON-LINEAR SHIFT.")
            print("      Reason: Shift is above statistical noise but within class decision boundaries.")
            print("      Action: Deep representation learning or conditional adaptation recommended.")

        # Esportazione finale del report scalare consolidato
        summary_scalar_path = os.path.join(args.output_dir, "global_non_linear_distances.csv")
        pd.DataFrame([
            {'metric': 'MMD_global_centroids', 'value': global_mmd},
            {'metric': 'Wasserstein_2D_global_centroids', 'value': global_wasserstein_2d},
            {'metric': 'H0_Wasserstein_baseline', 'value': h0_wasserstein},
            {'metric': 'Interclass_Separability_Threshold', 'value': threshold_separability if threshold_separability else -1.0}
        ]).to_csv(summary_scalar_path, index=False)
        print(f"\n   • Global non-linear report exported to: {summary_scalar_path}")
    else:
        print("   ⚠️ Warning: Spectral centroid npy files not found or mismatch occurred. Skipping global non-linear computation.")

if __name__ == "__main__":
    main()
