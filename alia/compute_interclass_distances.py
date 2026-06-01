import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

def parsing():
    parser = argparse.ArgumentParser(description='Compute Inter-class Spectral Distance Dynamics and Confusion Correlation')
    parser.add_argument('--results_dir', default='results/domain_analysis_online', help='Directory with partial outputs')
    parser.add_argument('--output_dir', default='results/domain_analysis_interclass', help='Output directory')
    parser.add_argument('--conf_0', required=True, type=str, help='Path to 0 octave confusion keys')
    parser.add_argument('--conf_3', required=True, type=str, help='Path to 3 octave confusion keys')
    return parser.parse_args()

def load_confusion_counts(file_path):
    """Parses raw misclassification logs to compute pairwise error counts."""
    if not os.path.exists(file_path):
        return {}
    try:
        df = pd.read_csv(file_path)
        # Group by true and predicted labels to get raw error counts
        counts = df.groupby(['true', 'pred']).size().to_dict()
        return counts
    except Exception as e:
        print(f"   ⚠️ Warning: Could not parse confusion file {file_path}: {e}")
        return {}

def main():
    args = parsing()
    logging.basicConfig(level=logging.INFO)
    print("🌐 STARTING INTER-CLASS GEOMETRIC COHERENCE ANALYSIS")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load confusion data passed from shell macros
    conf_0_dict = load_confusion_counts(args.conf_0)
    conf_3_dict = load_confusion_counts(args.conf_3)

    # Find available class profile files (using the raw tracks dataset as fallback to list active classes)
    files = [f for f in os.listdir(args.results_dir) if f.startswith("mel_raw_tracks_") and f.endswith(".csv")]
    classes = sorted(list(set([f.replace("mel_raw_tracks_", "").replace(".csv", "") for f in files])))
    
    if not classes:
        print("❌ No class profiles found. Run the extraction loop first.")
        return
        
    print(f"   • Detected {len(classes)} active classes for matrix crossing.")
    
    # Simulated 2D Full-Resolution Spectrogram extraction from the long-format track data
    # To maintain strict interactive compatibility without modifying upstream h5 writing, 
    # we reconstruct the mean [64, Time] matrix dynamically from the gathered track metrics.
    class_matrices = {}
    for cl in classes:
        track_file = os.path.join(args.results_dir, f"mel_raw_tracks_{cl}.csv")
        df_cl = pd.read_csv(track_file)
        
        # Determine unique tracks and unique bins
        n_bins = 64
        # We un-melt the long format back into a structured matrix representation: [64, N_tracks]
        # In an interactive scenario, this represents our static spectral centroid signature
        matrix_profile = np.zeros((64, len(df_cl['track_id'].unique())))
        
        for b in range(n_bins):
            vals = df_cl[df_cl['mel_bin'] == b]['discrepancy'].values
            matrix_profile[b, :len(vals)] = vals
            
        class_matrices[cl] = matrix_profile # Shape: [64, N_samples_extracted]

    correlation_report = []

    # Pairwise cross-comparison loop
    for i, class_A in enumerate(classes):
        print(f"   🔄 Cross-evaluating class [{class_A}] against all fields...")
        
        # Matrix structures for the specific class heatmaps (64 bins x 22 classes)
        mean_matrix_2d = np.zeros((64, len(classes)))
        std_matrix_2d = np.zeros((64, len(classes)))
        global_frobenius_row = []

        for j, class_B in enumerate(classes):
            mat_A = class_matrices[class_A]
            mat_B = class_matrices[class_B]
            
            # Match sizes if micro-batch sampling variations occurred
            min_samples = min(mat_A.shape[1], mat_B.shape[1])
            slice_A = mat_A[:, :min_samples]
            slice_B = mat_B[:, :min_samples]
            
            # 🎯 INTUITION 1: Calculate discrepancy INSTANT-BY-INSTANT (Sample-by-Sample) before averaging
            absolute_delta = np.abs(slice_A - slice_B) # Shape: [64, Min_Samples]
            
            # Extract mean and standard deviation across the distribution axis (preserves 64 mel bins)
            bin_means = np.mean(absolute_delta, axis=1) # [64,]
            bin_stds = np.std(absolute_delta, axis=1)   # [64,]
            
            mean_matrix_2d[:, j] = bin_means
            std_matrix_2d[:, j] = bin_stds
            
            # 🎯 INTUITION 2: Global Centroid Distance (Frobenius matrix norm of the overall shift)
            frobenius_total = float(np.linalg.norm(slice_A - slice_B, 'fro'))
            global_frobenius_row.append(frobenius_total)
            
            # Fetch corresponding misclassification counts from raw logs
            err_0 = conf_0_dict.get((class_A, class_B), 0)
            err_3 = conf_3_dict.get((class_A, class_B), 0)
            
            if class_A != class_B:
                correlation_report.append({
                    'class_true': class_A,
                    'class_pred_error': class_B,
                    'global_centroid_distance': frobenius_total,
                    'misclassifications_0_octave': err_0,
                    'misclassifications_3_octave': err_3
                })

        # --- EXPORT CLASS SPECIFIC HEATMAP DATA ---
        # Save structural 2D CSV tables for Matplotlib loading
        df_means = pd.DataFrame(mean_matrix_2d, columns=classes)
        df_stds = pd.DataFrame(std_matrix_2d, columns=classes)
        
        # Append the global row summary as the final line of the table
        df_means.loc['GLOBAL_CENTROID_DISTANCE'] = global_frobenius_row
        df_stds.loc['GLOBAL_CENTROID_DISTANCE'] = [0.0] * len(classes) # STD for a scalar is empty
        
        df_means.to_csv(os.path.join(args.output_dir, f"interclass_mean_matrix_{class_A}.csv"), index=True)
        df_stds.to_csv(os.path.join(args.output_dir, f"interclass_std_matrix_{class_A}.csv"), index=True)

    # --- EXPORT THE PREDICTOR CORRELATION REPORT ---
    df_corr = pd.DataFrame(correlation_report)
    df_corr.to_csv(os.path.join(args.output_dir, "interclass_distance_vs_confusion_report.csv"), index=False)
    print(f"📊 INTER-CLASS ANALYSIS COMPLETE. Master report saved to: {args.output_dir}/interclass_distance_vs_confusion_report.csv")

if __name__ == "__main__":
    main()
