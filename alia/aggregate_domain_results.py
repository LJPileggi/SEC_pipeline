import os
import sys
import argparse
import pandas as pd
import numpy as np
import logging

sys.path.insert(0, '/app')

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
    
    print(f"\n✅ Consolidation complete. Final outputs exported to: {args.output_dir}/")

if __name__ == "__main__":
    main()
