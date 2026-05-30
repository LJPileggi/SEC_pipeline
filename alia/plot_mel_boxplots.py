import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

def parsing():
    parser = argparse.ArgumentParser(description='Generate Professional Mel-Bin Boxplots using Pure Matplotlib')
    parser.add_argument('--results_dir', default='results/domain_analysis_online', help='Directory with partial raw CSVs')
    parser.add_argument('--output_dir', default='results/domain_analysis_plots', help='Directory where plots are exported')
    return parser.parse_args()

def main():
    args = parsing()
    logging.basicConfig(level=logging.INFO)
    print("📊 STARTING SPECTRAL BOXPLOT GENERATION (MATPLOTLIB NATIVE)")
    
    if not os.path.exists(args.results_dir):
        print(f"❌ Results directory not found: {args.results_dir}")
        return
        
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Reading all class-specific raw track profiles
    files = [f for f in os.listdir(args.results_dir) if f.startswith("mel_raw_tracks_") and f.endswith(".csv")]
    
    if not files:
        print("⚠️ No raw track profile files found for boxplots.")
        return
        
    for filename in files:
        file_path = os.path.join(args.results_dir, filename)
        try:
            df_class = pd.read_csv(file_path)
            if df_class.empty:
                continue
                
            current_class = df_class['class'].iloc[0]
            print(f"   ✏️ Plotting Native Boxplot for class: {current_class}...")
            
            # 🎯 Reconstruct the 64 distribution groups for Matplotlib
            # We create a list containing the distribution arrays for each of the 64 bins
            boxplot_data = []
            for b in range(64):
                bin_data = df_class[df_class['mel_bin'] == b]['discrepancy'].values
                boxplot_data.append(bin_data)
            
            # Figure Setup
            fig, ax = plt.subplots(figsize=(15, 6))
            
            # 🎯 Matplotlib Native Boxplot styling configuration
            # Customizing boxes, whiskers, caps and fliers (outliers) to match a professional template
            boxprops = dict(linestyle='-', linewidth=1.2, color='navy', facecolor='lightblue')
            whiskerprops = dict(linestyle='--', linewidth=1.0, color='gray')
            capprops = dict(linestyle='-', linewidth=1.2, color='black')
            medianprops = dict(linestyle='-', linewidth=1.5, color='firebrick') # Red median line
            flierprops = dict(marker='o', markerfacecolor='dimgray', markersize=3, linestyle='none', markeredgecolor='none')
            
            # Render the boxplot horizontally stacked along the 64 bins
            ax.boxplot(
                boxplot_data, 
                patch_artist=True, # Allows color filling
                boxprops=boxprops,
                whiskerprops=whiskerprops,
                capprops=capprops,
                medianprops=medianprops,
                flierprops=flierprops,
                manage_ticks=False # Prevent matplotlib from messing with X labels
            )
            
            # Layout & Aesthetics
            ax.set_title(f"Spectral Discrepancy Distribution across 64 Mel Bins - Class: {current_class}", fontsize=14, fontweight='bold', pad=15)
            ax.set_xlabel("Mel Frequency Bins (0 = Low Frequencies, 63 = High Frequencies)", fontsize=11, labelpad=10)
            ax.set_ylabel("Absolute Residual Magnitude ($|X_{native} - X_{injected}|$)", fontsize=11, labelpad=10)
            
            # Aligning the X axis grid precisely with the 64 bins indexation (1-based index in matplotlib plot layout)
            ax.set_xlim(0.5, 64.5)
            ax.set_xticks(range(1, 65, 2))
            ax.set_xticklabels(range(0, 64, 2), fontsize=9)
            
            ax.tick_params(axis='y', labelsize=9)
            ax.grid(True, linestyle='--', alpha=0.5, which='both')
            
            plt.tight_layout()
            
            # Exporting plot
            output_plot_path = os.path.join(args.output_dir, f"mel_boxplot_{current_class}.png")
            plt.savefig(output_plot_path, dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️ Could not generate native boxplot for file {filename}: {e}")
            continue
            
    print(f"🎉 Boxplot generation complete. Images exported to: {args.output_dir}/")

if __name__ == "__main__":
    main()
