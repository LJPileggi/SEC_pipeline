import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

def parsing():
    parser = argparse.ArgumentParser(description='Generate Professional Mel-Bin Boxplots')
    parser.add_argument('--results_dir', default='results/domain_analysis_online', help='Directory with partial raw CSVs')
    parser.add_argument('--output_dir', default='results/domain_analysis_plots', help='Directory where plots are exported')
    return parser.parse_args()

def main():
    args = parsing()
    logging.basicConfig(level=logging.INFO)
    print("📊 STARTING SPECTRAL BOXPLOT GENERATION (META-ANALYSIS 1)")
    
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
            print(f"   ✏️ Plotting Boxplot for class: {current_class}...")
            
            # Figure Setup
            plt.figure(figsize=(15, 6))
            sns.set_theme(style="whitegrid")
            
            # Drawing a professional boxplot with custom aesthetics
            ax = sns.boxplot(
                x='mel_bin', 
                y='discrepancy', 
                data=df_class,
                color='skyblue',
                fliersize=2, # Size of outlier dots
                linewidth=1.2
            )
            
            plt.title(f"Spectral Discrepancy Distribution across 64 Mel Bins - Class: {current_class}", fontsize=14, fontweight='bold', pad=15)
            plt.xlabel("Mel Frequency Bins (0 = Low Frequencies, 63 = High Frequencies)", fontsize=11, labelpad=10)
            plt.ylabel("Absolute Residual Magnitude ($|X_{native} - X_{injected}|$)", fontsize=11, labelpad=10)
            
            # Optimizing the X ticks readability
            plt.xticks(ticks=range(0, 64, 2), labels=range(0, 64, 2), rotation=0, fontsize=9)
            plt.yticks(fontsize=9)
            
            plt.tight_layout()
            
            # Exporting plot
            output_plot_path = os.path.join(args.output_dir, f"mel_boxplot_{current_class}.png")
            plt.savefig(output_plot_path, dpi=300)
            plt.close()
            
        except Exception as e:
            print(f"   ⚠️ Could not generate boxplot for file {filename}: {e}")
            continue
            
    print(f"🎉 Boxplot generation complete. Images exported to: {args.output_dir}/")

if __name__ == "__main__":
    main()
