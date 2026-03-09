import os
import h5py
import numpy as np
import pandas as pd
from pathlib import Path

def scan_embeddings(base_path):
    results = []
    base_path = Path(base_path)
    
    # Cerchiamo tutti i file combined_*.h5
    h5_files = list(base_path.glob("**/combined_*.h5"))
    print(f"🔍 Found {len(h5_files)} HDF5 files to scan...")

    for h5_path in h5_files:
        # Estraiamo i metadati dal path
        # Struttura: .../wav/3_octave/7_secs/combined_test.h5
        parts = h5_path.parts
        audio_format = parts[-4]
        n_octave = parts[-3]
        cut_secs = parts[-2]
        split = h5_path.stem.split('_')[-1]

        try:
            with h5py.File(h5_path, 'r') as hf:
                if 'embedding_dataset' not in hf:
                    continue
                
                embeddings = hf['embedding_dataset']['embeddings'][:]
                
                total_samples = embeddings.shape[0]
                # Conta quanti vettori hanno ALMENO un NaN
                nan_mask = np.isnan(embeddings).any(axis=1)
                nan_count = np.sum(nan_mask)
                
                # Percentuale di NaN sul totale dei campioni
                nan_perc = (nan_count / total_samples) * 100 if total_samples > 0 else 0
                
                results.append({
                    "Format": audio_format,
                    "Octave": n_octave,
                    "Cut_Secs": cut_secs,
                    "Split": split,
                    "Total_Samples": total_samples,
                    "Samples_with_NaN": nan_count,
                    "NaN_Percentage": nan_perc
                })
        except Exception as e:
            print(f"❌ Error reading {h5_path}: {e}")

    return pd.DataFrame(results)

if __name__ == "__main__":
    # Inserisci qui il path assoluto della tua cartella preprocessed
    # Es: /leonardo_scratch/large/userexternal/lpilegg1/dataSEC/PREPROCESSED_DATASET
    base_dir = os.getenv("PREPROCESSED_DATASET_PATH", "./")
    
    df = scan_embeddings(base_dir)
    if not df.empty:
        # Aggreghiamo per formato e configurazione
        summary = df.groupby(["Format", "Octave", "Cut_Secs"]).agg({
            "Total_Samples": "sum",
            "Samples_with_NaN": "sum"
        }).reset_index()
        summary["NaN_Global_Perc"] = (summary["Samples_with_NaN"] / summary["Total_Samples"]) * 100
        
        print("\n📊 NAN DIAGNOSTIC REPORT")
        print("="*80)
        print(summary.to_string(index=False))
        summary.to_csv(os.path.join(os.getenv("BASEDIR_PATH", "../../"), "nan_diagnostic_report.csv"), index=False)
        print("="*80)
        print("✅ Report saved to 'nan_diagnostic_report.csv'")
    else:
        print("📭 No HDF5 files found.")
