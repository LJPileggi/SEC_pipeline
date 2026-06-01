import os
import argparse
import numpy as np
import pandas as pd
import logging

def parsing():
    parser = argparse.ArgumentParser(description='Compare Native vs Injected Inter-class Spectral Distance Dynamics')
    parser.add_argument('--results_dir', default='results/domain_analysis_online', help='Directory with partial outputs')
    parser.add_argument('--output_dir', default='results/domain_analysis_interclass', help='Output directory')
    parser.add_argument('--conf_0', required=True, type=str, help='Path to 0 octave confusion keys')
    parser.add_argument('--conf_3', required=True, type=str, help='Path to 3 octave confusion keys')
    return parser.parse_args()

def load_confusion_counts(file_path):
    if not os.path.exists(file_path):
        return {}
    try:
        df = pd.read_csv(file_path)
        return df.groupby(['true', 'pred']).size().to_dict()
    except Exception as e:
        print(f"   ⚠️ Warning: Could not parse confusion file {file_path}: {e}")
        return {}

def main():
    args = parsing()
    logging.basicConfig(level=logging.INFO)
    print("🌐 CORREZIONE MATRICI INTERCLASSE DUALI (0 VS 3 OTTAVE)")
    os.makedirs(args.output_dir, exist_ok=True)
    
    conf_0_dict = load_confusion_counts(args.conf_0)
    conf_3_dict = load_confusion_counts(args.conf_3)

    # Rilevazione delle classi attive
    npy_files = [f for f in os.listdir(args.results_dir) if f.startswith("spectral_centroid_native_") and f.endswith(".npy")]
    classes = sorted(list(set([f.replace("spectral_centroid_native_", "").replace(".npy", "") for f in npy_files])))
    
    if not classes:
        print("❌ No spectral centroid .npy files found. Run evaluate_domain_distance_per_class.py first.")
        return
        
    print(f"   • Classi rilevate per l'incrocio incrociato: {len(classes)}")
    
    # Caricamento delle matrici temporali medie [64, Time]
    native_matrices = {}
    injected_matrices = {}
    for cl in classes:
        native_matrices[cl] = np.load(os.path.join(args.results_dir, f"spectral_centroid_native_{cl}.npy"))
        injected_matrices[cl] = np.load(os.path.join(args.results_dir, f"spectral_centroid_injected_{cl}.npy"))

    master_report = []

    # Loop cross-class definitivo
    for class_A in classes:
        print(f"   🔄 Analisi delle distanze spettrali per la classe base: [{class_A}]...")
        
        matrix_mean_0 = np.zeros((64, len(classes)))
        matrix_std_0 = np.zeros((64, len(classes)))
        
        matrix_mean_3 = np.zeros((64, len(classes)))
        matrix_std_3 = np.zeros((64, len(classes)))
        
        matrix_delta_comparison = np.zeros((64, len(classes)))
        
        frob_row_0, frob_row_3 = [], []

        for j, class_B in enumerate(classes):
            # 1. MONDO NATIVO (0 Ottave)
            nat_A = native_matrices[class_A]
            nat_B = native_matrices[class_B]
            t_nat = min(nat_A.shape[1], nat_B.shape[1])
            
            # Sottrazione dinamica istante per istante prima del collasso
            delta_nat = np.abs(nat_A[:, :t_nat] - nat_B[:, :t_nat])
            
            matrix_mean_0[:, j] = np.mean(delta_nat, axis=1)
            matrix_std_0[:, j] = np.std(delta_nat, axis=1)
            frob_0 = float(np.linalg.norm(nat_A[:, :t_nat] - nat_B[:, :t_nat], 'fro'))
            frob_row_0.append(frob_0)

            # 2. MONDO INIETTATO (3 Ottave)
            inj_A = injected_matrices[class_A]
            inj_B = injected_matrices[class_B]
            t_inj = min(inj_A.shape[1], inj_B.shape[1])
            
            # Sottrazione dinamica istante per istante nel mondo degradato
            delta_inj = np.abs(inj_A[:, :t_inj] - inj_B[:, :t_inj])
            
            matrix_mean_3[:, j] = np.mean(delta_inj, axis=1)
            matrix_std_3[:, j] = np.std(delta_inj, axis=1)
            frob_3 = float(np.linalg.norm(inj_A[:, :t_inj] - inj_B[:, :t_inj], 'fro'))
            frob_row_3.append(frob_3)
            
            # 3. CONFRONTO DELTA DI CONTRAZIONE TOPOLOGICA (Canale per Canale Mel)
            # Un valore positivo indica che le classi si sono avvicinate a causa delle ottave
            matrix_delta_comparison[:, j] = matrix_mean_0[:, j] - matrix_mean_3[:, j]
            
            err_0 = conf_0_dict.get((class_A, class_B), 0)
            err_3 = conf_3_dict.get((class_A, class_B), 0)
            
            if class_A != class_B:
                master_report.append({
                    'class_true': class_A,
                    'class_pred_error': class_B,
                    'distance_centroid_native_0': frob_0,
                    'distance_centroid_injected_3': frob_3,
                    'topological_contraction_delta': frob_0 - frob_3,
                    'misclassifications_0_octave': err_0,
                    'misclassifications_3_octave': err_3
                })

        # --- ESPORTAZIONE REPORT TABELLARI 64x22 ---
        index_labels = [f"mel_bin_{m}" for m in range(64)]
        
        def save_and_append_frob(matrix, row_data, name):
            df = pd.DataFrame(matrix, columns=classes, index=index_labels)
            df.loc['GLOBAL_CENTROID_DISTANCE'] = row_data
            df.to_csv(os.path.join(args.output_dir, f"{name}_{class_A}.csv"), index=True)

        save_and_append_frob(matrix_mean_0, frob_row_0, "interclass_NATIVE_mean_matrix")
        save_and_append_frob(matrix_std_0, [0.0]*len(classes), "interclass_NATIVE_std_matrix")
        
        save_and_append_frob(matrix_mean_3, frob_row_3, "interclass_INJECTED_mean_matrix")
        save_and_append_frob(matrix_std_3, [0.0]*len(classes), "interclass_INJECTED_std_matrix")
        
        # Matrice finale di puro confronto (La contrazione dello spazio)
        save_and_append_frob(matrix_delta_comparison, np.array(frob_row_0) - np.array(frob_row_3), "interclass_COMPARISON_delta_matrix")

    # Master report complessivo
    df_master = pd.DataFrame(master_report)
    df_master.to_csv(os.path.join(args.output_dir, "master_distance_vs_confusion_coherence.csv"), index=False)
    print(f"🎉 Correzione ultimata. I file riflettono ora le reali distanze tra i due mondi.")

if __name__ == "__main__":
    main()
