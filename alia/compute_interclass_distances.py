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
    print("🌐 STARTING DUAL-DOMAIN (NATIVE VS INJECTED) INTER-CLASS EXPERIMENT")
    os.makedirs(args.output_dir, exist_ok=True)
    
    conf_0_dict = load_confusion_counts(args.conf_0)
    conf_3_dict = load_confusion_counts(args.conf_3)

    # Rilevazione delle classi attive basandoci sui file nativi generati
    npy_files = [f for f in os.listdir(args.results_dir) if f.startswith("spectral_centroid_native_") and f.endswith(".npy")]
    classes = sorted(list(set([f.replace("spectral_centroid_native_", "").replace(".npy", "") for f in npy_files])))
    
    if not classes:
        print("❌ No spectral centroid .npy files found. Run the extraction loop first.")
        return
        
    print(f"   • Detected {len(classes)} active classes for dual-matrix crossing.")
    
    # Caricamento parallelo dei due mondi in RAM [64, Time]
    native_matrices = {}
    injected_matrices = {}
    for cl in classes:
        native_matrices[cl] = np.load(os.path.join(args.results_dir, f"spectral_centroid_native_{cl}.npy"))
        injected_matrices[cl] = np.load(os.path.join(args.results_dir, f"spectral_centroid_injected_{cl}.npy"))

    master_report = []

    # Loop cross-class di confronto
    for class_A in classes:
        print(f"   🔄 Cross-evaluating dual spectral distances for class: [{class_A}]...")
        
        # Allocazione delle matrici finali 64 canali Mel x 22 Classi Concorrenti
        # Calcoleremo Medie e Deviazioni Standard temporali per ENTRAMBE le configurazioni
        matrix_mean_0 = np.zeros((64, len(classes)))
        matrix_std_0 = np.zeros((64, len(classes)))
        
        matrix_mean_3 = np.zeros((64, len(classes)))
        matrix_std_3 = np.zeros((64, len(classes)))
        
        # Questa sarà la matrice di CONFRONTO DIRETTO (Contrazione Topologica)
        matrix_delta_comparison = np.zeros((64, len(classes)))
        
        frob_row_0, frob_row_3 = [], []

        for j, class_B in enumerate(classes):
            # --- 1. COMPUTAZIONE MONDO NATIVO (0 OTTAVE) ---
            nat_A, nat_B = native_matrices[class_A], native_matrices[class_B]
            t_nat = min(nat_A.shape[1], nat_B.shape[1])
            delta_nat = np.abs(nat_A[:, :t_nat] - nat_B[:, :t_nat]) # Distanza istante per istante
            
            mean_0 = np.mean(delta_nat, axis=1)
            std_0 = np.std(delta_nat, axis=1)
            frob_0 = float(np.linalg.norm(nat_A[:, :t_nat] - nat_B[:, :t_nat], 'fro'))
            
            matrix_mean_0[:, j] = mean_0
            matrix_std_0[:, j] = std_0
            frob_row_0.append(frob_0)

            # --- 2. COMPUTAZIONE MONDO INIETTATO (3 OTTAVE) ---
            inj_A, inj_B = injected_matrices[class_A], injected_matrices[class_B]
            t_inj = min(inj_A.shape[1], inj_B.shape[1])
            delta_inj = np.abs(inj_A[:, :t_inj] - inj_B[:, :t_inj]) # Distanza istante per istante
            
            mean_3 = np.mean(delta_inj, axis=1)
            std_3 = np.std(delta_inj, axis=1)
            frob_3 = float(np.linalg.norm(inj_A[:, :t_inj] - inj_B[:, :t_inj], 'fro'))
            
            matrix_mean_3[:, j] = mean_3
            matrix_std_3[:, j] = std_3
            frob_row_3.append(frob_3)
            
            # --- 3. CONFRONTO DIRETTO DEL CONTESTO (La Sottrazione delle Distanze Medie) ---
            # Un valore positivo indica che le due classi si sono AVVICINATE (contrazione dello spazio)
            matrix_delta_comparison[:, j] = mean_0 - mean_3
            
            # Popoliamo il master report di correlazione per gli scatter plot della tesi
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

        # --- EXPORT REPORT TABELLARI 64x22 PER LA CLASSE ATTIVA ---
        index_labels = [f"mel_bin_{m}" for m in range(64)]
        
        # Funzione helper interna per formattare e appendere la riga macro della distanza globale dei centroidi
        def save_and_append_frob(matrix, row_data, name):
            df = pd.DataFrame(matrix, columns=classes, index=index_labels)
            df.loc['GLOBAL_CENTROID_DISTANCE'] = row_data
            df.to_csv(os.path.join(args.output_dir, f"{name}_{class_A}.csv"), index=True)

        save_and_append_frob(matrix_mean_0, frob_row_0, "interclass_NATIVE_mean_matrix")
        save_and_append_frob(matrix_std_0, [0.0]*len(classes), "interclass_NATIVE_std_matrix")
        
        save_and_append_frob(matrix_mean_3, frob_row_3, "interclass_INJECTED_mean_matrix")
        save_and_append_frob(matrix_std_3, [0.0]*len(classes), "interclass_INJECTED_std_matrix")
        
        # Esportazione della matrice di confronto diretto (Delta di contrazione)
        save_and_append_frob(matrix_delta_comparison, np.array(frob_row_0) - np.array(frob_row_3), "interclass_COMPARISON_delta_matrix")

    # --- EXPORT MASTER REPORT DI CORRELAZIONE GENERALE ---
    df_master = pd.DataFrame(master_report)
    df_master.to_csv(os.path.join(args.output_dir, "master_distance_vs_confusion_coherence.csv"), index=False)
    print(f"📊 DUAL-DOMAIN INTERCLASSE COMPLETATO CON SUCCESSO.")
    print(f"   • Generati i report NATIVE, INJECTED e il CONFRONTO DELTA in: {args.output_dir}/")
    print(f"   • Master report di coerenza esportato: {args.output_dir}/master_distance_vs_confusion_coherence.csv")

if __name__ == "__main__":
    main()
