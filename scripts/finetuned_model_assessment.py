import os, sys, argparse, torch, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

# Inserimento prioritario del percorso app per i moduli src
sys.path.insert(0, '/app')
from src.utils import HDF5EmbeddingDatasetsManager, get_config_from_yaml #
from src.models import FinetunedModel # Corretto nome classe

def plot_cm(cm, classes, path, title):
    plt.figure(figsize=(14, 12))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=classes, yticklabels=classes)
    plt.title(title, fontsize=15)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_root', required=True)
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--results_base', required=True)
    parser.add_argument('--config_path', required=True)
    parser.add_argument('--batch_list', nargs='+', required=True)
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Recupero classi dal config YAML tramite utils
    # get_config_from_yaml restituisce (classes, patience, epochs, ...)
    config_data = get_config_from_yaml(args.config_path)
    classes = config_data[0] 
    
    # Inizializzazione modello corretto
    model = FinetunedModel(classes, device=device)
    # Caricamento pesi (FinetunedModel gestisce il caricamento internamente o via load_state_dict)
    state_dict = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    for file_rel in args.batch_list:
        h5_path = os.path.join(args.local_root, file_rel)
        
        # Mirroring della struttura cartelle per i risultati
        dir_rel = os.path.dirname(file_rel).lstrip("./")
        out_dir = os.path.join(args.results_base, dir_rel)
        os.makedirs(out_dir, exist_ok=True)
        
        manager = HDF5EmbeddingDatasetsManager(h5_path, 'r')
        test_group = manager.hf['test']
        X = torch.from_numpy(test_group['embeddings'][:]).float().to(device)
        y = torch.from_numpy(test_group['labels'][:]).long().to(device)
        keys = test_group['keys'][:]
        manager.close()

        with torch.no_grad():
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = y.cpu().numpy()

        # Calcolo metriche con bias F0.5
        acc = accuracy_score(y_true, preds)
        p, r, f05, _ = precision_recall_fscore_support(y_true, preds, average='macro', beta=0.5)
        cm = confusion_matrix(y_true, preds)

        # 1. Salvataggio metriche
        pd.DataFrame([{"accuracy": acc, "precision": p, "recall": r, "f05": f05}]).to_csv(
            os.path.join(out_dir, "assessment_metrics.csv"), index=False)

        # 2. Explainability: Salvataggio chiavi degli errori
        mis_mask = preds != y_true
        pd.DataFrame({
            "key": [k.decode('utf-8') for k in keys[mis_mask]],
            "true": [classes[i] for i in y_true[mis_mask]],
            "pred": [classes[i] for i in preds[mis_mask]]
        }).to_csv(os.path.join(out_dir, "misclassified_keys.csv"), index=False)

        # 3. Plot Confusion Matrix nello stile richiesto
        plot_cm(cm, classes, os.path.join(out_dir, "confusion_matrix.png"), 
                f"Assessment: {dir_rel} | F0.5: {f05:.4f}")

if __name__ == "__main__":
    main()
