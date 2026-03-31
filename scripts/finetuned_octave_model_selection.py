import os, sys, argparse, torch, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

sys.path.insert(0, '/app')
from src.utils import HDF5EmbeddingDatasetsManager, get_config_from_yaml, load_single_cut_secs_dataloaders
from src.models import FinetunedModel

# ==============================================================================
# 🎯 RECOVERY STRATEGY: Class-Wise Centroid Imputation
# ==============================================================================
def impute_nans_with_class_centroids(X, y, num_classes):
    """
    Sostituisce i valori NaN negli embeddings con la media (centroide) 
    della specifica classe di appartenenza.
    """
    X_imputed = X.clone()
    for c in range(num_classes):
        class_mask = (y == c)
        if not class_mask.any():
            continue
            
        class_samples = X_imputed[class_mask]
        centroid = torch.nanmean(class_samples, dim=0)
        centroid = torch.nan_to_num(centroid, nan=0.0)
        
        nan_mask = torch.isnan(class_samples)
        if nan_mask.any():
            global_indices = torch.where(class_mask)[0]
            for i, idx in enumerate(global_indices):
                sample_nan_mask = nan_mask[i]
                if sample_nan_mask.any():
                    X_imputed[idx, sample_nan_mask] = centroid[sample_nan_mask]
                    
    return X_imputed
# ==============================================================================

def plot_cm(cm, classes, path, title):
    plt.figure(figsize=(14, 12))
    im = plt.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.title(title, fontsize=15)
    plt.colorbar(im)
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, ha='right')
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

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
    
    config_data = get_config_from_yaml(args.config_path)
    classes = config_data[0] 
    
    for file_rel in args.batch_list:
        h5_path = os.path.join(args.local_root, file_rel)
        dir_rel = os.path.dirname(file_rel).lstrip("./")
        dataloaders, classes = load_single_cut_secs_dataloaders(h5_path, args.cut_secs, 1024, device)
        model = train(dataloaders['train'], dataloaders['es'], optim_config, epochs, patience, device=device,
                                                  classes=classes, pretrained_path=args.model_path)
        out_dir = os.path.join(args.results_base, dir_rel)
        os.makedirs(out_dir, exist_ok=True)
        
        manager = HDF5EmbeddingDatasetsManager(h5_path, 'r')
        emb_dataset = manager.hf['embedding_dataset']
        X = torch.from_numpy(emb_dataset['embeddings'][:]).float().to(device)
        class_to_idx = {cls: i for i, cls in enumerate(classes)}
        raw_labels = emb_dataset['classes'][:]
        
        y_list = []
        for rl in raw_labels:
            label_str = rl.decode('utf-8') if isinstance(rl, bytes) else rl
            y_list.append(class_to_idx[label_str])
        y = torch.tensor(y_list).long().to(device)
        keys = emb_dataset['ID'][:]
        manager.close()

        print(f"🩹 Applying Centroid Imputation to {h5_path}...")
        X = impute_nans_with_class_centroids(X, y, len(classes))

        with torch.no_grad():
            logits = model(X)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            y_true = y.cpu().numpy()

        # Global metrics
        acc = accuracy_score(y_true, preds)
        p, r, f05, _ = precision_recall_fscore_support(y_true, preds, average='macro', beta=0.5, zero_division=0)
        
        # Per-class metrics
        p_class, r_class, f05_class, support = precision_recall_fscore_support(
            y_true, preds, average=None, beta=0.5, labels=np.arange(len(classes)), zero_division=0
        )
        
        cm = confusion_matrix(y_true, preds)

        # 1. Save Global Metrics
        pd.DataFrame([{"accuracy": acc, "precision": p, "recall": r, "f05": f05}]).to_csv(
            os.path.join(out_dir, "assessment_metrics.csv"), index=False)
            
        # 2. Save Per-Class Metrics (Accuracy per class = Recall)
        pd.DataFrame({
            "class": classes,
            "accuracy": r_class, 
            "precision": p_class,
            "recall": r_class,
            "f05": f05_class,
            "support": support
        }).to_csv(os.path.join(out_dir, "per_class_metrics.csv"), index=False)

        # 3. Explainability
        mis_mask = preds != y_true
        pd.DataFrame({
            "key": [k.decode('utf-8') for k in keys[mis_mask]],
            "true": [classes[i] for i in y_true[mis_mask]],
            "pred": [classes[i] for i in preds[mis_mask]]
        }).to_csv(os.path.join(out_dir, "misclassified_keys.csv"), index=False)

        # 4. Plot CM
        plot_cm(cm, classes, os.path.join(out_dir, "confusion_matrix.png"), 
                f"Assessment: {dir_rel} | F0.5: {f05:.4f}")

if __name__ == "__main__":
    main()
