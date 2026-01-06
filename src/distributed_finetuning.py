import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.distributed as dist
from tqdm import tqdm
from xgboost import XGBClassifier

from .utils import *
from .dirs_config import *
from .losses import *
from .models import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

### Finetuned classifier training ###

def build_model():
    return FinetunedModel(classes, device='cpu')

def train_xgboost(tr_set, config):
    """
    Train a model with a given configuration.
    
    This function now supports both PyTorch models and non-PyTorch models
    like XGBoost.
    """
    # Extract features and labels from the PyTorch dataloaders
    X_train = np.vstack([x.cpu().numpy() for x, y in tr_set])
    y_train = np.hstack([y.cpu().numpy() for x, y in tr_set])

    # Define the XGBoost model with hyperparameters from the config
    model = XGBClassifier(
        objective='multi:softprob',
        eval_metric='mlogloss',
        use_label_encoder=False,
        **config
        )

    # Train the model
    model.fit(X_train, y_train)

    return model

def train(tr_set, es_set, config, epochs, patience, device='cpu', classes=None):
    """
    Funzione di training universale (CPU/GPU).
    """
    model = FinetunedModel(classes, device=device)
    optimizer, with_epochs = build_optimizer(config['optimizer'], model)
    
    # Se RR non usa epoche, forziamo a 1
    actual_epochs = epochs if with_epochs else 1
    criterion = torch.nn.CrossEntropyLoss().to(device)
    
    best_es_accuracy = 0.0
    best_params = model.state_dict()
    counter_es = 0

    for epoch in range(actual_epochs):
        model.train()
        for x, y in tr_set:
            x, y = x.to(device), y.to(device)
            if with_epochs:
                optimizer.zero_grad()
                h = model(x)
                loss = criterion(h, y)
                loss.backward()
                optimizer.step()
            else:
                optimizer(x, y) # Logica Ridge Regression
        
        if not with_epochs:
            optimizer.set_readout()
            
        model.eval()
        _, es_accuracy, _ = get_scores(model, es_set, device=device)
        
        if es_accuracy > best_es_accuracy:
            best_es_accuracy = es_accuracy
            best_params = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            counter_es = 0
        else:
            counter_es += 1
            if counter_es > patience:
                break
                
    model.load_state_dict(best_params)
    return model

def save_validation_plot(validation_filepath, k, config_label, o_acc, o_loss, o_cm, vl_acc, vl_loss, cm, classes, rank):
    """
    Modulo isolato per la generazione dei plot di confronto matrici di confusione.
    """
    disp_orig = ConfusionMatrixDisplay(confusion_matrix=o_cm, display_labels=classes)
    disp_ft = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, axs = plt.subplots(2, figsize=(15, 20))

    axs[0].set_title(f'Original {k} acc={round(o_acc * 100, 2)}% loss={round(o_loss, 4)}')
    disp_orig.plot(xticks_rotation='vertical', ax=axs[0])

    axs[1].set_title(f'Finetuning {k} acc={round(vl_acc * 100, 2)}% loss={round(vl_loss, 4)} config={config_label}')
    disp_ft.plot(xticks_rotation='vertical', ax=axs[1])

    plt.tight_layout()
    plt.savefig(os.path.join(validation_filepath, f'{k}_rank_{rank}_{config_label}.png'))
    plt.close(fig)

def process_and_save_final_results(local_results, validation_filepath, classifier_model, rank):
    """
    Salva i risultati del rank corrente in formato JSON e CSV.
    Seguendo la tua preferenza, ogni rank scrive il suo file per l'unione successiva via SH.
    """
    # 1. Save results to json
    json_path = os.path.join(validation_filepath, f'validation_ms_results_{classifier_model}_rank_{rank}.json')
    with open(json_path, 'w') as f:
        json.dump(local_results, f, indent=4)

    # 2. Sort and save to csv
    all_values = []
    for k, results in local_results.items():
        # Ordiniamo per accuratezza (decrescente)
        results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
        for t in results:
            # Appiattiamo il dizionario per il DataFrame
            values_dict = dict(time=k, **(t['hyperparams']['optimizer'] if 'hyperparams' in t else {}), **t['metrics'])
            if 'hyperparams' in t and 'reg' in t['hyperparams']['optimizer']:
                values_dict['reg'] = t['hyperparams']['optimizer']['reg']
            all_values.append(values_dict)

    df = pd.DataFrame(all_values)
    if not df.empty:
        if classifier_model == 'linear':
            cols = ['time', 'builder', 'lr', 'reg', 'accuracy', 'loss']
            df = df[[c for c in cols if c in df.columns]].sort_values('accuracy', ascending=False)
        else:
            cols = ['time', 'builder', 'learning_rate', 'gamma', 'reg_lambda','max_depth', 'accuracy', 'loss']
            df = df[[c for c in cols if c in df.columns]].sort_values('accuracy', ascending=False)
            
        csv_path = os.path.join(validation_filepath, f'validation_ms_results_{classifier_model}_rank_{rank}.csv')
        df.to_csv(csv_path, index=False)

def select_optim_distributed(rank, world_size, validation_filepath, octaveband_dir, 
                             my_cut_secs, classes_global, epochs, patience, clap_model, classifier_model):
    """
    Pipeline distribuita con salvataggio rank-specific.
    """
    device = setup_distributed_environment(rank, world_size, slurm=os.environ.get('SLURM_JOB_ID') is not None)
    
    all_configs = [
        *[{'optimizer': {'builder': 'RR', 'reg': r}} for r in [0, 0.001, 0.01, 0.1, 1, 10, 50, 100, 150]],
        *[{'optimizer': {'builder': b, 'lr': lr}} for b in ['SGD', 'Adam'] for lr in [0.1, 0.01, 0.001, 0.0001]]
    ] if classifier_model == 'linear' else [{'optimizer': {'builder': 'XGBoost', 'learning_rate': [0.01, 0.1], 'gamma': [0.1, 1, 5], 'reg_lambda': [5, 10, 20], 'max_depth': [4, 8]}}]

    local_results = {}

    for cut_secs in my_cut_secs:
        k = f"{cut_secs}_secs"
        local_results[k] = []
        
        # Caricamento ottimizzato (I/O efficiente)
        dataloaders, classes = load_single_cut_secs_dataloaders(octaveband_dir, cut_secs, 1024, device)
        
        # Zero-shot CLAP
        o_loss, o_acc, o_cm = get_scores(OriginalModel(classes, clap_model.get_text_embeddings, device=device), dataloaders['valid'], device=device)
        local_results[k].append(dict(metrics=dict(type_learning='original', accuracy=o_acc, loss=o_loss, cm=o_cm.tolist())))

        for config in tqdm(all_configs, desc=f"Rank {rank} | {k}"):
            # Training Universale (CPU/GPU)
            model = train(dataloaders['train'], dataloaders['es'], config, epochs, patience, device=device, classes=classes)
            vl_loss, vl_acc, cm = get_scores(model, dataloaders['valid'], device=device)
            
            # Label per il plot e il log
            label = f"{config['optimizer']['builder']}_{config['optimizer'].get('reg', config['optimizer'].get('lr', ''))}"
            
            # 🎯 Chiamata alla nuova funzione di plotting modulare
            save_validation_plot(validation_filepath, k, label, o_acc, o_loss, o_cm, vl_acc, vl_loss, cm, classes, rank)

            local_results[k].append(dict(metrics=dict(type_learning='finetuning', accuracy=vl_acc, loss=vl_loss, cm=cm.tolist()), hyperparams=config))

    # 🎯 Chiamata alla nuova funzione di salvataggio rank-specific
    process_and_save_final_results(local_results, validation_filepath, classifier_model, rank)
    
    cleanup_distributed_environment(rank)
