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
from .utils_directories import *
from .losses import *
from .models import *

### Finetuned classifier training ###

def build_model():
    return FinetunedModel(classes, device='cpu')

def train(tr_set, es_set, config, epochs, patience=patience):
    """
    Training container function to train finetuned classifier
    with the different loss functions.

    args:
     - tr_set: container for training set's x-s and y-s;
     - es_set: container for validation set's x-s and y-s;
     - config: optimiser configuration;
     - epochs: training epochs (only if optimiser != RR);
     - patience: number of epochs with no improvement after which
                       training will be stopped.

    returns:
     - model: trained torch model for finetuned classifier.
    """
    model = build_model()
    optimizer, with_epochs = build_optimizer(config['optimizer'], model)
    with_epochs = config['optimizer']['builder'] != 'RR'
    if not with_epochs:
        epochs = 1
    best_es_accuracy = None
    best_params = model.state_dict()
    counter_es = 0
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(tr_set, desc=f'{epoch + 1}/{epochs} training'):
            if with_epochs:
                h = model(x)
                loss = criterion(h, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                optimizer(x, y)
        if not with_epochs:
            optimizer.set_readout()
        model.eval()
        _, es_accuracy, _ = get_scores(model, es_set)
        if best_es_accuracy is None or es_accuracy > best_es_accuracy:
            best_es_accuracy = es_accuracy
            best_params = model.state_dict()
            counter_es = 0
        else:
            counter_es += 1
            if counter_es > patience:
                model.load_state_dict(best_params)
                break
        if callback is not None:
            callback(model)
    print(f'Best ES accuracy: {best_es_accuracy} after {epoch + 1} epochs')
    model.eval()
    return model


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


def setup_distributed_environment(rank, world_size):
    """Setup the distributed environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if torch.cuda.is_available():
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        print(f"Processo {rank} di {world_size} avviato su GPU {rank}.")
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size)
        device = torch.device('cpu')
        print(f"Processo {rank} di {world_size} avviato su CPU {rank}.")

def cleanup_distributed_environment():
    """Cleanup the distributed environment."""
    dist.destroy_process_group()

def select_optim_distributed(rank, world_size, validation_filepath, dataloaders, classes, \
                                          epochs, patience, clap_model, classifier_model):
    """
    Model selection pipeline to find the best optimizer configuration for a given
    n-octaveband dataset using a distributed, multi-GPU approach.

    This function distributes the hyperparameter search space across multiple
    GPUs. Each GPU trains a subset of models, and the results are then
    gathered and processed by the master process (rank 0).

    args:
     - rank (int): The unique ID of the current process (0 to world_size - 1).
     - world_size (int): The total number of processes/GPUs to use.
     - validation_filepath (str): The directory where results and plots will be saved.
     - dataloaders (dict): A dictionary of dataloaders, with keys representing
                           different octaveband configurations.
     - classes (list): A list of class names in the dataset.
     - epochs (int): The maximum number of training epochs.
     - patience (int): The number of epochs with no improvement after which
                       training will be stopped.
     - clap_model (msclap.CLAP): The pre-initialized CLAP model (on CPU).
     - classifier_model (str): type of classifier; choose between 'linear' and 'xgboost'.

    returns:
     - None: The function saves the results to JSON and CSV files and generates
             confusion matrix plots.
    """
    # 1. Setup the distributed environment
    setup_distributed_environment(rank, world_size)
    
    # Define all hyperparameter configurations
    all_configs = [
        *[
            {
                'optimizer': {
                    'builder': 'RR',
                    'reg': reg,
                }
            } for reg in [0, 0.001, 0.01, 0.1, 1, 10, 50, 100, 150]
        ],
        *[
            {
                'optimizer': {
                    'builder': builder,
                    'lr': lr,
                }
            } for builder in ['SGD', 'Adam'] for lr in [0.1, 0.01, 0.001, 0.0001]
        ]
    ] if classifier_model == 'linear' else [
        *[
            {
                'optimizer': {
                    'builder': 'XGBoost',
                    'learning_rate': [0.01, 0.1],
                    'gamma': [0.1, 1, 5],
                    'reg_lambda': [5, 10, 20],
                    'max_depth': [4, 8]
  
                }
            }
        ]
    ]
    
    # 2. Divide configurations among processes
    configs_per_process = math.ceil(len(all_configs) / world_size)
    start_idx = rank * configs_per_process
    end_idx = min((rank + 1) * configs_per_process, len(all_configs))
    configs_subset = all_configs[start_idx:end_idx]

    local_results = {}
    
    for k, (tr_set, es_set, vl_set, _) in dataloaders.items():
        if k not in local_results:
            local_results[k] = []
            
        # Get original model scores only once per process
        criterion = torch.nn.CrossEntropyLoss()
        o_vl_loss, o_vl_accuracy, o_cm = get_scores(OriginalModel(classes, clap_model.get_text_embeddings, device=torch.device(f'cuda:{rank}')), vl_set)
        
        # We only save the original model results once to avoid duplicates, on rank 0
        local_results[k].append(dict(metrics=dict(type_learning='original', accuracy=o_vl_accuracy, loss=o_vl_loss, cm=o_cm.tolist())))

        # 3. Process the assigned configurations
        for config in tqdm(configs_subset, desc=f"GPU {rank} Processing configs for '{k}'"):
            # Modifica per includere il parametro 'reg' nella label
            config_label = str(config['optimizer'])
            if config['optimizer']['builder'] == 'RR':
                config_label = f"RR_reg={config['optimizer']['reg']}"
            elif config['optimizer']['builder'] in ['SGD', 'Adam']:
                config_label = f"{config['optimizer']['builder']}_lr={config['optimizer']['lr']}"
            else:
                config_label = f"{config['optimizer']['builder']}_"       + \
                               f"{config['optimizer']['learning_rate']}_" + \
                               f"{config['optimizer']['gamma']}_"         + \
                               f"{config['optimizer']['reg_lambda']}_"    + \
                               f"{config['optimizer']['max_depth']}"
            
            # Move sets to the correct device
            tr_set.dataset.to(torch.device(f'cuda:{rank}'))
            es_set.dataset.to(torch.device(f'cuda:{rank}'))
            vl_set.dataset.to(torch.device(f'cuda:{rank}'))

            if classifier_model == 'linear':
                model = train(tr_set, es_set, config, epochs, patience=patience) # Pass classes to train
            else:
                model = train_xgboost(tr_set, config) # Pass classes to train
            vl_loss, vl_accuracy, cm = get_scores(model, vl_set)
            
            # --- PLOT START ---
            # Plots are generated only on rank 0 to avoid conflicts
            disp_orig = ConfusionMatrixDisplay(confusion_matrix=o_cm, display_labels=classes)
            disp_ft = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            fig, axs = plt.subplots(2, figsize=(15, 20))

            axs[0].set_title(f'Original {k} acc={round(o_vl_accuracy * 100, 2)}% loss={round(o_vl_loss, 4)}')
            disp_orig.plot(xticks_rotation='vertical', ax=axs[0])

            axs[1].set_title(f'Finetuning {k} acc={round(vl_accuracy * 100, 2)}% loss={round(vl_loss, 4)} config={config_label}')
            disp_ft.plot(xticks_rotation='vertical', ax=axs[1])

            plt.tight_layout()
            plt.savefig(os.path.join(validation_filepath, f'{k}_{rank}_{config_label}.png'))
            # --- PLOT END ---
            
            # Store results
            local_results[k].append(dict(metrics=dict(type_learning='finetuning', accuracy=vl_accuracy,
                                                    loss=vl_loss, cm=cm.tolist()), hyperparams=config))
            
    # 4. Gather results from all processes to rank 0
    gathered_results = [None for _ in range(world_size)]
    dist.gather_object(local_results, gathered_results if rank == 0 else None, dst=0)

    # 5. Process final results on given rank
    final_results = {}
    for res in gathered_results:
        for k, v in res.items():
            if k not in final_results:
                final_results[k] = []
            final_results[k].extend(v)

    # Save results to json
    json.dump(final_results, open(os.path.join(validation_filepath, f'validation_ms_results_{classifier_model}_{rank}.json'), 'w'))

    # Sort and save to csv
    all_values = []
    for k, results in final_results.items():
        results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
        for t in results:
            values_dict = dict(time=k, **(t['hyperparams']['optimizer'] if 'hyperparams' in t else {}), **t['metrics'])
            # Aggiungi 'reg' al dizionario se è presente
            if 'reg' in t['hyperparams']['optimizer']:
                values_dict['reg'] = t['hyperparams']['optimizer']['reg']
            all_values.append(values_dict)

    df = pd.DataFrame(all_values)
    # Assicurati che le colonne 'builder', 'lr' e 'reg' esistano prima di riordinare
    if classifier_model == 'linear':
        df = df[['time', 'builder', 'lr', 'reg', 'accuracy', 'loss']].sort_values('accuracy', ascending=False)
    else:
        df = df[['time', 'builder', 'learning_rate', 'gamma', 'reg_lambda','max_depth',
                         'accuracy', 'loss']].sort_values('accuracy', ascending=False)
    df.to_csv(os.path.join(validation_filepath, 'validation_ms_results_{classifier_model}_{rank}.csv'))

    # 6. Cleanup
    cleanup_distributed_environment()
