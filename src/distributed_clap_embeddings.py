import os
import sys
import math
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm_multiprocess.tqdm_multiprocess import MultiProcessTqdm
import numpy as np
import logging
import traceback
import json

from .models import CLAP_initializer, spectrogram_n_octaveband_generator
from .utils import *
from .dirs_config import *

def process_class_with_cut_secs(clap_model, audio_embedding, class_to_process, cut_secs, n_octave, config, audio_dataset_manager=None):
    # --- SETUP INIZIALE (Invariato) ---
    root_source = config['dirs']['root_source']
    root_target = config['dirs']['root_target']
    target_class_dir = os.path.join(root_target, f'{cut_secs}_secs', class_to_process)
    os.makedirs(target_class_dir, exist_ok=True)
    
    audio_format = config['audio']['audio_format']
    sr = config['spectrogram']['sr']
    ref = config['spectrogram']['ref']
    center_freqs = config['spectrogram']['center_freqs']
    noise_perc = config['audio']['noise_perc']
    seed = config['audio']['seed']
    device = config['device']
    rank = config.get('rank', 0)

    def diag_print(msg):
        if rank == 0: print(f"[RANK 0 - DIAG] {msg}", flush=True)

    class_seed = seed + hash(class_to_process) % 10000000
    offset_rng = np.random.default_rng(class_seed)
    noise_rng = np.random.default_rng(class_seed)

    division_names = [d[0] for d in config['data']['divisions_xc_sizes_names']]
    target_counts_list = np.cumsum([d[1] for d in config['data']['divisions_xc_sizes_names']])

    # Gestione Manager Sorgente
    own_manager = False
    if audio_dataset_manager is None:
        audio_dataset_manager = HDF5DatasetManager(os.path.join(root_source, class_to_process,
                                                    f'{class_to_process}_{audio_format}_dataset.h5'))
        own_manager = True

    # 🎯 PUNTO 1: Inizializzazione Manager di Output a None (per Lazy Init)
    split_emb_dataset_manager = None
    di = 0
    results = 0
    round_ = 0
    n_embeddings_per_run = 0

    try:
        perms_metadata = audio_dataset_manager.get_reproducible_permutation(class_seed)
        n_records = len(perms_metadata)
        while True:
            round_ += 1
            for i in range(n_records):
                metadata = perms_metadata.iloc[i]
                track_idx = metadata['hdf5_index']
                track = audio_dataset_manager[track_idx]
                window_size = int(cut_secs * sr)
                
                offset = 0
                if round_ > 1 and track.shape[0] > window_size:
                    max_offset = track.shape[0] - window_size
                    if max_offset > 0:
                        offset = offset_rng.integers(0, max_offset)
                n_buckets = math.ceil((track.shape[0] - offset) / window_size)

                for b in range(n_buckets):
                    # 🎯 PUNTO 2: Logica di Cambio Split (Tua originale con FLUSH)
                    diag_print(f"{class_to_process} {cut_secs} {results} {di}")
                    if results >= target_counts_list[di]:
                        logging.info(f"Split '{division_names[di]}' completato. Avvio flush...")
                        
                        if split_emb_dataset_manager:
                            split_emb_dataset_manager.flush_buffers() # Manteniamo il tuo flush
                            split_emb_dataset_manager.close()
                            split_emb_dataset_manager = None # Forza la creazione del nuovo file
                        
                        di += 1
                        if di >= len(division_names):
                            if own_manager: audio_dataset_manager.close()
                            logging.info(f"Classe '{class_to_process}' elaborata. Totale: {results}")
                            return n_embeddings_per_run, True

                    # Identificazione chiave
                    emb_pkey = f"{audio_dataset_manager.hf.attrs['class_idx']}_{track_idx}_{b}_{round_}_{results}"

                    # Preparazione Audio (Necessaria per determinare la shape dello spettrogramma)
                    start = b * window_size + offset
                    end = start + window_size
                    cut_data = track[start:end]
                    if len(cut_data) < window_size:
                        cut_data = np.pad(cut_data, (0, window_size - len(cut_data)), 'constant')

                    max_threshold = np.mean(np.abs(cut_data))
                    noise = noise_rng.uniform(-max_threshold, max_threshold, cut_data.shape)
                    new_audio = (1 - noise_perc) * cut_data + noise_perc * noise
                    
                    # Generazione Spettrogramma (Shape Reale)
                    spec_n_o = spectrogram_n_octaveband_generator(new_audio, sr, integration_seconds=0.1,
                                                    n_octave=n_octave, center_freqs=center_freqs, ref=ref)

                    # 🎯 PUNTO 3: Inizializzazione Manager se None (Lazy Init con Shape reale)
                    if split_emb_dataset_manager is None:
                        h5_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}_{audio_format}_emb.h5')
                        split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(h5_path, 'a')
                        
                        # Inizializziamo con la shape reale di spec_n_o
                        split_emb_dataset_manager.initialize_hdf5(
                            embedding_dim=1024, spec_shape=spec_n_o.shape, 
                            audio_format=audio_format, cut_secs=cut_secs, 
                            n_octave=n_octave, sample_rate=sr, seed=seed, 
                            noise_perc=noise_perc, split=division_names[di], 
                            class_name=class_to_process
                        )
                        diag_print(f"Manager inizializzato per {division_names[di]} con shape {spec_n_o.shape}")

                    # Check esistenza
                    if emb_pkey in split_emb_dataset_manager:
                        results += 1
                        continue

                    # Calcolo Embedding
                    x = torch.tensor(new_audio, dtype=torch.float32).to(device).unsqueeze(0)
                    with torch.no_grad():
                       embedding = audio_embedding(x)[0][0]

                    # 🎯 PUNTO 4: Aggiunta al buffer
                    split_emb_dataset_manager.add_to_data_buffer(embedding, spec_n_o, emb_pkey,
                                metadata['track_name'], class_to_process, metadata['subclass'])
                        
                    results += 1
                    n_embeddings_per_run += 1

    except Exception as e:
        # 🎯 PUNTO 5: Gestione Errori con Flush di sicurezza
        diag_print("here")
        if split_emb_dataset_manager:
            split_emb_dataset_manager.flush_buffers()
            split_emb_dataset_manager.close()
        if audio_dataset_manager: audio_dataset_manager.close()
        logging.error(f"{traceback.format_exc()}")
        return n_embeddings_per_run, False

### Workers ###

def worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, pbar_instance=None):
    """
    Worker process SLURM con isolamento per classe (Latenza 2 ottimizzata).
    """
    # Setup ambiente distribuito PyTorch
    device = setup_distributed_environment(rank, world_size, slurm=True)
    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)

    # Configurazione worker
    config['rank'] = rank
    config['dirs']['root_source'] = os.path.join(basedir_raw, f'{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    config['device'] = str(device)
    
    # 🎯 Raggruppamento task per classe per minimizzare I/O contesa
    from collections import defaultdict
    tasks_by_class = defaultdict(list)
    for cut_secs, class_name in my_tasks:
        tasks_by_class[class_name].append(cut_secs)

    # Iterazione sulle classi assegnate a questo Rank
    for class_name, assigned_cuts in tasks_by_class.items():
        # Apertura manager RAW (Unica per classe)
        h5_path = os.path.join(config['dirs']['root_source'], class_name, f'{class_name}_{audio_format}_dataset.h5')
        
        try:
            current_audio_manager = HDF5DatasetManager(h5_path, audio_format)
            
            for cut_secs in assigned_cuts:
                if rank == 0:
                    logging.info(f"\n[RANK 0] >>> PROCESSO: {class_name} | {cut_secs}s")

                start_time = time.time()
                
                # Calcolo embedding
                n_embeddings_per_run, completed = process_class_with_cut_secs(
                    clap_model, audio_embedding, class_name, cut_secs, 
                    n_octave, config, audio_dataset_manager=current_audio_manager
                )
                
                # Log rank-specific
                target_log_dir = os.path.join(config['dirs']['root_target'], f'{cut_secs}_secs')
                process_time = time.time() - start_time
                write_log(target_log_dir, (cut_secs, class_name), process_time, n_embeddings_per_run, rank, completed, **config)

                if pbar_instance:
                    pbar_instance.update(1)

            # Chiusura manager RAW
            current_audio_manager.close()
            
        except Exception as e:
            logging.error(f"Errore Rank {rank} su classe {class_name}: {traceback.format_exc()}")
            continue

    cleanup_distributed_environment(rank)


# Funzione Worker per l'ambiente locale (richiamata da mp.Process)
def local_worker_process(audio_format, n_octave, config, rank, world_size, my_tasks, pbar_instance=None):
    """
    Esecuzione locale compartimentalizzata per classe per evitare deadlock HDF5.
    """
    device = setup_distributed_environment(rank, world_size, False)
    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)
    
    # 🎯 FIX CRUCIALE: Assicuriamo che il rank sia nel config per write_log
    config['rank'] = rank
    config['dirs']['root_source'] = os.path.join(basedir_raw, f'{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    config['device'] = str(device)

    # Raggruppiamo i task ricevuti per classe per gestire i manager in isolamento
    from collections import defaultdict
    tasks_by_class = defaultdict(list)
    for cut_secs, class_name in my_tasks:
        tasks_by_class[class_name].append(cut_secs)

    for class_name, assigned_cuts in tasks_by_class.items():
        # 🎯 Apertura ESCLUSIVA del file RAW per questa classe
        h5_path = os.path.join(config['dirs']['root_source'], class_name, f'{class_name}_{audio_format}_dataset.h5')
        
        try:
            current_audio_manager = HDF5DatasetManager(h5_path, audio_format)
            
            for cut_secs in assigned_cuts:
                if rank == 0:
                    print(f"\n[RANK 0] >>> PROCESSO TASK: {class_name} | {cut_secs}s", flush=True)

                start_time = time.time()
        
                # Esecuzione del calcolo embedding
                n_embeddings_per_run, completed = process_class_with_cut_secs(
                    clap_model, 
                    audio_embedding, 
                    class_name, 
                    cut_secs, 
                    n_octave, 
                    config, 
                    audio_dataset_manager=current_audio_manager
                )
        
                # Logging dei tempi e dei risultati (rank-specific)
                target_log_dir = os.path.join(config['dirs']['root_target'], f'{cut_secs}_secs')
                process_time = time.time() - start_time
                write_log(target_log_dir, (cut_secs, class_name), process_time, n_embeddings_per_run, rank, completed, **config)
        
                if pbar_instance:
                    pbar_instance.update(1)
            
            # 🎯 Chiusura esplicita dopo aver finito tutti i cut_secs della classe
            current_audio_manager.close() 

        except Exception as e:
            logging.error(f"Errore critico nel Rank {rank} sulla classe {class_name}: {traceback.format_exc()}")
            # In caso di errore sulla classe, il manager viene chiuso nel blocco except se necessario
            continue

    cleanup_distributed_environment(rank)


### Executions ###

def run_distributed_slurm(config_file, audio_format, n_octave):
    """
    Lancia la pipeline distribuita su SLURM con raggruppamento per classe.
    Ogni rank SLURM processa classi intere per evitare contese I/O.
    """
    # 1. Recupero rank e world_size dalle variabili d'ambiente di SLURM
    rank, world_size = setup_environ_vars(slurm=True)

    # 2. Setup logging e cartelle
    embed_folder = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    os.makedirs(embed_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(filename=os.path.join(embed_folder, 'log.txt'))])

    # 3. Caricamento configurazione
    classes_list, _, _, _, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
        divisions_xc_sizes_names = get_config_from_yaml(config_file)

    config = {
        'dirs': {}, 'audio': {}, 'spectrogram': {}, 'log': {}, 'data': {
            'divisions_xc_sizes_names': divisions_xc_sizes_names
        }
    }
    config['spectrogram'].update({'sr': sampling_rate, 'ref': ref, 'center_freqs': center_freqs})
    config['audio'].update({'noise_perc': noise_perc, 'seed': seed})

    # 4. Caricamento log per resumability
    log_data = {}
    log_path = os.path.join(embed_folder, 'log.json')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
                logging.info(f"Rank {rank}: Ripresa da log esistente.")
        except Exception as e:
            logging.error(f"Errore lettura log: {e}")

    # 🎯 5. LOGICA DI DISTRIBUZIONE PER CLASSE (ROUND-ROBIN)
    # Identifichiamo le classi che hanno ancora task da completare
    active_classes = []
    for class_name in classes_list:
        needs_work = False
        for cut_secs in cut_secs_list:
            log_key_str = str((cut_secs, class_name))
            task_info = log_data.get(log_key_str)
            if not task_info or not task_info.get('completed', False):
                needs_work = True
                break
        if needs_work:
            active_classes.append(class_name)

    active_classes.sort()
    
    # Assegnazione classi al rank SLURM corrente
    my_assigned_classes = active_classes[rank::world_size]
    
    # 6. Costruzione lista task specifica per questo Rank
    my_tasks = []
    for class_name in my_assigned_classes:
        for cut_secs in cut_secs_list:
            log_key_str = str((cut_secs, class_name))
            task_info = log_data.get(log_key_str)
            if not task_info or not task_info.get('completed', False):
                my_tasks.append((cut_secs, class_name))
    
    logging.info(f"Rank {rank}: classi assegnate {my_assigned_classes}")

    # 7. Setup Progress Bar (Solo Rank 0 coordina la barra globale)
    manager = mp.Manager()
    message_queue = manager.Queue() if world_size > 1 else None
    pbar = None
    if rank == 0:
        # Nota: Rank 0 stima il totale dei task attivi di tutti i rank per la pbar
        total_active_tasks = sum(1 for c in active_classes for s in cut_secs_list 
                                 if not log_data.get(str((s, c)), {}).get('completed'))
        if total_active_tasks > 0:
            pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="Progresso Totale", total=total_active_tasks)

    # 8. Esecuzione
    try:
        worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, pbar)
    except Exception as e:
        logging.error(f"Errore critico nel processo Rank {rank}: {e}")
        raise e
    finally:
        if rank == 0 and pbar:
            pbar.close()
        logging.info(f"Rank {rank}: Esecuzione terminata.")


def run_local_multiprocess(config_file, audio_format, n_octave, world_size):
    """
    Lancia la pipeline in locale su più core/GPU con raggruppamento per classe
    per evitare contese di risorse sui file HDF5.
    """
    # 1. Setup ambiente e logging
    setup_environ_vars(slurm=False)

    embed_folder = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    os.makedirs(embed_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(filename=os.path.join(embed_folder, 'log.txt'))])

    # 2. Caricamento configurazione
    classes_list, _, _, _, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
        divisions_xc_sizes_names = get_config_from_yaml(config_file)

    config = {
        'dirs': {}, 'audio': {}, 'spectrogram': {}, 'log': {}, 'data': {
            'divisions_xc_sizes_names': divisions_xc_sizes_names
        }
    }
    config['spectrogram'].update({'sr': sampling_rate, 'ref': ref, 'center_freqs': center_freqs})
    config['audio'].update({'noise_perc': noise_perc, 'seed': seed})

    # 3. Caricamento log per resumability
    log_data = {}
    log_path = os.path.join(embed_folder, 'log.json')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        except Exception as e:
            logging.error(f"Errore lettura log: {e}")

    # 🎯 4. LOGICA DI COMPARTIMENTALIZZAZIONE: Identificazione classi attive
    # Filtriamo solo le classi che hanno almeno un cut_secs non completato
    active_classes = []
    for class_name in classes_list:
        needs_work = False
        for cut_secs in cut_secs_list:
            log_key_str = str((cut_secs, class_name))
            task_info = log_data.get(log_key_str)
            if not task_info or not task_info.get('completed', False):
                needs_work = True
                break
        if needs_work:
            active_classes.append(class_name)

    # Ordiniamo le classi per una distribuzione round-robin prevedibile
    active_classes.sort()

    # 5. Setup Multiprocessing
    processes = []
    manager = mp.Manager()
    message_queue = manager.Queue()
    
    # Calcoliamo il totale dei task effettivi per la pbar
    total_tasks_to_run = 0
    tasks_distribution = [] # Debug per visibilità distribuzione

    # 🎯 6. DISTRIBUZIONE TASK PER RANK
    for rank in range(world_size):
        # Ogni rank prende un blocco di classi (es. Rank 0 prende classe 0, 4, 8...)
        my_assigned_classes = active_classes[rank::world_size]
        
        my_rank_tasks = []
        for class_name in my_assigned_classes:
            for cut_secs in cut_secs_list:
                log_key_str = str((cut_secs, class_name))
                task_info = log_data.get(log_key_str)
                if not task_info or not task_info.get('completed', False):
                    my_rank_tasks.append((cut_secs, class_name))
        
        tasks_distribution.append((rank, my_rank_tasks))
        total_tasks_to_run += len(my_rank_tasks)

    # 7. Avvio Progress Bar
    pbar = None
    if total_tasks_to_run > 0:
        pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="Progresso Totale", total=total_tasks_to_run)

    # 8. Lancio dei Processi
    try:
        for rank, my_tasks in tasks_distribution:
            if not my_tasks:
                logging.info(f"Rank {rank} non ha task assegnati.")
                continue
                
            print(f"Lancio Rank {rank}: classi assegnate {[t[1] for t in my_tasks]}")
            
            p = mp.Process(
                target=local_worker_process, 
                args=(audio_format, n_octave, config, rank, world_size, my_tasks, pbar)
            )
            p.start()
            processes.append(p)
            
        # Attesa completamento
        for p in processes:
            p.join()
            
    except Exception as e:
        logging.error(f"Errore critico nel processo padre: {e}")
        traceback.print_exc()
        raise e
    finally:
        if pbar:
            pbar.close()
        logging.info("Esecuzione locale terminata.")
