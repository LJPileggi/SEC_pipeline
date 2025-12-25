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
    # --- SETUP INIZIALE ---
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
    if rank == 0:
        print(division_names)

    # Manager Sorgente (Raw Audio)
    own_manager = False
    if audio_dataset_manager is None:
        audio_dataset_manager = HDF5DatasetManager(os.path.join(root_source, class_to_process,
                                                    f'{class_to_process}_{audio_format}_dataset.h5'))
        own_manager = True

    # Inizializzazioni
    split_emb_dataset_manager = None
    results = 0
    n_embeddings_per_run = 0
    di = 0 # Inizia sempre dal primo split (train)

    try:
        perms_metadata = audio_dataset_manager.get_reproducible_permutation(class_seed)
        n_records = len(perms_metadata)
        round_ = 0

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
                    if max_offset > 0: offset = offset_rng.integers(0, max_offset)
                
                n_buckets = math.ceil((track.shape[0] - offset) / window_size)

                for b in range(n_buckets):
                    
                    # 🎯 1. GESTIONE DINAMICA DELLO SPLIT (di)
                    # Avanza di split se abbiamo raggiunto il target di campioni
                    while di < len(division_names) and results >= target_counts_list[di]:
                        di += 1
                        if split_emb_dataset_manager:
                            split_emb_dataset_manager.close()
                            split_emb_dataset_manager = None
                    
                    # Se abbiamo esaurito tutti gli split
                    if di >= len(division_names):
                        if split_emb_dataset_manager: split_emb_dataset_manager.close()
                        if own_manager: audio_dataset_manager.close()
                        return n_embeddings_per_run, True

                    # 🎯 2. CHIAVE UNICA E PREPARAZIONE AUDIO
                    emb_pkey = f"{audio_dataset_manager.hf.attrs['class_idx']}_{track_idx}_{b}_{round_}_{results}"

                    start = b * window_size + offset
                    end = start + window_size
                    cut_data = track[start:end]
                    if len(cut_data) < window_size:
                        cut_data = np.pad(cut_data, (0, window_size - len(cut_data)), 'constant')

                    # 🎯 3. INIZIALIZZAZIONE LAZY DEL MANAGER
                    # Viene eseguita solo se cambiamo split o all'inizio del task
                    if split_emb_dataset_manager is None:
                        h5_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}_{audio_format}_emb.h5')
                        split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(h5_path, 'a')
                        
                        # Generiamo uno spettrogramma di prova per determinare la shape
                        temp_max = np.mean(np.abs(cut_data))
                        temp_noise = noise_rng.uniform(-temp_max, temp_max, cut_data.shape)
                        temp_audio = (1 - noise_perc) * cut_data + noise_perc * temp_noise
                        temp_spec = spectrogram_n_octaveband_generator(temp_audio, sr, integration_seconds=0.1,
                                                        n_octave=n_octave, center_freqs=center_freqs, ref=ref)
                        
                        split_emb_dataset_manager.initialize_hdf5(
                            embedding_dim=1024, spec_shape=temp_spec.shape, 
                            audio_format=audio_format, cut_secs=cut_secs, n_octave=n_octave, 
                            sample_rate=sr, seed=seed, noise_perc=noise_perc, 
                            split=division_names[di], class_name=class_to_process
                        )
                        diag_print(f"Manager pronto: {division_names[di]} {di} {results} {class_to_process} (Shape: {temp_spec.shape})")

                    # 🎯 4. CONTROLLO ESISTENZA (Resumability basata su HDF5)
                    # Questo permette di saltare il calcolo se il dato è già presente
                    if emb_pkey in split_emb_dataset_manager:
                        results += 1
                        continue

                    # 🎯 5. CALCOLO EFFETTIVO
                    max_threshold = np.mean(np.abs(cut_data))
                    noise = noise_rng.uniform(-max_threshold, max_threshold, cut_data.shape)
                    new_audio = (1 - noise_perc) * cut_data + noise_perc * noise
                    
                    spec_n_o = spectrogram_n_octaveband_generator(new_audio, sr, integration_seconds=0.1,
                                                    n_octave=n_octave, center_freqs=center_freqs, ref=ref)

                    x = torch.tensor(new_audio, dtype=torch.float32).to(device).unsqueeze(0)
                    with torch.no_grad():
                       embedding = audio_embedding(x)[0][0]

                    # Salvataggio nel buffer strutturato
                    split_emb_dataset_manager.add_to_data_buffer(embedding, spec_n_o, emb_pkey,
                                metadata['track_name'], class_to_process, metadata['subclass'])
                        
                    results += 1
                    n_embeddings_per_run += 1

    except Exception as e:
        if split_emb_dataset_manager:
            split_emb_dataset_manager.flush_buffers()
            split_emb_dataset_manager.close()
        if audio_dataset_manager: audio_dataset_manager.close()
        logging.error(f"{traceback.format_exc()}")
        return n_embeddings_per_run, False

### Workers ###

def worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, pbar_instance=None):
    """
    Worker process per ambiente SLURM ottimizzato (Latenza 2).
    """
    device = setup_distributed_environment(rank, world_size)
    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)

    # Configurazione path e parametri
    config['rank'] = rank
    config['dirs']['root_source'] = os.path.join(basedir_raw, f'{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    config['device'] = str(device)
    
    logging.info(f"Processo {rank} avviato su GPU {rank}.")

    # 🎯 STEP 1: Ordinamento per classe per massimizzare il riutilizzo del Manager Raw
    my_tasks.sort(key=lambda x: x[1]) 

    current_audio_manager = None
    last_class_name = None

    # 🎯 Rimosso il try/finally: il worker crasha se incontra errori fuori da process_class_with_cut_secs
    for cut_secs, class_name in my_tasks:
        if rank == 0:
            print(f"\n[RANK 0] Elaborando Classe: {class_name} | Cut: {cut_secs}s", flush=True)

        # 🎯 STEP 2: Gestione dinamica del Manager Raw (Latenza 2)
        if class_name != last_class_name:
            if current_audio_manager is not None:
                if rank == 0: print(f"[RANK 0] Cambio classe. Chiudo manager per {last_class_name}", flush=True)
                current_audio_manager.close()
            
            h5_path = os.path.join(config['dirs']['root_source'], class_name, f'{class_name}_{audio_format}_dataset.h5')
            if rank == 0: print(f"[RANK 0] Apertura nuovo manager raw per: {class_name}", flush=True)
            current_audio_manager = HDF5DatasetManager(h5_path, audio_format)
            last_class_name = class_name
        else:
            if rank == 0: print(f"[RANK 0] RIUTILIZZO manager esistente per la classe: {class_name}", flush=True)

        start_time = time.time()
        
        # Esegui la funzione di elaborazione degli embedding
        n_embeddings_per_run, completed = process_class_with_cut_secs(
            clap_model, 
            audio_embedding, 
            class_name, 
            cut_secs, 
            n_octave, 
            config, 
            audio_dataset_manager=current_audio_manager
        )

        target_log_dir = os.path.join(config['dirs']['root_target'], f'{cut_secs}_secs')
        process_time = time.time() - start_time
        
        # Scrittura del log (rank-specific)
        write_log(target_log_dir, (cut_secs, class_name), process_time, n_embeddings_per_run, rank, completed, **config)

        if pbar_instance:
            pbar_instance.update(1)

    # Chiusura finale (eseguita solo al completamento del loop)
    if current_audio_manager:
        current_audio_manager.close()
    cleanup_distributed_environment(rank)


# Funzione Worker per l'ambiente locale (richiamata da mp.Process)
def local_worker_process(audio_format, n_octave, config, rank, world_size, my_tasks, pbar_instance=None):
    device = setup_distributed_environment(rank, world_size, False)
    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)
    
    config['dirs']['root_source'] = os.path.join(basedir_raw, f'{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    if not os.path.exists(config['dirs']['root_target']):
        os.makedirs(config['dirs']['root_target'])
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    config['device'] = str(device)

    my_tasks.sort(key=lambda x: x[1]) 

    current_audio_manager = None
    last_class_name = None

    # 🎯 Rimosso il try/finally generale del worker per permettere crash espliciti
    for cut_secs, class_name in my_tasks:
        if rank == 0:
            print(f"\n[RANK 0] >>> PROCESSO TASK: {class_name} | {cut_secs}s", flush=True)

        if class_name != last_class_name:
            if current_audio_manager is not None:
                current_audio_manager.close()
            
            h5_path = os.path.join(config['dirs']['root_source'], class_name, f'{class_name}_{audio_format}_dataset.h5')
            current_audio_manager = HDF5DatasetManager(h5_path, audio_format)
            last_class_name = class_name

        start_time = time.time()
        
        # Chiamata alla funzione di elaborazione
        n_embeddings_per_run, completed = process_class_with_cut_secs(
            clap_model, audio_embedding, class_name, cut_secs, 
            n_octave, config, audio_dataset_manager=current_audio_manager
        )
        
        target_log_dir = os.path.join(config['dirs']['root_target'], f'{cut_secs}_secs')
        process_time = time.time() - start_time
        write_log(target_log_dir, (cut_secs, class_name), process_time, n_embeddings_per_run, rank, completed, **config)
        
        if pbar_instance:
            pbar_instance.update(1)

    # Chiusura finale (eseguita solo se il loop termina correttamente)
    if current_audio_manager:
        current_audio_manager.close()
    cleanup_distributed_environment(rank)


### Executions ###

def run_distributed_slurm(config_file, audio_format, n_octave):
    """
    Lancia la pipeline distribuita su SLURM con raggruppamento per classe.
    """
    # Recupera rank e world_size dalle variabili d'ambiente di SLURM
    rank, world_size = setup_environ_vars(slurm=True)

    # Setup logging e cartelle
    embed_folder = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    os.makedirs(embed_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(filename=os.path.join(embed_folder, 'log.txt'))])

    # Caricamento configurazione
    classes_list, _, _, _, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
        divisions_xc_sizes_names = get_config_from_yaml(config_file)

    config = {
        'dirs': {}, 'audio': {}, 'spectrogram': {}, 'log': {}, 'data': {
            'divisions_xc_sizes_names': divisions_xc_sizes_names
        }
    }
    config['spectrogram'].update({'sr': sampling_rate, 'ref': ref, 'center_freqs': center_freqs})
    config['audio'].update({'noise_perc': noise_perc, 'seed': seed})

    # Caricamento log per resumability
    log_data = {}
    log_path = os.path.join(embed_folder, 'log.json')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
                logging.info(f"Ripresa da log esistente ({len(log_data)} task registrati).")
        except Exception as e:
            logging.error(f"Errore lettura log: {e}")

    # 🎯 GENERAZIONE TASK OTTIMIZZATA
    all_possible_tasks = []
    # Invertiamo i loop: raggruppiamo prima per classe
    for class_name in classes_list:
        for cut_secs in cut_secs_list:
            log_key_str = str((cut_secs, class_name))
            
            # Aggiungiamo il task solo se non è presente nel log o se non è 'completed'
            task_info = log_data.get(log_key_str)
            if not task_info or not task_info.get('completed', False):
                all_possible_tasks.append((cut_secs, class_name))
            else:
                if rank == 0:
                    logging.info(f"Skipping task: {log_key_str} (già completato)")
    
    # 🎯 ORDINAMENTO FINALE: Massimizza la contiguità delle classi per ogni worker
    all_possible_tasks.sort(key=lambda x: x[1])

    # Distribuzione dei task al rank corrente
    my_tasks = all_possible_tasks[rank::world_size]
    
    # Progress bar solo sul Rank 0
    manager = mp.Manager()
    message_queue = manager.Queue() if world_size > 1 else None
    pbar = None
    if rank == 0 and len(all_possible_tasks) > 0:
        pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="Progresso Totale", total=len(all_possible_tasks))

    try:
        worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, pbar)
    except Exception as e:
        logging.error(f"Errore critico nel processo Rank {rank}: {e}")
        raise e
    finally:
        if rank == 0 and pbar:
            pbar.close()
        logging.info(f"Rank {rank}: Task terminati.")


def run_local_multiprocess(config_file, audio_format, n_octave, world_size):
    """
    Lancia la pipeline in locale su più core/GPU con raggruppamento per classe.
    """
    setup_environ_vars(slurm=False)

    embed_folder = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    os.makedirs(embed_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(filename=os.path.join(embed_folder, 'log.txt'))])

    classes_list, _, _, _, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
        divisions_xc_sizes_names = get_config_from_yaml(config_file)

    config = {
        'dirs': {}, 'audio': {}, 'spectrogram': {}, 'log': {}, 'data': {
            'divisions_xc_sizes_names': divisions_xc_sizes_names
        }
    }
    config['spectrogram'].update({'sr': sampling_rate, 'ref': ref, 'center_freqs': center_freqs})
    config['audio'].update({'noise_perc': noise_perc, 'seed': seed})

    # Caricamento log
    log_data = {}
    log_path = os.path.join(embed_folder, 'log.json')
    if os.path.exists(log_path):
        with open(log_path, 'r') as f:
            log_data = json.load(f)

    # 🎯 GENERAZIONE E ORDINAMENTO TASK (Raggruppati per classe)
    all_tasks = []
    for class_name in classes_list:
        for cut_secs in cut_secs_list:
            log_key_str = str((cut_secs, class_name))
            task_info = log_data.get(log_key_str)
            if not task_info or not task_info.get('completed', False):
                all_tasks.append((cut_secs, class_name))

    # Ordiniamo per assicurarci che task della stessa classe siano vicini
    all_tasks.sort(key=lambda x: x[1])

    processes = []
    manager = mp.Manager()
    message_queue = manager.Queue()
    pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="Progresso Totale", total=len(all_tasks))

    try:
        for rank in range(world_size):
            # Ogni processo locale riceve la sua fetta di task pre-ordinata
            my_tasks = all_tasks[rank::world_size]
            p = mp.Process(target=local_worker_process, 
                           args=(audio_format, n_octave, config, rank, world_size, my_tasks, pbar))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
    except Exception as e:
        logging.error(f"Errore critico nel processo padre: {e}")
        raise e
    finally:
        if pbar:
            pbar.close()
