import os
import sys
import math
import time
import gc
import ctypes
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

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

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
        print(f"[RANK {rank} - DIAG] {msg}", flush=True)

    def trim_memory():
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

    class_seed = seed + hash(class_to_process) % 10000000
    offset_rng = np.random.default_rng(class_seed)
    noise_rng = np.random.default_rng(class_seed)

    division_names = [d[0] for d in config['data']['divisions_xc_sizes_names']]
    target_counts_list = np.cumsum([d[1] for d in config['data']['divisions_xc_sizes_names']])

    own_manager = False
    if audio_dataset_manager is None:
        audio_dataset_manager = HDF5DatasetManager(os.path.join(root_source, class_to_process,
                                                    f'{class_to_process}_{audio_format}_dataset.h5'))
        own_manager = True

    split_emb_dataset_manager = None
    di = 0
    results = 0
    round_ = 0
    n_embeddings_per_run = 0

    try:
        # 🎯 Otteniamo array di indici invece di DataFrame
        permuted_indices = audio_dataset_manager.get_reproducible_permutation(class_seed)
        n_records = len(permuted_indices)
        
        while True:
            round_ += 1
            for track_idx in permuted_indices:
                # 🎯 Lettura atomica: traccia + metadati
                track, metadata = audio_dataset_manager.get_audio_and_metadata(track_idx)
                
                window_size = int(cut_secs * sr)
                offset = 0
                if round_ > 1 and track.shape[0] > window_size:
                    max_offset = track.shape[0] - window_size
                    if max_offset > 0:
                        offset = offset_rng.integers(0, max_offset)
                n_buckets = math.ceil((track.shape[0] - offset) / window_size)

                for b in range(n_buckets):
                    if results % 10 == 0:
                        diag_print(f"{class_to_process}, {cut_secs} s: created {results}/{target_counts_list[di]} embeddings; split '{division_names[di]}'")
                    
                    if results >= target_counts_list[di]:
                        logging.info(f"Split '{division_names[di]}' per {class_to_process} completato. Avvio flush...")
                        if split_emb_dataset_manager:
                            split_emb_dataset_manager.close()
                            del split_emb_dataset_manager
                            split_emb_dataset_manager = None
                            gc.collect()
                            time.sleep(0.5)
                        
                        di += 1
                        if di >= len(division_names):
                            if own_manager: audio_dataset_manager.close()
                            logging.info(f"Classe '{class_to_process}' elaborata. Totale: {results}")
                            return n_embeddings_per_run, True

                    emb_pkey = f"{audio_dataset_manager.hf.attrs.get('class_idx', 0)}_{track_idx}_{b}_{round_}_{results}"

                    # Preparazione Audio
                    start = b * window_size + offset
                    end = start + window_size
                    cut_data = track[start:end]
                    if len(cut_data) < window_size:
                        cut_data = np.pad(cut_data, (0, window_size - len(cut_data)), 'constant')

                    max_threshold = np.mean(np.abs(cut_data))
                    noise = noise_rng.uniform(-max_threshold, max_threshold, cut_data.shape)
                    new_audio = (1 - noise_perc) * cut_data + noise_perc * noise
                    
                    spec_n_o = spectrogram_n_octaveband_generator(new_audio, sr, integration_seconds=0.1,
                                                    n_octave=n_octave, center_freqs=center_freqs, ref=ref)

                    if split_emb_dataset_manager is None:
                        h5_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}_{audio_format}_emb.h5')
                        split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(h5_path, 'a')
                        split_emb_dataset_manager.initialize_hdf5(
                            1024, spec_n_o.shape, audio_format, cut_secs, n_octave, 
                            sr, seed, noise_perc, division_names[di], class_to_process
                        )

                    if emb_pkey in split_emb_dataset_manager:
                        results += 1
                        continue

                    # Inferenza
                    x = torch.tensor(new_audio, dtype=torch.float32).to(device).unsqueeze(0)
                    with torch.inference_mode():
                       embedding = audio_embedding(x)[0][0]
                    embedding_cpu = embedding.detach().cpu().numpy()
                    del x, embedding

                    split_emb_dataset_manager.add_to_data_buffer(embedding_cpu, spec_n_o, emb_pkey,
                                    metadata['track_name'], class_to_process, metadata['subclass'])

                    # 🎯 PULIZIA TOTALE
                    del embedding_cpu, spec_n_o, new_audio, cut_data
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    torch.set_num_threads(1)
                    
                    results += 1
                    n_embeddings_per_run += 1
                
                # 🎯 Pulizia dopo ogni traccia
                del track, metadata
                if results % 10 == 0:
                    gc.collect()
                    trim_memory()

    except Exception:
        if split_emb_dataset_manager:
            split_emb_dataset_manager.close()
        if own_manager: audio_dataset_manager.close()
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
                gc.collect()
                
                # Log rank-specific
                target_log_dir = os.path.join(config['dirs']['root_target'], f'{cut_secs}_secs')
                process_time = time.time() - start_time
                write_log(target_log_dir, (cut_secs, class_name), process_time, n_embeddings_per_run, completed, **config)

                if pbar_instance:
                    pbar_instance.update(1)

            # Chiusura manager RAW
            current_audio_manager.close()
            del current_audio_manager
            gc.collect()
            
        except Exception as e:
            logging.error(f"Errore Rank {rank} su classe {class_name}: {traceback.format_exc()}")
            continue

    cleanup_distributed_environment(rank)


# Funzione Worker per l'ambiente locale (richiamata da mp.Process)
def local_worker_process(audio_format, n_octave, config, rank, world_size, my_tasks, pbar_instance=None):
    """
    Esecuzione locale compartimentalizzata per classe per evitare deadlock HDF5.
    """
    import faulthandler
    faulthandler.enable()

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
                # if rank == 0:
                print(f"\n[RANK {rank}] >>> PROCESSO TASK: {class_name} | {cut_secs}s", flush=True)

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
                gc.collect()
        
                # Logging dei tempi e dei risultati (rank-specific)
                target_log_dir = os.path.join(config['dirs']['root_target'], f'{cut_secs}_secs')
                process_time = time.time() - start_time
                write_log(target_log_dir, (cut_secs, class_name), process_time, n_embeddings_per_run, completed, **config)
        
                if pbar_instance:
                    pbar_instance.update(1)
            
            # 🎯 Chiusura esplicita dopo aver finito tutti i cut_secs della classe
            current_audio_manager.close() 
            del current_audio_manager
            gc.collect()

        except Exception as e:
            logging.error(f"Errore critico nel Rank {rank} sulla classe {class_name}: {traceback.format_exc()}")
            # In caso di errore sulla classe, il manager viene chiuso nel blocco except se necessario
            continue

    cleanup_distributed_environment(rank)


### Executions ###

def run_distributed_slurm(config_file, audio_format, n_octave):
    """
    Lancia la pipeline distribuita su SLURM adattando il world_size 
    al numero di classi attive per evitare stalli nel rendezvous.
    """
    # 1. Recupero parametri SLURM originali
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
        except Exception as e:
            logging.error(f"Errore lettura log: {e}")

    # 🎯 5. IDENTIFICAZIONE CLASSI ATTIVE
    active_classes = []
    for class_name in classes_list:
        needs_work = False
        for cut_secs in cut_secs_list:
            log_key_str = str((cut_secs, class_name))
            if not log_data.get(log_key_str, {}).get('completed', False):
                needs_work = True
                break
        if needs_work:
            active_classes.append(class_name)

    active_classes.sort()

    # 🎯 6. RIDIMENSIONAMENTO DINAMICO DEL WORLD SIZE PER SLURM
    # Il world_size effettivo non può superare il numero di classi attive
    actual_world_size = min(len(active_classes), world_size)

    # Se il rank corrente è fuori dal nuovo world_size, esce subito
    if rank >= actual_world_size:
        logging.info(f"Rank {rank} inattivo: actual_world_size ridotto a {actual_world_size}. Uscita.")
        return

    if actual_world_size == 0:
        if rank == 0: logging.info("Tutti i task completati.")
        return

    # 🎯 7. DISTRIBUZIONE TASK (Round-robin sul nuovo world_size)
    my_assigned_classes = active_classes[rank::actual_world_size]
    my_tasks = []
    for class_name in my_assigned_classes:
        for cut_secs in cut_secs_list:
            log_key_str = str((cut_secs, class_name))
            if not log_data.get(log_key_str, {}).get('completed'):
                my_tasks.append((cut_secs, class_name))

    logging.info(f"Rank {rank}/{actual_world_size}: classi assegnate {my_assigned_classes}")

    # 8. Setup Progress Bar (Solo Rank 0)
    manager = mp.Manager()
    message_queue = manager.Queue() if actual_world_size > 1 else None
    pbar = None
    if rank == 0:
        total_active_tasks = sum(1 for c in active_classes for s in cut_secs_list 
                                 if not log_data.get(str((s, c)), {}).get('completed'))
        if total_active_tasks > 0:
            pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="Progresso Totale SLURM", total=total_active_tasks)

    # 9. Esecuzione Worker
    try:
        # Passiamo actual_world_size per il rendezvous corretto in setup_distributed_environment
        worker_process_slurm(audio_format, n_octave, config, rank, actual_world_size, my_tasks, pbar)
    except Exception as e:
        logging.error(f"Errore critico Rank {rank}: {e}")
        raise e
    finally:
        if rank == 0 and pbar:
            pbar.close()
        logging.info(f"Rank {rank}: Esecuzione terminata.")


def run_local_multiprocess(config_file, audio_format, n_octave, world_size):
    """
    Lancia la pipeline in locale adattando dinamicamente il numero di processi
    al numero di classi attive per evitare deadlock nel rendezvous.
    """
    # 1. Setup ambiente e variabili
    setup_environ_vars(slurm=False)
    
    # 🎯 FIX 1: Porta dinamica per evitare "Address already in use"
    import random
    os.environ['MASTER_PORT'] = str(random.randint(29500, 29999))

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
    # Filtriamo solo le classi che hanno almeno un task incompleto
    active_classes = []
    for class_name in classes_list:
        needs_work = False
        for cut_secs in cut_secs_list:
            log_key_str = str((cut_secs, class_name))
            if not log_data.get(log_key_str, {}).get('completed', False):
                needs_work = True
                break
        if needs_work:
            active_classes.append(class_name)

    active_classes.sort()

    # 🎯 FIX 2: Ridimensionamento dinamico del World Size
    # Usiamo il minimo tra le classi disponibili e i core/GPU richiesti
    actual_world_size = min(len(active_classes), world_size)
    
    if actual_world_size == 0:
        logging.info("Tutti i task risultano completati nel log.json.")
        return

    print(f"Ambiente locale rilevato: {len(active_classes)} classi attive. Avvio con {actual_world_size} processi.")

    # 5. Distribuzione Task per Rank
    processes = []
    manager = mp.Manager()
    message_queue = manager.Queue()
    
    tasks_distribution = []
    total_tasks_to_run = 0

    for rank in range(actual_world_size):
        # Distribuzione round-robin basata sul NUOVO world size
        my_assigned_classes = active_classes[rank::actual_world_size]
        
        my_rank_tasks = []
        for class_name in my_assigned_classes:
            for cut_secs in cut_secs_list:
                log_key_str = str((cut_secs, class_name))
                if not log_data.get(log_key_str, {}).get('completed'):
                    my_rank_tasks.append((cut_secs, class_name))
        
        if my_rank_tasks:
            tasks_distribution.append((rank, my_rank_tasks))
            total_tasks_to_run += len(my_rank_tasks)

    # 6. Progress Bar
    pbar = None
    if total_tasks_to_run > 0:
        pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="Progresso Totale", total=total_tasks_to_run)

    # 7. Lancio dei Processi
    try:
        for rank, my_tasks in tasks_distribution:
            print(f"Lancio Rank {rank}: classi assegnate {[t[1] for t in my_tasks]}")
            
            # Passiamo actual_world_size per garantire il rendezvous corretto
            p = mp.Process(
                target=local_worker_process, 
                args=(audio_format, n_octave, config, rank, actual_world_size, my_tasks, pbar)
            )
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            if p.exitcode != 0:
                logging.error(f"⚠️ PROCESSO CRASHATO! Rank: {processes.index(p)}, Exit Code: {p.exitcode}")
                if p.exitcode == -11:
                    logging.error("Diagnosi: Segmentation Fault (SIGSEGV)")
                elif p.exitcode == -9:
                    logging.error("Diagnosi: Processo ucciso dal sistema (probabile OOM Killer)")
            
    except Exception as e:
        logging.error(f"Errore critico nel processo padre: {e}")
        traceback.print_exc()
        raise e
    finally:
        if pbar:
            pbar.close()
        logging.info("Esecuzione locale completata.")
