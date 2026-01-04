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

from .models import CLAP_initializer, spectrogram_n_octaveband_generator, spectrogram_n_octaveband_generator_gpu
from .utils import *
from .dirs_config import *

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

def process_class_with_cut_secs_slurm_batched(clap_model, audio_embedding, class_to_process, cut_secs, n_octave, config, audio_dataset_manager=None):
    # --- SETUP ORIGINALE (Intatto) ---
    root_source, root_target = config['dirs']['root_source'], config['dirs']['root_target']
    target_class_dir = os.path.join(root_target, f'{cut_secs}_secs', class_to_process)
    os.makedirs(target_class_dir, exist_ok=True)
    
    audio_format, sr, ref = config['audio']['audio_format'], config['spectrogram']['sr'], config['spectrogram']['ref']
    center_freqs, noise_perc, seed = config['spectrogram']['center_freqs'], config['audio']['noise_perc'], config['audio']['seed']
    device, rank = config['device'], config.get('rank', 0)

    # 📊 PERFORMANCE & LOGGING
    stats = {"load": 0, "gpu_total": 0, "save": 0, "count": 0}
    BATCH_SIZE = 128 
    perf_log_path = os.path.join(root_target, f"perf_stats_rank_{rank}.txt")
    
    def write_perf_log():
        if stats["count"] > 0:
            c = stats["count"]
            with open(perf_log_path, "a") as f:
                f.write(f"\n[RANK {rank} - {class_to_process} {cut_secs}s @ {time.strftime('%H:%M:%S')}] Analisi su {c} campioni:\n")
                f.write(f"  - I/O Disco:       {stats['load']/c:.6f}s\n")
                f.write(f"  - GPU (Spec+Emb):  {stats['gpu_total']/c:.6f}s\n")
                f.write(f"  - Scrittura H5:    {stats['save']/c:.6f}s\n")
                f.write(f"  - TOTALE X AUDIO:  {sum(v for k,v in stats.items() if k!='count')/c:.6f}s\n")

    class_seed = int(seed + hash(class_to_process) % 10000000)
    offset_rng = np.random.default_rng(class_seed)
    division_names = [d[0] for d in config['data']['divisions_xc_sizes_names']]
    target_counts_list = np.cumsum([d[1] for d in config['data']['divisions_xc_sizes_names']])

    own_manager = False
    if audio_dataset_manager is None:
        audio_dataset_manager = HDF5DatasetManager(os.path.join(root_source, class_to_process, f'{class_to_process}_{audio_format}_dataset.h5'))
        own_manager = True

    split_emb_dataset_manager = None
    di = 0; results = 0; round_ = 0; n_embeddings_per_run = 0
    class_idx_attr = audio_dataset_manager.hf.attrs.get('class_idx', 0)
    adaptive_buffer_size = max(BATCH_SIZE, int(256 / cut_secs))
    torch.set_num_threads(1)

    batch_audio = []; batch_meta = []

    def flush_batch():
        nonlocal split_emb_dataset_manager, n_embeddings_per_run
        if not batch_audio: return
        
        t_gpu_start = time.perf_counter()
        raw_batch = torch.stack(batch_audio).pin_memory().to(device, non_blocking=True)
        
        # 🎯 DETERMINISMO: Seed locale basato sulla posizione globale nella classe
        current_batch_start_idx = results - len(batch_audio)
        batch_noise_seed = class_seed + current_batch_start_idx
        g = torch.Generator(device=device).manual_seed(batch_noise_seed)
        
        with torch.inference_mode():
            # Rumore Vettorizzato Deterministicamente
            means = torch.mean(torch.abs(raw_batch), dim=1, keepdim=True)
            # Uniforme [-1, 1] usando il generatore con seed dedicato
            noise = (torch.rand(raw_batch.shape, generator=g, device=device) * 2 - 1) * means
            batch_tensor = (1 - noise_perc) * raw_batch + noise_perc * noise
            
            # Mixed Precision Inference
            with torch.cuda.amp.autocast():
                from .models import spectrogram_n_octaveband_generator_gpu
                specs_gpu = spectrogram_n_octaveband_generator_gpu(batch_tensor, sr, n_octave, center_freqs=center_freqs, ref=ref, device=device)
                
                output = audio_embedding(batch_tensor)
                embeddings = output[0] if isinstance(output, (tuple, list)) else output
                if embeddings.dim() > 2: embeddings = embeddings.squeeze(1)

        embeddings_cpu = embeddings.float().cpu().numpy()
        specs_cpu = specs_gpu.float().cpu().numpy()
        stats["gpu_total"] += (time.perf_counter() - t_gpu_start)

        t_save_start = time.perf_counter()
        for i in range(len(embeddings_cpu)):
            if split_emb_dataset_manager is None:
                h5_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}_{audio_format}_emb.h5')
                split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(h5_path, 'a', buffer_size=adaptive_buffer_size)
                split_emb_dataset_manager.initialize_hdf5(1024, specs_cpu[i].shape, audio_format, cut_secs, n_octave, sr, seed, noise_perc, division_names[di], class_to_process)

            split_emb_dataset_manager.add_to_data_buffer(embeddings_cpu[i], specs_cpu[i], batch_meta[i]['pkey'], batch_meta[i]['name'], class_to_process, batch_meta[i]['sub'])
        
        stats["save"] += (time.perf_counter() - t_save_start)
        cur_batch = len(batch_audio)
        stats["count"] += cur_batch
        n_embeddings_per_run += cur_batch
        
        if stats["count"] % (BATCH_SIZE * 4) == 0: write_perf_log()
        batch_audio.clear(); batch_meta.clear()

    try:
        permuted_indices = audio_dataset_manager.get_reproducible_permutation(class_seed)
        while True:
            round_ += 1
            for track_idx in permuted_indices:
                t_load_s = time.perf_counter()
                track, metadata = audio_dataset_manager.get_audio_and_metadata(track_idx)
                stats["load"] += (time.perf_counter() - t_load_s)
                
                window_size = int(cut_secs * sr)
                offset = offset_rng.integers(0, track.shape[0]-window_size) if round_>1 and track.shape[0]>window_size else 0
                n_buckets = math.ceil((track.shape[0] - offset) / window_size)

                for b in range(n_buckets):
                    if results >= target_counts_list[di]:
                        flush_batch()
                        if split_emb_dataset_manager:
                            split_emb_dataset_manager.close()
                            split_emb_dataset_manager = None
                        di += 1
                        if di >= len(division_names):
                            if own_manager: audio_dataset_manager.close()
                            return n_embeddings_per_run, True

                    emb_pkey = f"{class_idx_attr}_{track_idx}_{b}_{round_}_{results}"
                    if split_emb_dataset_manager and emb_pkey in split_emb_dataset_manager:
                        results += 1; continue

                    start, end = b*window_size+offset, (b+1)*window_size+offset
                    cut_data = track[start:end]
                    if len(cut_data) < window_size: cut_data = np.pad(cut_data, (0, window_size-len(cut_data)), 'constant')
                    
                    batch_audio.append(torch.from_numpy(cut_data).float())
                    batch_meta.append({'pkey': emb_pkey, 'name': metadata['track_name'], 'sub': metadata['subclass']})

                    results += 1 # 🎯 Incremento qui per garantire il seed corretto nel flush
                    if len(batch_audio) >= BATCH_SIZE: flush_batch()

                if results % 100 == 0: gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
            flush_batch()
    except Exception:
        if split_emb_dataset_manager: split_emb_dataset_manager.close()
        if own_manager: audio_dataset_manager.close()
        logging.error(f"{traceback.format_exc()}"); return n_embeddings_per_run, False

def process_class_with_cut_secs_slurm(clap_model, audio_embedding, class_to_process, cut_secs, n_octave, config, audio_dataset_manager=None):
    # --- SETUP INIZIALE (Ripristinato Integrale) ---
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

    # 📊 STATISTICHE DI PERFORMANCE (Solo per Rank 0)
    stats = {"load": 0, "prep": 0, "gpu": 0, "save": 0, "count": 0}

    def print_performance_stats():
        if rank == 0 and stats["count"] > 0:
            c = stats["count"]
            print(f"\n[RANK 0 - PERFORMANCE] Media su {c} campioni (secondi):", flush=True)
            print(f"  - I/O Disco:       {stats['load']/c:.4f}s", flush=True)
            print(f"  - Pre-proc (CPU):  {stats['prep']/c:.4f}s", flush=True)
            print(f"  - Inferenza (GPU): {stats['gpu']/c:.4f}s", flush=True)
            print(f"  - Scrittura H5:    {stats['save']/c:.4f}s", flush=True)
            print(f"  - TOTALE:          {(sum(stats.values())-c)/c:.4f}s\n", flush=True)

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
    di = 0; results = 0; round_ = 0; n_embeddings_per_run = 0
    
    class_idx_attr = audio_dataset_manager.hf.attrs.get('class_idx', 0)
    adaptive_buffer_size = max(1, int(100 / cut_secs))
    torch.set_num_threads(1)

    try:
        permuted_indices = audio_dataset_manager.get_reproducible_permutation(class_seed)
        
        while True:
            round_ += 1
            for track_idx in permuted_indices:
                # 🕒 1. LETTURA (I/O)
                t_load_start = time.perf_counter()
                track, metadata = audio_dataset_manager.get_audio_and_metadata(track_idx)
                t_load_end = time.perf_counter()
                
                window_size = int(cut_secs * sr)
                offset = 0
                if round_ > 1 and track.shape[0] > window_size:
                    max_offset = track.shape[0] - window_size
                    if max_offset > 0:
                        offset = offset_rng.integers(0, max_offset)
                n_buckets = math.ceil((track.shape[0] - offset) / window_size)

                for b in range(n_buckets):
                    if results % 10 == 0:
                        diag_print(f"{class_to_process}, {cut_secs} s: created {results}/{target_counts_list[di]}")
                    
                    if results >= target_counts_list[di]:
                        if split_emb_dataset_manager:
                            split_emb_dataset_manager.close()
                            del split_emb_dataset_manager
                            split_emb_dataset_manager = None
                            gc.collect()
                        di += 1
                        if di >= len(division_names):
                            if own_manager: audio_dataset_manager.close()
                            return n_embeddings_per_run, True

                    # 🕒 2. PRE-PROCESSING (CPU)
                    t_prep_start = time.perf_counter()
                    start = b * window_size + offset
                    end = start + window_size
                    cut_data = track[start:end]
                    if len(cut_data) < window_size:
                        cut_data = np.pad(cut_data, (0, window_size - len(cut_data)), 'constant')

                    max_threshold = np.mean(np.abs(cut_data))
                    noise = noise_rng.uniform(-max_threshold, max_threshold, cut_data.shape)
                    new_audio = (1 - noise_perc) * cut_data + noise_perc * noise
                    
                    spec_n_o = spectrogram_n_octaveband_generator(new_audio, sr, n_octave=n_octave, center_freqs=center_freqs, ref=ref)
                    t_prep_end = time.perf_counter()

                    if split_emb_dataset_manager is None:
                        h5_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}_{audio_format}_emb.h5')
                        split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(h5_path, 'a', buffer_size=adaptive_buffer_size)
                        split_emb_dataset_manager.initialize_hdf5(1024, spec_n_o.shape, audio_format, cut_secs, n_octave, sr, seed, noise_perc, division_names[di], class_to_process)

                    emb_pkey = f"{class_idx_attr}_{track_idx}_{b}_{round_}_{results}"
                    if emb_pkey in split_emb_dataset_manager:
                        results += 1; continue

                    # 🕒 3. INFERENZA (GPU)
                    t_gpu_start = time.perf_counter()
                    x = torch.tensor(new_audio, dtype=torch.float32).to(device).unsqueeze(0)
                    with torch.inference_mode():
                       embedding = audio_embedding(x)[0][0]
                    embedding_cpu = embedding.detach().cpu().numpy()
                    t_gpu_end = time.perf_counter()

                    # 🕒 4. SALVATAGGIO + CHECK INTEGRITÀ
                    t_save_start = time.perf_counter()
                    if np.isnan(embedding_cpu).any() or np.all(embedding_cpu == 0):
                        logging.error(f"EMBEDDING CORROTTO rilevato per {metadata['track_name']}!")
                    
                    split_emb_dataset_manager.add_to_data_buffer(embedding_cpu, spec_n_o, emb_pkey, metadata['track_name'], class_to_process, metadata['subclass'])
                    t_save_end = time.perf_counter()

                    # 📊 AGGIORNAMENTO STATS (Solo Rank 0)
                    if rank == 0:
                        stats["load"] += (t_load_end - t_load_start)
                        stats["prep"] += (t_prep_end - t_prep_start)
                        stats["gpu"] += (t_gpu_end - t_gpu_start)
                        stats["save"] += (t_save_end - t_save_start)
                        stats["count"] += 1
                        if stats["count"] % 50 == 0: print_performance_stats()

                    del x, embedding, embedding_cpu, spec_n_o, new_audio, cut_data
                    if torch.cuda.is_available(): torch.cuda.empty_cache()
                    results += 1; n_embeddings_per_run += 1
                
                del track, metadata
                if results % 5 == 0:
                    gc.collect(); trim_memory()

    except Exception:
        if split_emb_dataset_manager: split_emb_dataset_manager.close()
        if own_manager: audio_dataset_manager.close()
        logging.error(f"{traceback.format_exc()}"); return n_embeddings_per_run, False

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

    # 🎯 1. Recupera l'indice classe UNA VOLTA fuori dal loop
    class_idx_attr = audio_dataset_manager.hf.attrs.get('class_idx', 0)
    
    # 🎯 2. BUFFER ADATTIVO: Più lungo è il segmento, più piccolo è il buffer in RAM
    # Se cut_secs = 30, buffer_size = 3. Se cut_secs = 1, buffer_size = 100.
    adaptive_buffer_size = max(1, int(100 / cut_secs))
    
    # 🎯 3. Ottimizzazione Threading (una sola volta)
    torch.set_num_threads(1)

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

                    emb_pkey = f"{class_idx_attr}_{track_idx}_{b}_{round_}_{results}"

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
                        split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(h5_path, 'a', buffer_size=adaptive_buffer_size)
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
                if results % 5 == 0:
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
    Worker process SLURM con inizializzazione collettiva e uscita condizionale per processi vuoti.
    """
    import faulthandler
    faulthandler.enable() # 🎯 Diagnostica per crash di basso livello

    # 🎯 Tutti i rank DEVONO eseguire il setup, indipendentemente dai task
    device = setup_distributed_environment(rank, world_size, slurm=True)

    if not my_tasks:
        print(f"Rank {rank}: nessun task. In attesa degli altri...")
        dist.barrier() # Rimane nel gruppo finché gli altri non finiscono
        return

    # Inizializzazione modello solo per chi ha effettivamente lavoro
    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)

    config['rank'] = rank
    config['dirs']['root_source'] = os.path.join(basedir_raw, f'{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    config['device'] = str(device)
    
    from collections import defaultdict
    tasks_by_class = defaultdict(list)
    for cut_secs, class_name in my_tasks:
        tasks_by_class[class_name].append(cut_secs)

    for class_name, assigned_cuts in tasks_by_class.items():
        h5_path = os.path.join(config['dirs']['root_source'], class_name, f'{class_name}_{audio_format}_dataset.h5')
        
        try:
            current_audio_manager = HDF5DatasetManager(h5_path, audio_format)
            
            for cut_secs in assigned_cuts:
                if rank == 0:
                    logging.info(f"\n[RANK 0] >>> PROCESSO: {class_name} | {cut_secs}s")

                start_time = time.time()
                
                n_embeddings_per_run, completed = process_class_with_cut_secs_slurm_batched(
                    clap_model, audio_embedding, class_name, cut_secs, 
                    n_octave, config, audio_dataset_manager=current_audio_manager
                )
                
                gc.collect() # 🎯 Pulizia memoria post-task
                
                target_log_dir = os.path.join(config['dirs']['root_target'], f'{cut_secs}_secs')
                process_time = time.time() - start_time
                write_log(target_log_dir, (cut_secs, class_name), process_time, n_embeddings_per_run, completed, **config)

                if pbar_instance:
                    pbar_instance.update(1)

            current_audio_manager.close()
            del current_audio_manager # 🎯 Distruzione esplicita manager
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
    Lancia la pipeline su SLURM mantenendo il world_size originale per la stabilità del rendezvous.
    """
    # 1. Recupero parametri SLURM originali (es. 4 processi richiesti)
    rank, world_size = setup_environ_vars(slurm=True)

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

    log_data = {}
    log_path = os.path.join(embed_folder, 'log.json')
    if os.path.exists(log_path):
        try:
            with open(log_path, 'r') as f:
                log_data = json.load(f)
        except Exception as e:
            logging.error(f"Errore lettura log: {e}")

    # 2. Identificazione classi attive
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

    # 🎯 DISTRIBUZIONE TASK: Usiamo il world_size originale di SLURM
    # Se world_size=4 e classi=3, il Rank 3 avrà una lista vuata
    my_assigned_classes = active_classes[rank::world_size]
    my_tasks = []
    for class_name in my_assigned_classes:
        for cut_secs in cut_secs_list:
            log_key_str = str((cut_secs, class_name))
            if not log_data.get(log_key_str, {}).get('completed'):
                my_tasks.append((cut_secs, class_name))

    logging.info(f"Rank {rank}/{world_size}: classi assegnate {my_assigned_classes}")

    # 3. Setup Progress Bar (Solo Rank 0)
    manager = mp.Manager()
    message_queue = manager.Queue() if world_size > 1 else None
    pbar = None
    if rank == 0:
        total_active_tasks = sum(1 for c in active_classes for s in cut_secs_list 
                                 if not log_data.get(str((s, c)), {}).get('completed'))
        if total_active_tasks > 0:
            pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="Progresso Totale SLURM", total=total_active_tasks)

    # 4. Esecuzione Worker
    try:
        # Passiamo il world_size originale per garantire il rendezvous di tutti i processi
        worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, pbar)
    except Exception as e:
        logging.error(f"Errore critico Rank {rank}: {e}")
        raise e
    finally:
        if rank == 0 and pbar:
            pbar.close()
        logging.info(f"Rank {rank}: Fine script.")


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
