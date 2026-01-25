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

# 🎯 PRODUCTION GLOBAL VARIABLE
# Reads from environment to toggle detailed diagnostic prints
VERBOSE = os.environ.get("VERBOSE", "False").lower() == "true"

def process_class_with_cut_secs_slurm_batched(clap_model, audio_embedding, class_to_process, cut_secs, n_octave, config, audio_dataset_manager=None):
    """
    Processes a specific audio class by extracting segments of a fixed length and generating 
    CLAP embeddings and n-octave band spectrograms in a batched, GPU-accelerated manner.
    This function is optimized for high-performance clusters (SLURM), utilizing deterministic 
    noise generation for reproducibility and mixed-precision inference to maximize throughput. 
    It manages dataset splits (train/es/valid/test) and uses a buffered HDF5 writing system 
    to minimize I/O overhead.

    🎯 FIXED: Implements intermediate VRAM cleanup and dynamic batching to prevent OOM 
    on long segments (up to 30s) and high spectral resolution (up to 1/32 octave).

    args:
     - clap_model (msclap.CLAP): The initialized CLAP model wrapper;
     - audio_embedding (torch.nn.Module): The specific sub-model for audio encoding;
     - class_to_process (str): Name of the audio class to process;
     - cut_secs (int/float): Duration of the audio segments in seconds;
     - n_octave (int): Number of bands per octave for spectrogram generation;
     - config (dict): Global configuration dictionary;
     - audio_dataset_manager (HDF5DatasetManager, default: None): Manager for the source audio HDF5.

    returns:
     - n_embeddings_per_run (int): Total number of embeddings successfully generated;
     - success (bool): True if the class was processed entirely, False if an exception occurred.
    """
    # --- INITIAL SETUP ---
    # Retrieve directory paths and audio parameters from configuration
    root_source, root_target = config['dirs']['root_source'], config['dirs']['root_target']
    
    # 🎯 OUTPUT STRUCTURE: Maintained as nested for combine_hdf5_files compatibility
    target_class_dir = os.path.join(root_target, f'{cut_secs}_secs', class_to_process)
    os.makedirs(target_class_dir, exist_ok=True)
    
    audio_format, sr, ref = config['audio']['audio_format'], config['spectrogram']['sr'], config['spectrogram']['ref']
    center_freqs, noise_perc, seed = config['spectrogram']['center_freqs'], config['audio']['noise_perc'], config['audio']['seed']
    device, rank = config['device'], config.get('rank', 0)

    # 📊 PERFORMANCE & MONITORING SETUP
    stats = {"load": 0, "gpu_total": 0, "save": 0, "count": 0}
    
    # 🎯 DYNAMIC BATCH SIZE: Adjusted for both duration AND spectral granularity
    # Memory footprint scales linearly with n_octave. We reduce the batch size to
    # maintain a stable VRAM peak during spectrogram generation.
    base_batch = 128 if cut_secs <= 15 else 64 
    # Scaling factor based on standard 1/3 octave baseline
    BATCH_SIZE = max(8, int(base_batch * (3 / n_octave)))
    
    if VERBOSE and rank == 0:
        print(f"[RANK {rank}] Dynamic Batch set to {BATCH_SIZE} (n_octave={n_octave}, cut={cut_secs}s)", flush=True)
    
    def write_perf_log():
        """Writes performance statistics to a rank-specific text file to keep stdout clean."""
        if stats["count"] > 0:
            c = stats["count"]
            perf_log_path = os.path.join(root_target, f"perf_stats_rank_{rank}.txt")
            with open(perf_log_path, "a") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] {class_to_process} {cut_secs}s | Total: {c} | Avg: {sum(v for k,v in stats.items() if k!='count')/c:.6f}s/audio\n")

    # REPRODUCIBILITY SETUP
    # Use a unique seed per class based on global seed and class name hash
    class_seed = int(seed + hash(class_to_process) % 10000000)
    offset_rng = np.random.default_rng(class_seed)
    
    # Define dataset split boundaries based on proportions provided in config
    division_names = [d[0] for d in config['data']['divisions_xc_sizes_names']]
    target_counts_list = np.cumsum([d[1] for d in config['data']['divisions_xc_sizes_names']])

    # 🎯 SOURCE MANAGER INITIALIZATION (Flat Path Logic)
    # Following the stage-in logic: files are directly in root_source/
    own_manager = False
    if audio_dataset_manager is None:
        h5_raw_path = os.path.join(root_source, f'{class_to_process}_{audio_format}_dataset.h5')
        audio_dataset_manager = HDF5DatasetManager(h5_raw_path, audio_format)
        own_manager = True

    split_emb_dataset_manager = None
    di = 0; results = 0; round_ = 0; n_embeddings_per_run = 0
    
    # Retrieve class index attribute once for the entire processing run
    class_idx_attr = audio_dataset_manager.hf.attrs.get('class_idx', 0)
    
    # BUFFER SIZE ADAPTATION: Smaller buffer for longer segments to conserve VRAM
    adaptive_buffer_size = max(BATCH_SIZE, int(256 / cut_secs))
    torch.set_num_threads(1)

    batch_audio = []; batch_meta = []

    def flush_batch():
        """Moves the current batch to GPU, adds noise, performs inference, and saves results."""
        nonlocal split_emb_dataset_manager, n_embeddings_per_run
        if not batch_audio: return
        
        t_gpu_start = time.perf_counter()
        
        # 🎯 ANTI-OOM: Pre-inference cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Transfer batch to GPU as Float32 and pin memory for faster transfer
        raw_batch = torch.stack(batch_audio).pin_memory().to(device, non_blocking=True).float()
        
        # 🎯 DETERMINISTIC NOISE GENERATION
        # Seed the GPU generator based on the global position within the class
        current_batch_start_idx = results - len(batch_audio)
        g = torch.Generator(device=device).manual_seed(class_seed + current_batch_start_idx)
        
        with torch.inference_mode():
            # Apply uniform noise scaled by the mean absolute amplitude of the batch
            means = torch.mean(torch.abs(raw_batch), dim=1, keepdim=True)
            noise = (torch.rand(raw_batch.shape, generator=g, device=device) * 2 - 1) * means
            batch_tensor = (1 - noise_perc) * raw_batch + noise_perc * noise
            
            # Step 1: Generate Spectrograms (Memory-safe GPU generator)
            specs_gpu = spectrogram_n_octaveband_generator_gpu(batch_tensor, sr, n_octave, 
                                                               center_freqs=center_freqs, ref=ref, device=device)
            
            # 🎯 ANTI-OOM MILESTONE: Intermediate VRAM Cleanup
            # Explicitly delete temporary noise tensors and clear cache before CLAP inference
            del raw_batch, noise, means
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Step 2: Perform CLAP Inference in Mixed Precision (FP16)
            with torch.cuda.amp.autocast():
                output = audio_embedding(batch_tensor)
                embeddings = output[0] if isinstance(output, (tuple, list)) else output
                if embeddings.dim() > 2: embeddings = embeddings.squeeze(1)

        # Transfer results back to CPU for HDF5 storage
        embeddings_cpu = embeddings.float().cpu().numpy()
        specs_cpu = specs_gpu.float().cpu().numpy()
        stats["gpu_total"] += (time.perf_counter() - t_gpu_start)

        # 🎯 ANTI-OOM: Immediate deletion of GPU tensors
        del batch_tensor, specs_gpu, embeddings
        
        t_save_start = time.perf_counter()
        for i in range(len(embeddings_cpu)):
            # Initialize or update the split-specific HDF5 manager
            if split_emb_dataset_manager is None:
                h5_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}_{audio_format}_emb.h5')
                split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(h5_path, 'a', buffer_size=adaptive_buffer_size)
                split_emb_dataset_manager.initialize_hdf5(1024, specs_cpu[i].shape, audio_format, cut_secs, n_octave, 
                                                          sr, seed, noise_perc, division_names[di], class_to_process)

            # Add data to the manager's buffer with Resumability support
            split_emb_dataset_manager.add_to_data_buffer(embeddings_cpu[i], specs_cpu[i], batch_meta[i]['pkey'], 
                                                        batch_meta[i]['name'], class_to_process, batch_meta[i]['sub'])
        
        stats["save"] += (time.perf_counter() - t_save_start)
        cur_batch_len = len(batch_audio)
        stats["count"] += cur_batch_len
        n_embeddings_per_run += cur_batch_len
        
        # 🎯 EXECUTION HEARTBEAT
        print(f"[RANK {rank}] {class_to_process} | {cut_secs}s: {results}/{target_counts_list[di]} ({division_names[di]})", flush=True)

        if stats["count"] % (BATCH_SIZE * 4) == 0: 
            write_perf_log()
            
        batch_audio.clear(); batch_meta.clear()
        
        # 🎯 ANTI-OOM: Post-batch cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # --- MAIN PROCESSING LOOP ---
    try:
        # Generate a reproducible permutation of indices to ensure consistent data access across runs
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
                    
                    # 🎯 RESUMABILITY CHECK: Skip if already exists
                    if split_emb_dataset_manager and emb_pkey in split_emb_dataset_manager:
                        results += 1; continue

                    start, end = b*window_size+offset, (b+1)*window_size+offset
                    cut_data = track[start:end]
                    if len(cut_data) < window_size: 
                        cut_data = np.pad(cut_data, (0, window_size - len(cut_data)), 'constant')
                    
                    batch_audio.append(torch.from_numpy(cut_data).float())
                    batch_meta.append({'pkey': emb_pkey, 'name': metadata['track_name'], 'sub': metadata['subclass']})

                    results += 1
                    if len(batch_audio) >= BATCH_SIZE: flush_batch()

                if results % 100 == 0: 
                    gc.collect(); ctypes.CDLL('libc.so.6').malloc_trim(0)
            flush_batch()
            
    except Exception:
        if split_emb_dataset_manager: split_emb_dataset_manager.close()
        if own_manager: audio_dataset_manager.close()
        logging.error(f"❌ CRITICAL ERROR: {traceback.format_exc()}")
        return n_embeddings_per_run, False

def process_class_with_cut_secs(clap_model, audio_embedding, class_to_process, cut_secs, n_octave, config, audio_dataset_manager=None):
    """
    Processes a specific audio class by extracting segments of a fixed length in a sequential 
    (non-batched) manner. This version is designed for environments where memory isolation 
    per audio clip is critical, featuring an adaptive buffer system and aggressive memory 
    cleanup (RAM and VRAM) after each processed segment.

    The function handles deterministic noise generation for reproducibility, generates 
    n-octave band spectrograms on the CPU, and performs CLAP embedding inference on the 
    specified device.

    args:
     - clap_model (msclap.CLAP): The initialized CLAP model wrapper;
     - audio_embedding (torch.nn.Module): The specific sub-model for audio encoding;
     - class_to_process (str): Name of the audio class to process;
     - cut_secs (int/float): Duration of the audio segments in seconds;
     - n_octave (int): Number of bands per octave for spectrogram generation;
     - config (dict): Global configuration dictionary containing directory paths, audio 
                      parameters, and dataset split sizes;
     - audio_dataset_manager (HDF5DatasetManager, default: None): Manager for the source 
                      audio HDF5. If None, it will be instantiated locally.

    returns:
     - n_embeddings_per_run (int): Total number of embeddings successfully generated in this session;
     - success (bool): True if the class was processed entirely, False if an exception occurred.
    """
    # --- INITIAL SETUP ---
    # Retrieve directory paths and processing parameters from the config
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

    # 🎯 VERBOSITY CONTROL
    # diag_print only outputs to stdout if the global VERBOSE flag is True
    def diag_print(msg):
        if VERBOSE:
            print(f"[RANK {rank} - DIAG] {msg}", flush=True)

    def trim_memory():
        """Aggressively release free memory back to the OS system."""
        try:
            ctypes.CDLL('libc.so.6').malloc_trim(0)
        except Exception:
            pass

    # REPRODUCIBILITY SETUP
    # Unique seeds for augmentation and noise based on the class hash
    class_seed = seed + hash(class_to_process) % 10000000
    offset_rng = np.random.default_rng(class_seed)
    noise_rng = np.random.default_rng(class_seed)

    division_names = [d[0] for d in config['data']['divisions_xc_sizes_names']]
    target_counts_list = np.cumsum([d[1] for d in config['data']['divisions_xc_sizes_names']])

    # SOURCE DATA MANAGER INITIALIZATION
    own_manager = False
    if audio_dataset_manager is None:
        audio_dataset_manager = HDF5DatasetManager(os.path.join(root_source, f'{class_to_process}_{audio_format}_dataset.h5'))
        own_manager = True

    split_emb_dataset_manager = None
    di = 0
    results = 0
    round_ = 0
    n_embeddings_per_run = 0

    # 🎯 OPTIMIZATION 1: Retrieve class index once outside the main loops
    class_idx_attr = audio_dataset_manager.hf.attrs.get('class_idx', 0)
    
    # 🎯 OPTIMIZATION 2: ADAPTIVE BUFFER
    # Dynamically scales the HDF5 writing buffer based on segment duration to manage RAM usage
    adaptive_buffer_size = max(1, int(100 / cut_secs))
    
    # 🎯 OPTIMIZATION 3: Threading control for parallel worker stability
    torch.set_num_threads(1)

    try:
        # Get a reproducible permutation of track indices for the source dataset
        permuted_indices = audio_dataset_manager.get_reproducible_permutation(class_seed)
        
        while True:
            round_ += 1
            for track_idx in permuted_indices:
                # Load audio track and metadata directly from HDF5
                track, metadata = audio_dataset_manager.get_audio_and_metadata(track_idx)
                
                window_size = int(cut_secs * sr)
                offset = 0
                # Apply random offset for augmentation in rounds greater than 1
                if round_ > 1 and track.shape[0] > window_size:
                    max_offset = track.shape[0] - window_size
                    if max_offset > 0:
                        offset = offset_rng.integers(0, max_offset)
                n_buckets = math.ceil((track.shape[0] - offset) / window_size)

                for b in range(n_buckets):
                    # 🎯 STATUS MONITORING: Heartbeat printed every 10 embeddings
                    if results % 10 == 0:
                        print(f"[RANK {rank}] {class_to_process} | {cut_secs}s: {results}/{target_counts_list[di]} ({division_names[di]})", flush=True)
                    
                    # Manage transition between dataset splits (e.g., train -> es)
                    if results >= target_counts_list[di]:
                        diag_print(f"Split '{division_names[di]}' completed. Starting flush...")
                        
                        if split_emb_dataset_manager:
                            split_emb_dataset_manager.close()
                            del split_emb_dataset_manager
                            split_emb_dataset_manager = None
                            gc.collect()
                            time.sleep(0.5)
                        
                        di += 1
                        if di >= len(division_names):
                            if own_manager: audio_dataset_manager.close()
                            return n_embeddings_per_run, True

                    # Generate unique ID for resumability check
                    emb_pkey = f"{class_idx_attr}_{track_idx}_{b}_{round_}_{results}"

                    # Slice audio and apply padding if the segment is too short
                    start = b * window_size + offset
                    end = start + window_size
                    cut_data = track[start:end]
                    if len(cut_data) < window_size:
                        cut_data = np.pad(cut_data, (0, window_size - len(cut_data)), 'constant')

                    # 🎯 AUGMENTATION: Add deterministic uniform noise
                    max_threshold = np.mean(np.abs(cut_data))
                    noise = noise_rng.uniform(-max_threshold, max_threshold, cut_data.shape)
                    new_audio = (1 - noise_perc) * cut_data + noise_perc * noise
                    
                    # Generate spectrogram using CPU generator
                    spec_n_o = spectrogram_n_octaveband_generator(new_audio, sr, integration_seconds=0.1,
                                                    n_octave=n_octave, center_freqs=center_freqs, ref=ref)

                    # Initialize split-specific HDF5 manager if required
                    if split_emb_dataset_manager is None:
                        h5_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}_{audio_format}_emb.h5')
                        split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(h5_path, 'a', buffer_size=adaptive_buffer_size)
                        split_emb_dataset_manager.initialize_hdf5(
                            1024, spec_n_o.shape, audio_format, cut_secs, n_octave, 
                            sr, seed, noise_perc, division_names[di], class_to_process
                        )

                    # Skip if result already exists (resumability check)
                    if emb_pkey in split_emb_dataset_manager:
                        results += 1
                        continue

                    # --- INFERENCE ---
                    x = torch.tensor(new_audio, dtype=torch.float32).to(device).unsqueeze(0)
                    with torch.inference_mode():
                       # Extract the specific audio embedding tensor
                       embedding = audio_embedding(x)[0][0]
                    
                    embedding_cpu = embedding.detach().cpu().numpy()
                    del x, embedding # Immediate deletion to free VRAM

                    # Write results to the HDF5 manager buffer
                    split_emb_dataset_manager.add_to_data_buffer(embedding_cpu, spec_n_o, emb_pkey,
                                    metadata['track_name'], class_to_process, metadata['subclass'])

                    # 🎯 TOTAL CLEANUP: Prevent VRAM/RAM fragmentation
                    del embedding_cpu, spec_n_o, new_audio, cut_data
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    torch.set_num_threads(1)
                    
                    results += 1
                    n_embeddings_per_run += 1
                
                # 🎯 Periodic memory cleanup after each full track
                del track, metadata
                if results % 5 == 0:
                    gc.collect()
                    trim_memory()

    except Exception:
        # Emergency closure of managers on failure
        if split_emb_dataset_manager:
            split_emb_dataset_manager.close()
        if own_manager: audio_dataset_manager.close()
        logging.error(f"{traceback.format_exc()}")
        return n_embeddings_per_run, False

### Workers ###

def worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, pbar_instance=None, device=None):
    """
    Main worker function for SLURM environments. It orchestrates model setup 
    and task execution for a specific rank. It ensures that each process works 
    on a unique subset of class/cut_secs pairs to maximize parallel efficiency.

    args:
     - audio_format (str): The audio format (e.g., 'wav');
     - n_octave (int): The octave band resolution for spectrograms;
     - config (dict): Global configuration dictionary;
     - rank (int): The unique identifier for the current process/GPU;
     - world_size (int): Total number of processes in the distributed group;
     - my_tasks (list): List of (cut_secs, class_name) tuples assigned to this rank;
     - pbar_instance (MultiProcessTqdm, default: None): Shared progress bar;
     - device (torch.device, optional): Pre-initialized device for this rank.
    """
    import faulthandler
    faulthandler.enable() # Diagnostic tool for low-level crash debugging

    # --- ENVIRONMENT INITIALIZATION ---
    # 🎯 FIX: setup_distributed_environment is NOT called here as it is pre-handled
    if device is None:
        device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')

    if not my_tasks:
        print(f"⚠️ [RANK {rank}] No tasks assigned. Waiting at barrier...", flush=True)
        dist.barrier() # Ensure even idle ranks wait for the group
        return

    # 📊 INITIAL STATUS DECLARATION
    print(f"🚀 [RANK {rank}] Worker started on {device}. Tasks assigned: {len(my_tasks)}", flush=True)

    # Initialize CLAP models on the assigned GPU
    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)

    # Update local config with process-specific paths and device info
    config['rank'] = rank
    # Use basedir_raw/preprocessed from dirs_config (respects NODE_TEMP_BASE_DIR)
    config['dirs']['root_source'] = os.path.join(basedir_raw, f'raw_{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    config['device'] = str(device)
    
    # Organize tasks by class to minimize opening/closing the source HDF5 manager
    from collections import defaultdict
    tasks_by_class = defaultdict(list)
    for cut_secs, class_name in my_tasks:
        tasks_by_class[class_name].append(cut_secs)

    # --- MAIN TASK LOOP ---
    for class_idx, (class_name, assigned_cuts) in enumerate(tasks_by_class.items()):
        # 🎯 FLAT SOURCE LOGIC: Raw files are directly under the format folder
        h5_path = os.path.join(config['dirs']['root_source'], f'{class_name}_{audio_format}_dataset.h5')
        
        try:
            # Inform about the current class being processed
            print(f"📦 [RANK {rank}] Processing Class {class_idx+1}/{len(tasks_by_class)}: {class_name}", flush=True)
            
            # Open the source audio manager for the current class
            current_audio_manager = HDF5DatasetManager(h5_path, audio_format)
            
            for cut_secs in assigned_cuts:
                # Segment-specific duration feedback
                print(f"🔹 [RANK {rank}] Current Task: {class_name} | {cut_secs}s", flush=True)

                start_time = time.time()
                
                # Execute the batched processing logic with anti-OOM measures
                n_embeddings_per_run, completed = process_class_with_cut_secs_slurm_batched(
                    clap_model, audio_embedding, class_name, cut_secs, 
                    n_octave, config, audio_dataset_manager=current_audio_manager
                )
                
                # Immediate garbage collection after a segment to free VRAM
                gc.collect()
                
                # 🎯 LOG HIERARCHY FIX: Write logs to the duration-specific subfolder
                target_log_dir = os.path.join(config['dirs']['root_target'], f'{cut_secs}_secs')
                process_time = time.time() - start_time
                write_log(target_log_dir, (cut_secs, class_name), process_time, n_embeddings_per_run, completed, **config)

                # Update the shared progress bar if provided
                if pbar_instance:
                    pbar_instance.update(1)

            # Cleanup class-specific resources
            current_audio_manager.close()
            del current_audio_manager
            gc.collect()
            
            print(f"✅ [RANK {rank}] Completed class: {class_name}", flush=True)
            
        except Exception as e:
            # Log errors without stopping the whole rank execution
            logging.error(f"❌ [RANK {rank}] Error in class {class_name}: {traceback.format_exc()}")
            continue

    # --- FINAL CLEANUP ---
    print(f"🏁 [RANK {rank}] All assigned tasks completed successfully.", flush=True)
    cleanup_distributed_environment(rank)


def local_worker_process(audio_format, n_octave, config, rank, world_size, my_tasks, pbar_instance=None):
    """
    Worker function for local multi-processing execution.

    args:
     - audio_format (str): The audio format (e.g., 'wav');
     - n_octave (int): Octave band resolution;
     - config (dict): Global configuration dictionary;
     - rank (int): Local process index;
     - world_size (int): Total number of local processes;
     - my_tasks (list): List of (cut_secs, class_name) tuples;
     - pbar_instance (MultiProcessTqdm): Shared progress bar.
    """
    import faulthandler
    faulthandler.enable()

    # --- ENVIRONMENT SETUP ---
    # Initialize the local distributed group (using 'gloo' backend)
    device = setup_distributed_environment(rank, world_size, False)
    
    # Initialize CLAP models on the local device
    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)
    
    # Configure local paths and device strings
    config['rank'] = rank
    config['dirs']['root_source'] = os.path.join(basedir_raw, f'raw_{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    config['device'] = str(device)

    # Group tasks by class name
    from collections import defaultdict
    tasks_by_class = defaultdict(list)
    for cut_secs, class_name in my_tasks:
        tasks_by_class[class_name].append(cut_secs)

    print(f"🚀 [LOCAL RANK {rank}] Started. Classes assigned: {len(tasks_by_class)}", flush=True)

    # --- PROCESSING LOOP ---
    for class_idx, (class_name, assigned_cuts) in enumerate(tasks_by_class.items()):
        h5_path = os.path.join(config['dirs']['root_source'], f'{class_name}_{audio_format}_dataset.h5')
        
        try:
            print(f"📦 [LOCAL RANK {rank}] Task {class_idx+1}/{len(tasks_by_class)}: {class_name}", flush=True)
            
            # Open the source audio manager
            current_audio_manager = HDF5DatasetManager(h5_path, audio_format)
            
            for cut_secs in assigned_cuts:
                print(f"🔹 [LOCAL RANK {rank}] Processing: {class_name} | {cut_secs}s", flush=True)

                start_time = time.time()
        
                # Execute non-batched processing with adaptive buffering
                n_embeddings_per_run, completed = process_class_with_cut_secs(
                    clap_model, audio_embedding, class_name, cut_secs, 
                    n_octave, config, audio_dataset_manager=current_audio_manager
                )
                
                gc.collect()
        
                # 🎯 LOG HIERARCHY FIX: Write to duration subfolder
                target_log_dir = os.path.join(config['dirs']['root_target'], f'{cut_secs}_secs')
                process_time = time.time() - start_time
                write_log(target_log_dir, (cut_secs, class_name), process_time, n_embeddings_per_run, completed, **config)
        
                if pbar_instance:
                    pbar_instance.update(1)
            
            current_audio_manager.close()
            del current_audio_manager
            gc.collect()

        except Exception as e:
            logging.error(f"❌ [LOCAL RANK {rank}] Critical error on {class_name}: {traceback.format_exc()}")
            continue

    print(f"🏁 [LOCAL RANK {rank}] Job finished.", flush=True)
    cleanup_distributed_environment(rank)


### Executions ###

def run_distributed_slurm(config_file, audio_format, n_octave):
    """
    Orchestrates the distributed embedding pipeline specifically for SLURM environments.
    It manages the global setup, identifies incomplete tasks by inspecting existing logs, 
    and assigns a deterministic subset of work to the current rank. Initializes the
    distributed group IMMEDIATELY to prevent desynchronization timeouts. Uses a single-pass
    log cache to minimize filesystem metadata latency on Lustre.

    args:
     - config_file (str): Name of the YAML configuration file to load;
     - audio_format (str): Audio format to process (e.g., 'wav', 'mp3');
     - n_octave (int): Octave band resolution for the spectrograms.
    """
    # 1. IMMEDIATE ENVIRONMENT INITIALIZATION
    # Initializes rank and world_size based on SLURM environmental variables.
    # We setup the world before hitting the filesystem to ensure all ranks are synchronized.
    rank, world_size = setup_environ_vars(slurm=True)
    device = setup_distributed_environment(rank, world_size, slurm=True)

    # 2. Path and Logging Setup
    # Create the target directory and initialize the logging system
    # Path: <preprocessed_dir>/<format>/<n_octave>_octave/
    embed_folder = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    os.makedirs(embed_folder, exist_ok=True)
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(filename=os.path.join(embed_folder, 'log.txt'))])

    # Load configuration parameters from the YAML file
    classes_list, _, _, _, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
        divisions_xc_sizes_names = get_config_from_yaml(config_file)

    # Build the configuration object for workers
    config = {
        'dirs': {}, 'audio': {}, 'spectrogram': {}, 'log': {}, 'data': {
            'divisions_xc_sizes_names': divisions_xc_sizes_names
        }, 'device': str(device), 'rank': rank
    }
    config['spectrogram'].update({'sr': sampling_rate, 'ref': ref, 'center_freqs': center_freqs})
    config['audio'].update({'noise_perc': noise_perc, 'seed': seed})

    # 3. SINGLE-PASS TASK DISCOVERY (I/O Optimization)
    # 🎯 FIX: We open each duration-specific log only ONCE to bypass metadata bottleneck on Lustre.
    task_status_cache = {} 
    
    for cut_secs in cut_secs_list:
        # According to the Sacred Mantra: logs are in the duration subdirectory
        duration_log_path = os.path.join(embed_folder, f'{cut_secs}_secs', 'log.json')
        log_data = {}
        if os.path.exists(duration_log_path):
            try:
                with open(duration_log_path, 'r') as f:
                    log_data = json.load(f)
            except Exception:
                pass # Treat as empty if corrupted to ensure re-processing
        
        # Pre-populate cache for all classes for this specific duration
        for class_name in classes_list:
            log_key = str((cut_secs, class_name))
            task_status_cache[(cut_secs, class_name)] = log_data.get(log_key, {}).get('completed', False)

    # 4. Resumability Check: Identify active classes and assigned tasks
    # A class is active if at least one of its durations needs work
    active_classes = [c for c in classes_list if any(not task_status_cache[(s, c)] for s in cut_secs_list)]
    active_classes.sort()

    # 🎯 DETERMINISTIC TASK DISTRIBUTION
    # Assign classes to the current rank using round-robin slicing
    my_assigned_classes = active_classes[rank::world_size]
    my_tasks = [(s, c) for c in my_assigned_classes for s in cut_secs_list if not task_status_cache[(s, c)]]

    # Log task distribution for the current rank
    if VERBOSE:
        logging.info(f"Rank {rank}/{world_size}: Assigned classes {my_assigned_classes}")

    # Barrier: ensure all ranks finished their checks before starting the worker
    dist.barrier()

    # 5. Global Progress Monitoring (Rank 0 only)
    manager = mp.Manager()
    message_queue = manager.Queue() if world_size > 1 else None
    pbar = None
    if rank == 0:
        # Fast calculation of total pending tasks across all duration subdirectories using the cache
        total_active_tasks = sum(1 for c in active_classes for s in cut_secs_list if not task_status_cache[(s, c)])
        
        if total_active_tasks > 0:
            pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="SLURM Global Progress", total=total_active_tasks)

    # 6. Worker Dispatch (Pass pre-initialized device)
    try:
        # 🎯 FIX: setup_distributed_environment is NOT called again inside the worker
        worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, pbar, device=device)
    except Exception as e:
        logging.error(f"Critical error on Rank {rank}: {traceback.format_exc()}")
        raise e
    finally:
        if rank == 0 and pbar:
            pbar.close()

def run_local_multiprocess(config_file, audio_format, n_octave, world_size):
    """
    Executes the embedding pipeline on a local machine by spawning multiple processes. 
    It dynamically adjusts the number of processes (world_size) based on the number 
    of active classes to prevent rendezvous deadlocks. Aligned log search logic with
    duration-specific subdirectory structure and implemented single-pass log caching.

    args:
     - config_file (str): Name of the YAML configuration file;
     - audio_format (str): Audio format to process;
     - n_octave (int): Octave band resolution for spectrograms;
     - world_size (int): Requested number of parallel processes.
    """
    # 1. Local Environment Initialization
    setup_environ_vars(slurm=False)
    
    import random
    os.environ['MASTER_PORT'] = str(random.randint(29500, 29999))

    embed_folder = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    os.makedirs(embed_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.StreamHandler(),
                                  logging.FileHandler(filename=os.path.join(embed_folder, 'log.txt'))])

    # 2. Configuration and Log Loading
    classes_list, _, _, _, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
        divisions_xc_sizes_names = get_config_from_yaml(config_file)

    config = {
        'dirs': {}, 'audio': {}, 'spectrogram': {}, 'log': {}, 'data': {
            'divisions_xc_sizes_names': divisions_xc_sizes_names
        }
    }
    config['spectrogram'].update({'sr': sampling_rate, 'ref': ref, 'center_freqs': center_freqs})
    config['audio'].update({'noise_perc': noise_perc, 'seed': seed})

    # 3. SINGLE-PASS LOG CACHING (Optimization)
    # Corrected log search path for local resumability within duration subfolders
    task_status_cache = {}
    for cut_secs in cut_secs_list:
        duration_log_path = os.path.join(embed_folder, f'{cut_secs}_secs', 'log.json')
        log_data = {}
        if os.path.exists(duration_log_path):
            try:
                with open(duration_log_path, 'r') as f:
                    log_data = json.load(f)
            except Exception:
                pass
        for class_name in classes_list:
            task_status_cache[(cut_secs, class_name)] = log_data.get(str((cut_secs, class_name)), {}).get('completed', False)

    # 4. Active Class Identification
    active_classes = [c for c in classes_list if any(not task_status_cache[(s, c)] for s in cut_secs_list)]
    active_classes.sort()

    # 🎯 DYNAMIC WORLD SIZE ADJUSTMENT
    actual_world_size = min(len(active_classes), world_size)
    if actual_world_size == 0:
        logging.info("All tasks are already completed according to duration-specific logs.")
        return

    print(f"Local Environment: {len(active_classes)} active classes. Starting with {actual_world_size} processes.")

    # 5. Task Distribution
    processes = []; manager = mp.Manager(); message_queue = manager.Queue()
    tasks_distribution = []; total_tasks_to_run = 0

    for rank in range(actual_world_size):
        # Round-robin class assignment and cached task filtering
        my_rank_tasks = [(s, c) for c in active_classes[rank::actual_world_size] 
                         for s in cut_secs_list if not task_status_cache[(s, c)]]
        
        if my_rank_tasks:
            tasks_distribution.append((rank, my_rank_tasks))
            total_tasks_to_run += len(my_rank_tasks)

    # 6. Local Progress Monitoring
    pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="Local Progress", total=total_tasks_to_run) if total_tasks_to_run > 0 else None

    # 7. Process Spawning
    try:
        for rank, my_tasks in tasks_distribution:
            p = mp.Process(
                target=local_worker_process, 
                args=(audio_format, n_octave, config, rank, actual_world_size, my_tasks, pbar)
            )
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
            
    except Exception as e:
        logging.error(f"Critical error in local parent process: {traceback.format_exc()}")
        raise e
    finally:
        if pbar:
            pbar.close()
