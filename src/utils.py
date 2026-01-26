import os
import json
import yaml
import glob
import logging
import gc
import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import traceback

# 🎯 PRODUCTION GLOBAL VARIABLE
# Reads from environment to toggle detailed diagnostic prints
VERBOSE = os.environ.get("VERBOSE", "False").lower() == "true"

__all__ = [
           "get_config_from_yaml",

           "write_log",
           "join_logs",

           "HDF5DatasetManager",
           "HDF5EmbeddingDatasetsManager",
           "combine_hdf5_files",

           "get_track_reproducibility_parameters",
           "reconstruct_tracks_from_embeddings",
           "TorchEmbeddingDataset",
           "load_single_cut_secs_dataloaders",

           "setup_environ_vars",
           "setup_distributed_environment",
           "cleanup_distributed_environment"
          ]

def get_config_from_yaml(config_file="config0.yaml"):
    """
    Loads model, training, and spectrogram configurations from a YAML file 
    located in the 'configs' directory.

    args:
     - config_file (str, default: 'config0.yaml'): Name of the YAML configuration 
                      file to be loaded.

    returns:
     - classes (list): List of audio class names;
     - patience (int): Early stopping patience;
     - epochs (int): Maximum training epochs;
     - batch_size (int): Size of training batches;
     - sampling_rate (int): Audio sampling rate;
     - ref (float): Reference value for spectrogram decibel conversion;
     - noise_perc (float): Percentage of noise to add for augmentation;
     - seed (int): Global random seed;
     - center_freqs (np.array/None): Predefined center frequencies for octave bands;
     - valid_cut_secs (list): Durations of audio segments in seconds;
     - splits_xc_sizes_names (list): List of tuples containing split names and sizes.
    """
    config_path = os.path.join('configs', config_file)
    with open(config_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
    
    classes = configs["classes"]
    patience = configs["patience"]
    epochs = configs["epochs"]
    batch_size = configs["batch_size"]
    sampling_rate = configs["sampling_rate"]
    ref = configs["ref"]
    noise_perc = configs["noise_perc"]
    seed = configs["seed"]
    
    # center_freqs to be removed completely at some point
    center_freqs = np.array(configs["center_freqs"]) if configs.get("center_freqs") else None
    
    valid_cut_secs = configs["valid_cut_secs"]
    splits_xc_sizes_names = [("train", configs["train_size"]),
                             ("es", configs["es_size"]),
                             ("valid", configs["valid_size"]),
                             ("test", configs["test_size"])]
    
    return classes, patience, epochs, batch_size, sampling_rate, ref, noise_perc, seed, \
           center_freqs, valid_cut_secs, splits_xc_sizes_names


def write_log(log_path, new_cut_secs_class, process_time, n_embeddings_per_run, completed, **kwargs):
    """
    Writes or updates a rank-specific JSON log file for tracking the embedding 
    process of a specific (duration, class) pair. It ensures atomic writes 
    using fsync to prevent data corruption on cluster filesystems.

    args:
     - log_path (str): Directory where the log file will be stored;
     - new_cut_secs_class (str/tuple): The (cut_secs, class_name) identifier;
     - process_time (float): Time taken to process the current task;
     - n_embeddings_per_run (int): Number of embeddings generated in this run;
     - completed (bool): Whether the task was finished entirely;
     - **kwargs: Configuration parameters and rank info to store in the log.

    returns:
     - None: The log is written directly to disk.
    """
    os.makedirs(log_path, exist_ok=True)
    
    new_cut_secs_class = str(new_cut_secs_class)
    # Each rank writes its own file to avoid I/O contention
    logfile = os.path.join(log_path, f"log_rank_{kwargs['rank']}.json")
    
    log = {"config": {}}
    if os.path.exists(logfile):
        try:
            with open(logfile, 'r') as f:
                log = json.load(f)
        except (json.JSONDecodeError, Exception):
            pass

    if not log.get("config"): 
        log["config"].update(kwargs)

    # Use the identifier as key to allow multiple classes in the same rank log
    if new_cut_secs_class not in log:
        log[new_cut_secs_class] = {"process_time": [], "n_embeddings_per_run": [], "rank": []}
        
    log[new_cut_secs_class]["process_time"].append(process_time)
    log[new_cut_secs_class]["n_embeddings_per_run"].append(n_embeddings_per_run)
    log[new_cut_secs_class]["rank"].append(kwargs["rank"])
    log[new_cut_secs_class]["completed"] = completed

    # ATOMIC WRITE: Ensure data is physically written to disk
    with open(logfile, 'w') as f:
        json.dump(log, f, indent=4)
        f.flush() 
        os.fsync(f.fileno()) 


def join_logs(log_dir):
    """
    Merges all rank-specific JSON logs (log_rank_*.json) found in a directory 
    into a single master 'log.json' file. It aggregates statistics for 
    partially completed classes across different ranks. If a joint log is
    already present, merges its information with the one from rank logs.

    args:
     - log_dir (str): Path to the directory containing individual rank logs.

    returns:
     - None: Deletes individual rank logs after merging into the final file.
    """
    final_log_file = os.path.join(log_dir, "log.json")
    final_log = {"config": {}}
    
    # 1. LOAD PRE-EXISTING LOG (if any)
    if os.path.exists(final_log_file):
        try:
            with open(final_log_file, 'r') as f:
                final_log = json.load(f)
        except Exception as e:
            print(f"⚠️ Warning: Could not load existing master log: {e}")

    pattern = os.path.join(log_dir, "log_rank_*.json")
    log_files = glob.glob(pattern)
    
    if not log_files:
        return

    # 2. MERGE NEW RANK LOGS
    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                current_rank_log = json.load(f)
                
                for key, data in current_rank_log.items():
                    if key == "config":
                        if not final_log["config"]:
                            final_log["config"].update(data)
                        continue

                    # INCREMENTAL MERGE LOGIC
                    if key in final_log:
                        # Extend historical lists with new run data
                        for field in ["process_time", "n_embeddings_per_run", "rank"]:
                            if field in data:
                                final_log[key][field].extend(data[field])
                        # Task is completed if it was already True OR if the new run finished it
                        final_log[key]["completed"] = final_log[key].get("completed", False) or data.get("completed", False)
                    else:
                        final_log[key] = data
                        
        except Exception as e:
            print(f"Error merging log {log_file}: {e}")

    # 3. ATOMIC SAVE
    with open(final_log_file, 'w') as f:
        json.dump(final_log, f, indent=4)
        
    for log_file in log_files:
        os.remove(log_file)

### hdf5 raw dataset class ###

class HDF5DatasetManager:
    """
    Manages high-speed access to raw audio data stored in a primary HDF5 "Super-Dataset". 
    It implements lazy loading and direct index-based retrieval to minimize memory overhead 
    by avoiding the use of intermediate data structures like Pandas for metadata.

    args:
     - h5_file_path (str): Absolute path to the source HDF5 file;
     - audio_format (str, default: 'wav'): The audio extension used to identify datasets 
                      within the HDF5 structure.

    returns:
     - Instance: A manager object with an open file handle and metadata descriptors.
    """
    
    def __init__(self, h5_file_path: str, audio_format: str = 'wav'):
        self.h5_file_path = h5_file_path
        self.hf = None
        self.audio_format = audio_format
        self.metadata_dset_name = f'metadata_{self.audio_format}'
        self.audio_dset_name = f'audio_{self.audio_format}'
        self.n_records = 0 # Initialized via lazy evaluation upon file opening
        
        try:
            # Open HDF5 with rdcc_nbytes=0 to bypass the internal chunk cache, 
            # as our access pattern is sequential/permuted but not repetitive.
            self.hf = h5py.File(self.h5_file_path, 'r', rdcc_nbytes=0)
            self.n_records = self.hf[self.metadata_dset_name].shape[0]
            logging.info(f"HDF5 Manager Ready: {self.n_records} tracks detected.")
        except Exception as e:
            logging.error(f"Error opening {h5_file_path}: {e}")
            raise

    def get_audio_and_metadata(self, hdf5_index):
        """
        Retrieves a single audio waveform and its associated metadata dictionary 
        directly from the HDF5 datasets using atomic index slicing.

        args:
         - hdf5_index (int): The numerical index of the record to retrieve.

        returns:
         - audio (np.ndarray): The raw audio signal;
         - meta_dict (dict): Decoded metadata fields (e.g., track name, subclass).
        """
        audio = self.hf[self.audio_dset_name][hdf5_index]
        # Retrieve row as a numpy void object (structured array)
        raw_meta = self.hf[self.metadata_dset_name][hdf5_index]
        # Decode bytes to strings to ensure compatibility with downstream logic
        meta_dict = {n: (raw_meta[n].decode('utf-8') if isinstance(raw_meta[n], bytes) else raw_meta[n]) 
                     for n in raw_meta.dtype.names}
        return audio, meta_dict

    def get_reproducible_permutation(self, seed: int):
        """
        Generates a deterministic permutation of the dataset indices.

        args:
         - seed (int): The seed for the random number generator.

        returns:
         - (np.ndarray): A shuffled array of indices from 0 to n_records - 1.
        """
        rng = np.random.default_rng(seed)
        return rng.permutation(np.arange(self.n_records))

    def close(self):
        """Safely closes the HDF5 file handle and triggers garbage collection."""
        if self.hf:
            self.hf.close()
            self.hf = None
        gc.collect()


class HDF5EmbeddingDatasetsManager(Dataset):
    """
    An advanced HDF5 manager designed for writing and reading audio embeddings 
    and spectrograms. It features an O(1) in-memory key lookup for resumability, 
    adaptive buffering to optimize disk I/O, and structured metadata handling.

    args:
     - h5_path (str): Path to the target embedding HDF5 file;
     - mode (str, default: 'r'): File access mode ('r', 'w', 'a');
     - partitions (set, default: {'classes', 'splits'}): Controls the internal 
                      dtype structure of the dataset;
     - buffer_size (int, default: 10): Number of records to hold in RAM before flushing 
                      to disk.
    """
    def __init__(self, h5_path, mode='r', partitions=set(('classes', 'splits')), buffer_size=10):
        super().__init__()
        self.h5_path = h5_path
        self.partitions = set(partitions)
        self.mode = mode
        self.hf = None
        self.buffer_size = buffer_size
        self.buffer_count = 0
        self.existing_keys = set() # For O(1) latency checks during resumption
        self.dt = None           
        self.buffer_array = None 
        
        try:
            self.hf = h5py.File(self.h5_path, self.mode, rdcc_nbytes=0)
        except Exception as e:
            logging.error(f"Could not open file {h5_path}: {e}")
            raise

        # 🎯 METADATA & KEY RECOVERY (Resumability Logic)
        if self.hf is not None and 'embedding_dataset' in self.hf:
            try:
                # Recover structural parameters from HDF5 attributes
                emb_dim = self.hf.attrs.get('embedding_dim')
                spec_sh = self.hf.attrs.get('spec_shape')
                
                if emb_dim is not None and spec_sh is not None:
                    self.dt = self._set_dataset_format(emb_dim, spec_sh)
                    
                    # Load existing IDs into memory to avoid repeated disk reads during skips
                    dset = self.hf['embedding_dataset']
                    if dset.shape[0] > 0:
                        raw_ids = dset['ID'][:]
                        self.existing_keys = {k.decode('utf-8') if isinstance(k, bytes) else k for k in raw_ids}
                    
                    if self.mode == 'a':
                        self.buffer_array = np.empty(self.buffer_size, dtype=self.dt)
            except Exception as e:
                logging.warning(f"Manager: Error loading metadata from {h5_path}: {e}")

    def __contains__(self, emb_pkey: str) -> bool:
        """Checks if a specific embedding ID already exists in the file (O(1) lookup)."""
        return emb_pkey in self.existing_keys

    def _set_dataset_format(self, embedding_dim, spec_shape):
        """Defines the structured numpy dtype for the HDF5 records."""
        if 'classes' in self.partitions:
            dt = np.dtype([
                    ('ID', 'S100'),
                    ('embeddings', (np.float64, (embedding_dim,))),
                    ('spectrograms', (np.float64, spec_shape)),
                    ('track_names', 'S100'),
                    ('subclasses', 'S100')
                ])
        else:
            dt = np.dtype([
                    ('ID', 'S100'),
                    ('embeddings', (np.float64, (embedding_dim,))),
                    ('spectrograms', (np.float64, spec_shape)),
                    ('track_names', 'S100'),
                    ('classes', 'S100'),
                    ('subclasses', 'S100')
                ])
        return dt

    def initialize_hdf5(self, embedding_dim, spec_shape, audio_format, cut_secs, n_octave, 
                        sample_rate, seed, noise_perc, split, class_name=None):
        """
        Prepares a new HDF5 file for writing by setting global attributes and 
        pre-allocating the 'embedding_dataset' with resizable dimensions.
        """
        if self.mode == 'a':
            # Save structural metadata as HDF5 attributes
            self.hf.attrs['audio_format'] = audio_format
            self.hf.attrs['cut_secs'] = cut_secs
            self.hf.attrs['n_octave'] = n_octave
            self.hf.attrs['sample_rate'] = sample_rate
            self.hf.attrs['noise_perc'] = noise_perc
            self.hf.attrs['seed'] = seed
            self.hf.attrs['split'] = split
            self.hf.attrs['embedding_dim'] = embedding_dim
            self.hf.attrs['spec_shape'] = spec_shape 
            
            self.dt = self._set_dataset_format(embedding_dim, spec_shape)
            self.buffer_array = np.empty(self.buffer_size, dtype=self.dt)
            
            if 'classes' in self.partitions:
                self.hf.attrs['class'] = class_name
            
            if 'embedding_dataset' not in self.hf:
                # Use chunks=True to enable dynamic resizing via dataset.resize()
                self.hf.create_dataset('embedding_dataset', 
                                        shape=(0,),
                                        maxshape=(None,),
                                        dtype=self.dt,
                                        chunks=True)
            self.hf.flush()

    def add_to_data_buffer(self, embedding, spectrogram, hash_keys, track_name, class_=None, subclass=None):
        """
        Stages data into the RAM buffer. Flushes to disk automatically when the buffer is full.
        """
        idx = self.buffer_count
        
        def to_bytes(s):
            return s.encode('utf-8') if isinstance(s, str) else s

        self.buffer_array[idx]['ID'] = to_bytes(hash_keys)
        self.buffer_array[idx]['embeddings'] = embedding.detach().cpu().numpy() if torch.is_tensor(embedding) else embedding
        self.buffer_array[idx]['spectrograms'] = spectrogram
        self.buffer_array[idx]['track_names'] = to_bytes(track_name)
        
        if 'classes' not in self.partitions:
            self.buffer_array[idx]['classes'] = to_bytes(class_) if class_ else b''
        self.buffer_array[idx]['subclasses'] = to_bytes(subclass) if subclass else b''
        
        self.buffer_count += 1
        # Update the memory-safe key set immediately
        self.existing_keys.add(hash_keys if isinstance(hash_keys, str) else hash_keys.decode('utf-8'))
        
        if self.buffer_count >= self.buffer_size:
            self.flush_buffers()

    def flush_buffers(self):
        """
        Physically writes the buffered data to the HDF5 file and resets the RAM buffer.
        """
        if self.buffer_count == 0:
            return

        # Check for environmental override to disable saving (useful for dry runs)
        if os.getenv('NO_EMBEDDING_SAVE', 'False').lower() in ('false', '0', 'f'):
            dataset = self.hf['embedding_dataset']
            current_size = dataset.shape[0]
            dataset.resize(current_size + self.buffer_count, axis=0)
            dataset[current_size:] = self.buffer_array[:self.buffer_count]

        # Reset buffer by creating a new empty array to release C-level memory
        self.buffer_array = np.empty(self.buffer_size, dtype=self.dt)
        self.buffer_count = 0
        gc.collect()

    def extend_dataset(self, new_data):
        """
        Directly extends the HDF5 dataset using a pre-formatted numpy array, 
        bypassing the buffer for bulk operations.
        """
        dataset = self.hf['embedding_dataset']
        current_size = dataset.shape[0]
        dataset.resize(current_size + len(new_data), axis=0)
        dataset[current_size:] = new_data
        
        # 🎯 SYNC LOOKUP SET
        new_ids = new_data['ID']
        self.existing_keys.update(k.decode('utf-8') if isinstance(k, bytes) else k for k in new_ids)

    def close(self):
        """Ensures final buffer flush before closing the file handle."""
        if self.hf:
            self.flush_buffers() 
            self.hf.close()
            self.hf = None
        self.buffer_array = None

def combine_hdf5_files(root_dir, cut_secs_list, embedding_dim, spec_shape, audio_format, cut_secs, n_octave, 
                       sample_rate, seed, noise_perc, splits_list):
    """
    Consolidates individual class-specific HDF5 files into unified files for each dataset split.
    This version dynamically detects the spectrogram shape for each segment duration to prevent
    broadcasting errors when merging classes with different temporal dimensions.

    args:
     - root_dir (str): Base directory containing processed segment folders;
     - cut_secs_list (list): List of segment durations processed (e.g., [1, 3, 5]);
     - embedding_dim (int): Dimension of the audio embeddings (e.g., 1024);
     - spec_shape (tuple): Global shape (unused here as it is detected dynamically);
     - audio_format (str): Original audio format (e.g., 'wav');
     - cut_secs (int/float): Current segment duration being merged (from orchestrator);
     - n_octave (int): Octave band resolution;
     - sample_rate (int): Audio sampling rate;
     - seed (int): Global seed used for processing;
     - noise_perc (float): Noise percentage used for augmentation;
     - splits_list (list): List of tuples containing split names (e.g., [('train', size), ...]).
    """

    # --- ITERATE THROUGH EACH DURATION ---
    # 🎯 FIX: Move the shape detection inside the duration loop to handle variable segment lengths.
    for current_cut_secs in cut_secs_list:
        
        # 1. DYNAMIC SHAPE DETECTION
        # We must find the specific spectrogram shape for the current duration (e.g., 1s vs 5s)
        detected_spec_shape = None
        current_duration_path = os.path.join(root_dir, f'{current_cut_secs}_secs')
        
        # Guard check: Ensure the duration directory exists
        if not os.path.exists(current_duration_path):
            continue

        # Walk through directories to find a valid class-level HDF5 file
        for root, dirs, files in os.walk(current_duration_path):
            for file in files:
                # Identify intermediate embedding files, excluding master or raw datasets
                if file.endswith(".h5") and "_dataset" not in file and "combined_" not in file:
                    try:
                        sample_path = os.path.join(root, file)
                        with h5py.File(sample_path, 'r') as hf:
                            # Attempt to read spec_shape from global attributes
                            detected_spec_shape = hf.attrs.get('spec_shape')
                            # Fallback: inspect the dtype if the attribute is missing
                            if detected_spec_shape is None and 'embedding_dataset' in hf:
                                detected_spec_shape = hf['embedding_dataset'].dtype['spectrograms'].shape
                            
                            if detected_spec_shape is not None:
                                break
                    except Exception:
                        continue
            if detected_spec_shape is not None: break

        # If no data is found for this duration, skip to the next one
        if detected_spec_shape is None:
            print(f"⚠️ Warning: Could not find any valid .h5 to merge in {current_duration_path}. Skipping.")
            continue
        
        # Assign the detected shape for initializing the master file for this specific duration
        current_spec_shape = detected_spec_shape

        # Identify all processed classes for the current duration
        classes_list = sorted([d for d in os.listdir(current_duration_path) 
                                if os.path.isdir(os.path.join(current_duration_path, d))])
        
        # 2. DATASET SPLIT MERGING
        for split_tuple in splits_list:
            split_name = split_tuple[0]
            output_h5_path = os.path.join(root_dir, f'{current_cut_secs}_secs', f'combined_{split_name}.h5')
            
            # Initialize the output master file manager with the duration-specific spec_shape
            out_h5 = HDF5EmbeddingDatasetsManager(output_h5_path, mode='a', partitions=set(('splits',)))
            out_h5.initialize_hdf5(embedding_dim, current_spec_shape, audio_format, current_cut_secs, n_octave,
                                   sample_rate, seed, noise_perc, split_name)

            # Accumulate data from every class directory for the current split
            for class_name in classes_list:
                # Target the file generated by the distributed pipeline
                class_h5_path = os.path.join(root_dir, f'{current_cut_secs}_secs', class_name, 
                                             f'{class_name}_{split_name}_{audio_format}_emb.h5')
                        
                if not os.path.exists(class_h5_path):
                    continue
                        
                print(f"Merging: {class_name} | Split: {split_name} | Duration: {current_cut_secs}s")
                        
                try:
                    # Open class-specific file in read mode
                    in_h5 = HDF5EmbeddingDatasetsManager(class_h5_path, 'r', set(('splits', 'classes')))
                    class_data = in_h5.hf['embedding_dataset'][:]
                    
                    # Create an array matching the master file's structured dtype for this duration
                    class_data_extended = np.empty(class_data.shape, dtype=out_h5.dt)
                    
                    # Map common fields between structures
                    for name in class_data_extended.dtype.names:
                        if name in class_data.dtype.names: 
                            class_data_extended[name] = class_data[name]

                    # Inject the class label into the master dataset
                    if 'classes' in class_data_extended.dtype.names:
                        class_data_extended['classes'] = np.array([class_name.encode('utf-8')] * len(class_data),
                                                                    dtype=class_data_extended.dtype['classes'])
                        
                    # Bulk extend the master HDF5 dataset
                    out_h5.extend_dataset(class_data_extended)
                    in_h5.close()
                    
                except Exception as e:
                    print(f"ERROR: Failed to merge {class_name} ({split_name}): {e}")
                    
            # Ensure final flush and closure of the master file
            out_h5.close()


def get_track_reproducibility_parameters(idx):
    """
    Parses a primary key string into its constituent parameters. This key allows for 
    the exact reconstruction of the track slice and augmentation state used for a specific embedding.

    args:
     - idx (str): The embedding key in format 'classIdx_hdf5Index_bucket_round_results'.

    returns:
     - rep_params (dict): Dictionary mapping parameter names to their parsed values.
    """
    rep_params = {}
    params = idx.split('_')
    param_names = [
            "class_idx",    # Global index of the class
            "hdf5_index",   # Original index in the source audio HDF5
            "bucket",       # Segment index within the track
            "round_",       # Processing round (for offset calculation)
            "results"       # Global counter (for deterministic noise)
        ]
    for param, name in zip(params, param_names):
        rep_params[name] = param
    return rep_params


def reconstruct_tracks_from_embeddings(base_tracks_dir, hdf5_emb_path, idx_list):
    """
    Reconstructs the exact augmented audio waveforms used to generate embeddings.
    FIX: Added UTF-8 decoding for class names to prevent 'b' prefix in file paths.
    """
    # Open embedding file to retrieve global processing attributes
    hdf5_emb = HDF5EmbeddingDatasetsManager(hdf5_emb_path, 'r', ('splits',))
    audio_format = hdf5_emb.hf.attrs['audio_format']
    cut_secs = hdf5_emb.hf.attrs['cut_secs']
    sample_rate = hdf5_emb.hf.attrs['sample_rate']
    noise_perc = hdf5_emb.hf.attrs['noise_perc']
    seed = hdf5_emb.hf.attrs['seed']
    
    # --- FIX: Decode bytes from HDF5 classes dataset ---
    raw_classes = np.unique(hdf5_emb.hf['embedding_dataset']['classes'])
    classes_list = sorted([c.decode('utf-8') if isinstance(c, bytes) else str(c) for c in raw_classes])
    
    repr_params_list = [get_track_reproducibility_parameters(idx) for idx in idx_list]
    repr_params_list = sorted(repr_params_list, key=lambda a : int(a['class_idx']))
    
    curr_class = None
    reconstr_tracks = {}

    for track_idx_key, repr_params in zip(idx_list, repr_params_list):
        if repr_params['class_idx'] != curr_class:
            if curr_class is not None:
                hdf5_class_tracks.close()
            curr_class = repr_params['class_idx']
            
            # Recompute class seed and path using decoded class name
            class_name = classes_list[int(curr_class)]
            class_seed = int(seed + hash(class_name) % 10000000) 
            hdf5_class_path = os.path.join(base_tracks_dir, f'raw_{audio_format}', f'{class_name}_{audio_format}_dataset.h5')
            hdf5_class_tracks = HDF5DatasetManager(hdf5_class_path, audio_format)
        
        original_track = hdf5_class_tracks.hf[f'audio_{audio_format}'][int(repr_params['hdf5_index'])]

        # 1. RECONSTRUCT OFFSET
        offset_rng = np.random.default_rng(class_seed)
        offset = 0
        window_size = int(cut_secs * sample_rate)
        if int(repr_params['round_']) > 1 and original_track.shape[0] > window_size:
            max_offset = original_track.shape[0] - window_size
            if max_offset > 0:
                for _ in range(int(repr_params['round_'])):
                    offset = offset_rng.integers(0, max_offset)

        # 2. EXTRACT SEGMENT
        start = int(repr_params['bucket']) * window_size + offset
        end = start + window_size
        cut_track = original_track[start:end]

        if len(cut_track) < window_size:
            pad_length = window_size - len(cut_track)
            cut_track = np.pad(cut_track, (0, pad_length), 'constant')

        # 🎯 3. RECONSTRUCT DETERMINISTIC NOISE
        max_threshold = np.mean(np.abs(cut_track))
        audio_specific_seed = class_seed + int(repr_params['results'])
        noise_rng_batched = np.random.default_rng(audio_specific_seed)
        
        noise = noise_rng_batched.uniform(-max_threshold, max_threshold, cut_track.shape)
        reconstr_track = (1 - noise_perc) * cut_track + noise_perc * noise
        
        reconstr_tracks[track_idx_key] = reconstr_track

    hdf5_emb.close()
    if 'hdf5_class_tracks' in locals():
        hdf5_class_tracks.close()
    return reconstr_tracks

class TorchEmbeddingDataset(Dataset):
    """
    A standard PyTorch Dataset wrapper for CLAP embeddings. 
    It handles the conversion of numpy arrays retrieved from HDF5 into 
    Float32 and Long tensors, managing their placement on the target device.

    args:
     - embeddings (np.ndarray): The matrix of audio embeddings;
     - labels (np.ndarray): The array of numerical class indices;
     - classes (list): List of class names (strings) for reference;
     - device (str/torch.device, default: 'cpu'): The hardware device where 
                      tensors will be mapped.

    returns:
     - Instance: A dataset object compatible with torch.utils.data.DataLoader.
    """
    def __init__(self, embeddings, labels, classes, device='cpu'):
        self.embeddings = torch.tensor(embeddings, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.classes = classes
        self.device = device

    def __len__(self): 
        return len(self.embeddings)

    def __getitem__(self, idx):
        """
        Retrieves a single embedding-label pair, mapped to the specified device.
        """
        return self.embeddings[idx].to(self.device), self.labels[idx].to(self.device)


def load_single_cut_secs_dataloaders(octaveband_dir, cut_secs, batch_size, device='cpu'):
    """
    High-level utility to load pre-calculated embeddings from HDF5 files and 
    instantiate PyTorch DataLoaders for 'train', 'es' (early stopping), and 
    'valid' splits for a specific segment duration.

    The function performs automatic class-to-index encoding based on the 
    sorted unique class names found in the datasets.

    args:
     - octaveband_dir (str): Root directory for the specific octave resolution;
     - cut_secs (int/float): The duration of segments to load;
     - batch_size (int): Number of samples per batch for the DataLoaders;
     - device (str/torch.device, default: 'cpu'): Target device for tensors.

    returns:
     - dataloaders (dict): Dictionary mapping split names to their respective 
                      torch.utils.data.DataLoader instances;
     - classes (list): The sorted list of unique class names used for encoding.
    """
    splits = ['train', 'es', 'valid']
    dataloaders = {}
    classes = None

    for split in splits:
        # Construct path following the established directory structure
        h5_path = os.path.join(octaveband_dir, f"{cut_secs}_secs", f"combined_{split}.h5")
        if not os.path.exists(h5_path):
            raise FileNotFoundError(f"Dataset not found at: {h5_path}")

        with h5py.File(h5_path, 'r') as hf:
            data = hf['embedding_dataset']
            # Load embeddings directly into memory
            embeddings = data['embeddings'][:]
            
            # Decode class names from bytes to strings
            raw_classes = [c.decode('utf-8') if isinstance(c, bytes) else c for c in data['classes'][:]]
            
            # Ensure consistent class ordering across all splits
            if classes is None:
                classes = sorted(list(set(raw_classes)))
            
            # Create numerical mapping
            class_to_idx = {cls: i for i, cls in enumerate(classes)}
            labels = np.array([class_to_idx[c] for c in raw_classes])

            # Wrap in custom Dataset and create DataLoader
            dataset = TorchEmbeddingDataset(embeddings, labels, classes, device=device)
            # Shuffle is enabled only for the training set
            dataloaders[split] = DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))
            
    return dataloaders, classes


### Distributed environment functions ###

def setup_environ_vars(slurm=True):
    """
    Configures the necessary environment variables (MASTER_ADDR and MASTER_PORT) 
    required by PyTorch Distributed for process synchronization.

    For SLURM environments, it uses the Job ID to generate a unique port, 
    minimizing collisions between different concurrent jobs on the same cluster.

    args:
     - slurm (bool, default: True): If True, retrieves parameters from SLURM environment 
                      variables. If False, configures variables for local execution.

    returns:
     - rank (int): The unique identifier of the current process within the group;
     - world_size (int): Total number of processes in the distributed group.
    """
    if slurm:
        rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 4))
        
        # 🎯 MASTER_ADDR: For SLURM multi-node, this should ideally be the Rank 0 node.
        # Here we default to localhost for single-node multi-GPU stability.
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        
        # 🎯 MASTER_PORT: Deterministically generated from SLURM_JOB_ID.
        # This ensures all tasks in the same job use the same port, but different 
        # jobs use different ones.
        if "MASTER_PORT" not in os.environ:
            job_id = os.environ.get("SLURM_JOB_ID", "29500")
            base_port = 20000 + (int(job_id) % 10000)
            os.environ["MASTER_PORT"] = str(base_port)
            
        return rank, world_size
    else:
        # Local configuration using a random port for interactive sessions
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(np.random.randint(29500, 29999))
        return 0, 1 # Default values for non-slurm sequential start


def setup_distributed_environment(rank, world_size, slurm=True):
    """
    Initializes the PyTorch distributed process group using the 'gloo' backend. 
    It manages the rendezvous process via a shared synchronization file to 
    ensure robust group initialization across multiple processes.

    args:
     - rank (int): The rank of the current process;
     - world_size (int): The total number of processes in the group;
     - slurm (bool, default: True): Whether the environment is SLURM-managed.

    returns:
     - device (torch.device): The specific hardware device (CPU or CUDA:rank) 
                      assigned to this process.
    """
    import datetime
    
    # 🎯 FIX 1: Unique sync file per Job to avoid collisions
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    sync_file = f"/tmp_data/torch_sync_file_{job_id}"
    
    if rank == 0 and os.path.exists(sync_file):
        try:
            os.remove(sync_file)
        except Exception: pass

    init_method = f"file://{sync_file}"
    
    if VERBOSE:
        print(f"[RANK {rank}] Rendezvous on {sync_file} (WS={world_size})", flush=True)

    try:
        # 🎯 FIX 2: Extended timeout (300s) for heavy cluster starts
        dist.init_process_group(
            backend="gloo", 
            init_method=init_method,
            rank=rank, 
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )
        
        device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
        
        if rank == 0:
            print(f"🌐 [DIST] Group synchronized with {world_size} processes.", flush=True)
            
        return device
    except Exception as e:
        print(f"❌ [RANK {rank}] DISTRIBUTED ERROR: {e}", flush=True)
        raise e


def cleanup_distributed_environment(rank):
    """
    Gracefully shuts down the distributed process group.

    args:
     - rank (int): The rank of the process being shut down.

    returns:
     - None
    """
    try:
        dist.destroy_process_group()
        if VERBOSE:
            logging.info(f"Process {rank}: Group destroyed successfully.")
    except Exception as e:
        # Cleanup errors are typically non-fatal but logged in VERBOSE mode
        if VERBOSE:
            logging.warning(f"Rank {rank}: Error during dist cleanup: {e}")
