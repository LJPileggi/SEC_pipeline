import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np

# Dynamic root injection to safely import core production modules from src/
current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if sys.path.insert(0, src_root) if src_root not in sys.path else None: pass

# Import directly from the verified project core
from utils import get_config_from_yaml, HDF5DatasetManager

# Load production configurations dynamically to extract core parameters
classes, _, _, _, sampling_rate, _, noise_perc, _, _, _, _ = get_config_from_yaml("config0.yaml")

# Global immutable physical constants assigned directly from configuration
SAMPLE_RATE = sampling_rate       # 51200 Hz
SEGMENT_DURATION = 7.0            # Fixed target duration in seconds for U-Net extraction

class DistributedAudioRAWDataset(Dataset):
    """
    A lightweight index-mapping shell that coordinates multiple HDF5DatasetManager instances.
    Interleaves source track indices to mix acoustic classes at batch level.
    Extracts a completely random 7-second window per track and applies noise augmentation.
    """
    def __init__(self, base_dir, target_samples_per_class=500):
        self.base_dir = base_dir
        self.classes = sorted(classes)
        self.target_samples_per_class = target_samples_per_class
        self.window_size = int(SEGMENT_DURATION * SAMPLE_RATE)
        self.noise_perc = noise_perc
        
        self.managers = {}
        self.registry = []
        
        class_pools = {c: [] for c in self.classes}
        
        # Initialize native managers and extract raw track mappings
        for class_idx, class_name in enumerate(self.classes):
            h5_name = f"{class_name}_wav_dataset.h5"
            h5_path = os.path.join(base_dir, h5_name)
            
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"Missing mandatory production dataset: {h5_path}")
                
            # Initialize the production manager
            mgr = HDF5DatasetManager(h5_path, audio_format='wav')
            self.managers[class_name] = mgr
            
            # Simple linear index array based on the total records available in the HDF5
            num_records = mgr.n_records
            
            # Populate class entries using circular replication if records are less than the target quota
            for i in range(self.target_samples_per_class):
                real_hdf5_idx = i % num_records
                class_pools[class_name].append({
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'hdf5_index': real_hdf5_idx
                })

        # Interleave entries across classes to guarantee heterogeneous mini-batches
        for i in range(self.target_samples_per_class):
            for class_name in self.classes:
                self.registry.append(class_pools[class_name][i])

    def __len__(self):
        return len(self.registry)

    def __getitem__(self, idx):
        record = self.registry[idx]
        class_name = record['class_name']
        hdf5_index = record['hdf5_index']
        
        # Lazy extraction of the full 1D float32 waveform from disk
        mgr = self.managers[class_name]
        track, _ = mgr.get_audio_and_metadata(hdf5_index)
        
        # --- STOCHASTIC TIME-SHIFTING ---
        # Pick a completely random start offset anywhere within the valid track length
        offset = 0
        if track.shape[0] > self.window_size:
            max_offset = track.shape[0] - self.window_size
            offset = np.random.randint(0, max_offset)
            
        # Isolate the segment
        cut_data = track[offset:offset + self.window_size].astype(np.float32)
        
        # Padding safety guard for files physically shorter than 7 seconds
        if len(cut_data) < self.window_size:
            pad_length = self.window_size - len(cut_data)
            cut_data = np.pad(cut_data, (0, pad_length), 'constant')
            
        # --- STOCHASTIC NOISE AUGMENTATION ---
        # Evaluate dynamic amplitude boundary based on segment mean absolute energy
        max_threshold = np.mean(np.abs(cut_data))
        noise = np.random.uniform(-max_threshold, max_threshold, cut_data.shape)
        
        # Linear blending using the exact configuration percentage
        augmented_signal = (1 - self.noise_perc) * cut_data + self.noise_perc * noise
        augmented_signal = np.nan_to_num(augmented_signal, nan=0.0, posinf=0.0, neginf=0.0)
            
        return torch.from_numpy(augmented_signal).float(), record['class_idx']

    def close(self):
        """Safely release core file handles via their native destruction logic."""
        for mgr in self.managers.values():
            try:
                mgr.close() # Triggers hf.close() and gc.collect() inside utils.py
            except Exception:
                pass
        self.managers.clear()
