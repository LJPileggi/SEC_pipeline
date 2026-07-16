import os
import sys
import torch
from torch.utils.data import Dataset
import numpy as np
import math

current_dir = os.path.dirname(os.path.abspath(__file__))
src_root = os.path.abspath(os.path.join(current_dir, "..", ".."))
if src_root not in sys.path: sys.path.insert(0, src_root)

from utils import get_config_from_yaml, HDF5DatasetManager

classes, _, _, _, sampling_rate, _, noise_perc, _, _, _, _ = get_config_from_yaml("config0.yaml")

SAMPLE_RATE = sampling_rate       
SEGMENT_DURATION = 7.0            

class DistributedAudioRAWDataset(Dataset):
    """
    Interleaved HDF5 dataset manager supporting isolated 'train' and 'test' track boundaries[cite: 14].
    """
    def __init__(self, base_dir, split="train", target_samples_per_class=500):
        self.base_dir = base_dir
        self.classes = sorted(classes)
        self.target_samples_per_class = target_samples_per_class
        self.window_size = int(SEGMENT_DURATION * SAMPLE_RATE)
        self.noise_perc = noise_perc
        self.split = split
        
        self.managers = {}
        self.registry = []
        class_pools = {c: [] for c in self.classes}
        
        for class_idx, class_name in enumerate(self.classes):
            h5_path = os.path.join(base_dir, f"{class_name}_wav_dataset.h5")
            if not os.path.exists(h5_path):
                raise FileNotFoundError(f"Missing mandatory dataset: {h5_path}")
                
            mgr = HDF5DatasetManager(h5_path, audio_format='wav')
            self.managers[class_name] = mgr
            
            # Split indices boundaries to ensure independent test isolation
            num_records = mgr.n_records
            split_boundary = int(num_records * 0.8)
            
            if self.split == "train":
                available_indices = np.arange(0, split_boundary)
            else: # test split
                available_indices = np.arange(split_boundary, num_records)
                
            if len(available_indices) == 0:
                available_indices = np.arange(0, num_records) # Fallback safe guard
                
            for i in range(self.target_samples_per_class):
                real_idx = available_indices[i % len(available_indices)]
                class_pools[class_name].append({
                    'class_name': class_name,
                    'class_idx': class_idx,
                    'hdf5_index': real_idx
                })

        for i in range(self.target_samples_per_class):
            for class_name in self.classes:
                if i < len(class_pools[class_name]):
                    self.registry.append(class_pools[class_name][i])

    def __len__(self):
        return len(self.registry)

    def __getitem__(self, idx):
        record = self.registry[idx]
        class_name = record['class_name']
        hdf5_index = record['hdf5_index']
        
        mgr = self.managers[class_name]
        track, _ = mgr.get_audio_and_metadata(hdf5_index)
        
        offset = 0
        if track.shape[0] > self.window_size:
            max_offset = track.shape[0] - self.window_size
            offset = np.random.randint(0, max_offset)
            
        cut_data = track[offset:offset + self.window_size].astype(np.float32)
        if len(cut_data) < self.window_size:
            cut_data = np.pad(cut_data, (0, self.window_size - len(cut_data)), 'constant')
            
        max_threshold = np.mean(np.abs(cut_data))
        noise = np.random.uniform(-max_threshold, max_threshold, cut_data.shape)
        
        augmented_signal = (1 - self.noise_perc) * cut_data + self.noise_perc * noise
        augmented_signal = np.nan_to_num(augmented_signal, nan=0.0, posinf=0.0, neginf=0.0)
            
        return torch.from_numpy(augmented_signal).float(), record['class_idx']

    def close(self):
        for mgr in self.managers.values():
            try: mgr.close()
            except Exception: pass
        self.managers.clear()
