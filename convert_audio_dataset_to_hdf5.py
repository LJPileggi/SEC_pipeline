import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import librosa
import logging

logging.basicConfig(level=logging.INFO)

def extract_metadata_from_path(file_path: Path):
    """
    Mantiene la logica originale di estrazione dei metadati.
   
    """
    parts = file_path.parts
    class_name = parts[-3] if len(parts) >= 3 else "Undefined"
    subclass_name = parts[-2] if len(parts) >= 2 else "None"
    track_name = file_path.name
    return class_name, subclass_name, track_name

def process_audio_dir_to_hdf5(base_dir: Path, target_dir: Path, audio_format: str):
    """
    Converte i file audio in HDF5 rispettando il mantra: 'non toccare quello che già funziona'.
    Corregge solo il posizionamento della creazione del dataset metadati.
   
    """
    target_sr = 51200 # Sample rate originale delle tracce

    METADATA_DTYPE = np.dtype([
        ('subclass', h5py.string_dtype(encoding='utf-8')), 
        ('track_name', h5py.string_dtype(encoding='utf-8')),
        ('hdf5_index', np.int32)
    ])

    for class_dir_name in sorted(os.listdir(base_dir)):
        class_path = base_dir / class_dir_name
        if not class_path.is_dir():
            continue
            
        logging.info(f"Inizio serializzazione classe: {class_dir_name}")
        file_list = list(class_path.rglob(f'*.{audio_format}'))
        
        if not file_list:
            continue

        metadata_list = []
        for file_path in tqdm(file_list, desc=f"Scanning {class_dir_name}"):
            _, subclass_name, track_name = extract_metadata_from_path(file_path)
            metadata_list.append({
                'subclass': subclass_name,
                'track_name': track_name,
                'file_path': file_path
            })

        df = pd.DataFrame(metadata_list)
        h5_path = target_dir / f'{class_dir_name}_{audio_format}_dataset.h5'

        with h5py.File(h5_path, 'w') as hf:
            audio_dtype = h5py.vlen_dtype(np.dtype('float32'))
            audio_data_dset = hf.create_dataset(f'audio_{audio_format}', (len(df),), dtype=audio_dtype, chunks=True)

            hf.attrs['audio_format'] = audio_format
            hf.attrs['class'] = class_dir_name
            hf.attrs['sample_rate'] = target_sr

            metadata_array = np.empty(len(df), dtype=METADATA_DTYPE)
            
            # --- MODIFICA MINIMA: Il loop ora popola solo l'array in memoria ---
            for i, row in tqdm(df.iterrows(), total=len(df), desc="Scrittura Audio"):
                metadata_array[i] = (row['subclass'], row['track_name'], i)
                try:
                    data, _ = librosa.load(row['file_path'], sr=target_sr, mono=True)
                    audio_data_dset[i] = data.astype('float32')
                except Exception as e:
                    logging.error(f"Errore su {row['file_path']}: {e}")

            # --- MODIFICA MINIMA: Scrittura dei metadati spostata qui per evitare errori di duplicazione ---
            hf.create_dataset(f'metadata_{audio_format}', data=metadata_array)

        logging.info(f"Completato: {h5_path}")

if __name__ == '__main__':
    user = os.environ.get('USER')
    BASE_DIR = Path(f'/leonardo/home/userexternal/{user}/SEC/dataSEC/RAW_DATASET')
    TARGET_DIR = Path(f'/leonardo_scratch/large/userexternal/{user}/dataSEC/RAW_DATASET')
    if not os.path.exists(TARGET_DIR):
        os.makedir(TARGET_DIR)
    
    for fmt in ['wav', 'mp3', 'flac']:
        basedir_format = BASE_DIR / f'raw_{fmt}'
        targetdir_format = TARGET_DIR / f'raw_{fmt}'
        if not os.path.exists(targetdir_format):
            os.mkdir(targetdir_format)
        if basedir_format.exists():
            process_audio_dir_to_hdf5(basedir_format, targetdir_format, fmt)
