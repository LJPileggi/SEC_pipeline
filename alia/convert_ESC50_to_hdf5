import h5py
import numpy as np
import pandas as pd
import os
import librosa
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)

def process_esc50_to_hdf5(base_dir: Path, target_dir: Path):
    """
    Converte ESC-50 in file HDF5 per classe, compatibili con IDOFASC_HPC.
    Usa 'category' dal CSV come nome classe e mantiene la coerenza con DataSEC.
    """
    # ESC-50 ha un sample rate nativo di 44.1kHz
    target_sr = 44100 
    audio_format = 'wav'
    
    metadata_path = base_dir / "meta" / "esc50.csv"
    audio_dir = base_dir / "audio"
    
    if not metadata_path.exists():
        logging.error(f"File metadati non trovato: {metadata_path}")
        return

    # Caricamento e raggruppamento per categoria
    df_meta = pd.read_csv(metadata_path)
    categories = df_meta['category'].unique()

    METADATA_DTYPE = np.dtype([
        ('subclass', h5py.string_dtype(encoding='utf-8')), 
        ('track_name', h5py.string_dtype(encoding='utf-8')),
        ('hdf5_index', np.int32)
    ])

    for category in sorted(categories):
        logging.info(f"Classe in elaborazione: {category}")
        class_df = df_meta[df_meta['category'] == category].reset_index(drop=True)
        
        h5_path = target_dir / f'{category}_{audio_format}_dataset.h5'
        
        with h5py.File(h5_path, 'w') as hf:
            # Creazione dataset audio a lunghezza variabile
            audio_dtype = h5py.vlen_dtype(np.dtype('float32'))
            audio_data_dset = hf.create_dataset(f'audio_{audio_format}', (len(class_df),), dtype=audio_dtype, chunks=True)

            # Attributi globali per il manager IDOFASC
            hf.attrs['audio_format'] = audio_format
            hf.attrs['class'] = category
            hf.attrs['sample_rate'] = target_sr

            metadata_array = np.empty(len(class_df), dtype=METADATA_DTYPE)
            
            for i, row in tqdm(class_df.iterrows(), total=len(class_df), desc=f"Writing {category}"):
                # Usiamo il 'fold' come sottoclasse per mantenere la struttura DataSEC
                subclass = f"fold_{row['fold']}"
                track_filename = row['filename']
                metadata_array[i] = (subclass, track_filename, i)
                
                file_path = audio_dir / track_filename
                try:
                    # Caricamento e normalizzazione
                    data, _ = librosa.load(file_path, sr=target_sr, mono=True)
                    audio_data_dset[i] = data.astype('float32')
                except Exception as e:
                    logging.error(f"Errore su {file_path}: {e}")

            # Scrittura finale metadati
            hf.create_dataset(f'metadata_{audio_format}', data=metadata_array)

        logging.info(f"HDF5 creato: {h5_path}")

if __name__ == '__main__':
    user = os.environ.get('USER')
    # Path basati sulla struttura standard della repo GitHub di ESC-50
    BASE_DIR = Path(f'/leonardo_scratch/large/userexternal/{user}/ESC-50')
    TARGET_DIR = Path(f'/leonardo_scratch/large/userexternal/{user}/ESC50_HDF5/raw_wav')
    
    os.makedirs(TARGET_DIR, exist_ok=True)
    process_esc50_to_hdf5(BASE_DIR, TARGET_DIR)
