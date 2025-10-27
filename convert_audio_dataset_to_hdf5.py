import h5py
import numpy as np
import pandas as pd
import os
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa # Per i formati non WAV

# Funzione per estrarre classe, sottoclasse e nome del file (da adattare al tuo schema)
def extract_metadata_from_path(file_path: Path):
    """
    Estrae classe, sottoclasse e nome del file dal percorso.
    Assumiamo uno schema standard: /path/to/CLASSE/SOTTOCLASSE/nomefile.ext
    """
    parts = file_path.parts

    data_root_index = -3 # Fallback o adatta a dove inizia la tua gerarchia

    # La classe è due livelli sopra il file
    if len(parts) > data_root_index + 2:
        class_name = parts[data_root_index + 1]
    else:
        class_name = "Undefined"
        
    # La sottoclasse è un livello sopra il file (se esiste)
    if len(parts) > data_root_index + 3:
        subclass_name = parts[data_root_index + 2]
    else:
        subclass_name = "None"
        
    track_name = file_path.name
    
    return class_name, subclass_name, track_name

def process_audio_dir_to_hdf5(base_dir: Path, audio_format: str):
    """
    Scansiona i file audio, li legge, li converte in array NumPy a lunghezza variabile
    e li salva in un singolo file HDF5.
    """

    METADATA_DTYPE = np.dtype([
        ('subclass', h5py.string_dtype(encoding='utf-8')), # Sottoclasse
        ('track_name', h5py.string_dtype(encoding='utf-8')), # Nome traccia
        ('original_index', np.int32) # Indice di riga originale (0, 1, 2...)
    ])

    for class_idx, class_dir in enumerate(sorted(os.listdir(base_dir))):
        if os.path.isdir(os.path.join(base_dir, class_dir)):
            print(f"Inizio serializzazione per classe: {class_dir}...")
            file_list = list(os.path.join(base_dir, class_dir).rglob(f'*.{audio_format}'))

            # 1. Preparazione dei metadati
            metadata_list = []

            for file_path in tqdm(file_list, desc=f"Scansione Metadati classe {class_dir}"):
                class_name, subclass_name, track_name = extract_metadata_from_path(file_path)
                metadata_list.append({
                    'subclass': subclass_name,
                    'track_name': track_name
                })

            if not metadata_list:
                print(f"Attenzione: Nessun file .{audio_format} trovato in {base_dir}")
                return

            df = pd.DataFrame(metadata_list)

            # 2. Creazione del file HDF5 e del dataset
    
            with h5py.File(os.path.join(base_dir, f'{class_dir}_{audio_format}_dataset.h5'), 'w') as hf:
        
                # 2.1. Creazione del Dataset Audio a Lunghezza Variabile (VLEN)
                # TIPO: 'float32' è lo standard per i dati audio normalizzati
                audio_dtype = h5py.vlen_dtype(np.dtype('float32'))
                audio_data_dset = hf.create_dataset(f'audio_{audio_format}', (len(df),), dtype=audio_dtype, 
                                            chunks=(1024,), maxshape=(None,)) # Usa chunks per performance
        
                # 2.2. Salvataggio degli Attributi di Metadati
                # Usiamo HDF5 Attributes per le specifiche di serializzazione
                hf.attrs['audio_format'] = audio_format
                hf.attrs['class'] = class_dir
                hf.attrs['class_idx'] = class_idx
                hf.attrs['sample_rate'] = 52100 # Sample rate target, da includere nei metadati
                hf.attrs['description'] = f'Raw audio data as variable-length float32 arrays.'
        
                # 2.3. Dataset Metadati (STRUTTURATO)
                # Crea un array NumPy per i metadati strutturati
                metadata_array = np.empty(len(df), dtype=METADATA_DTYPE)
        
                # 3. Lettura e Serializzazione dei Dati Audio
                print(f"Serializzazione di {len(df)} tracce in HDF5...")
                for i, row in tqdm(df.iterrows(), total=len(df), desc="Serializzazione Audio"):
                    file_path = Path(row['file_path'])

                    metadata_array[i] = (
                                row['subclass'], 
                                row['track_name'], 
                                i # L'indice HDF5 è semplicemente l'indice di riga
                            )

                    # Crea il dataset HDF5 strutturato.
                    # Questo mantiene l'allineamento perfetto tra indice audio e indice metadati.
                    metadata_dset = hf.create_dataset(f'metadata_{audio_format}', data=metadata_array)

                    try:
                        # Lettura e resample: Importante standardizzare il sample rate per CLAP (48kHz)
                        if audio_format == 'wav':
                            # soundfile è efficiente per WAV
                            data, sr = sf.read(file_path, dtype='float32')
                        else:
                            # librosa è più robusto per MP3/FLAC
                            data, sr = librosa.load(file_path, sr=None, mono=True)
                
                        # Conversione in mono e resample a 48kHz, se necessario
                        if sr != 52100:
                            data = librosa.resample(data, orig_sr=sr, target_sr=52100)
                            sr = 52100
                    
                        # Se il file è stereo, prendiamo un solo canale o lo convertiamo in mono
                        if data.ndim > 1:
                            data = np.mean(data, axis=1)
                    
                        # Salva l'array nel dataset VLEN
                        audio_data_dset[i] = data.astype('float32')
                
                    except Exception as e:
                        print(f"Errore nella lettura del file {file_path}: {e}")
                        # Potresti voler segnare questa riga come errore nel DataFrame se l'errore è critico
                
            print(f"File HDF5 creato con successo: {class_dir}_emb.h5")


if __name__ == '__main__':
    
    # Assumiamo che la tua struttura sia: BASE_DIR/PREPROCESSED_DATASET/
    user = os.environ.get('USER')
    BASE_DIR = Path(f'/leonardo_scratch/large/userexternal/{user}/dataSEC/RAW_DATASET') 
    
    # Formati da processare
    for fmt in ['wav', 'mp3', 'flac']:
        # Percorso dove salvare il file HDF5 (es. nella cartella principale del dataset)
        basedir_format = Path(os.path.join(BASE_DIR, f'raw_{fmt}'))
        process_audio_dir_to_hdf5(basedir_format, fmt)
