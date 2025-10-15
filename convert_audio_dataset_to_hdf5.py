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
    # Esempio: .../PREPROCESSED_DATASET/CLASSE/SOTTOCLASSE/file.wav
    
    # Trova l'indice di 'PREPROCESSED_DATASET' per avere un punto di riferimento
    try:
        data_root_index = parts.index("PREPROCESSED_DATASET")
    except ValueError:
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

def process_audio_dir_to_hdf5(base_dir: Path, output_h5_file: Path, audio_format: str):
    """
    Scansiona i file audio, li legge, li converte in array NumPy a lunghezza variabile
    e li salva in un singolo file HDF5.
    """
    print(f"Inizio serializzazione per formato: {audio_format.upper()}...")
    file_list = list(base_dir.rglob(f'*.{audio_format}'))
    
    # 1. Preparazione dei metadati
    metadata_list = []
    
    for file_path in tqdm(file_list, desc="Scansione Metadati"):
        class_name, subclass_name, track_name = extract_metadata_from_path(file_path)
        metadata_list.append({
            'class': class_name,
            'subclass': subclass_name,
            'track_name': track_name,
            'file_path': str(file_path)
        })

    if not metadata_list:
        print(f"Attenzione: Nessun file .{audio_format} trovato in {base_dir}")
        return

    df = pd.DataFrame(metadata_list)
    
    # 2. Creazione del file HDF5 e del dataset
    # Usiamo un dataset con tipo "object" (PyTables) o "VLEN" (H5Py)
    # H5PY ha bisogno di un dataset a lunghezza variabile (VLEN) per array NumPy.
    
    with h5py.File(output_h5_file, 'w') as hf:
        
        # 2.1. Creazione del Dataset Audio a Lunghezza Variabile (VLEN)
        # TIPO: 'float32' è lo standard per i dati audio normalizzati
        audio_dtype = h5py.vlen_dtype(np.dtype('float32'))
        audio_data = hf.create_dataset(f'audio_{audio_format}', (len(df),), dtype=audio_dtype)
        
        # 2.2. Salvataggio degli Attributi di Metadati
        # Usiamo HDF5 Attributes per le specifiche di serializzazione
        hf.attrs['audio_format'] = audio_format
        hf.attrs['sample_rate'] = 52100 # Sample rate target, da includere nei metadati
        hf.attrs['description'] = f'Raw audio data as variable-length float32 arrays.'
        
        # 2.3. Salvataggio delle Colonne come Dataset separati (come un DataFrame)
        hf.create_dataset('class', data=df['class'].to_numpy(dtype=h5py.string_dtype()))
        hf.create_dataset('subclass', data=df['subclass'].to_numpy(dtype=h5py.string_dtype()))
        hf.create_dataset('track_name', data=df['track_name'].to_numpy(dtype=h5py.string_dtype()))
        
        # 3. Lettura e Serializzazione dei Dati Audio
        print(f"Serializzazione di {len(df)} tracce in HDF5...")
        for i, row in tqdm(df.iterrows(), total=len(df), desc="Serializzazione Audio"):
            file_path = Path(row['file_path'])
            
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
                audio_data[i] = data.astype('float32')
                
            except Exception as e:
                print(f"Errore nella lettura del file {file_path}: {e}")
                # Potresti voler segnare questa riga come errore nel DataFrame se l'errore è critico
                
    print(f"File HDF5 creato con successo: {output_h5_file}")


if __name__ == '__main__':
    # Esempio di utilizzo:
    
    # Assumiamo che la tua struttura sia: BASE_DIR/PREPROCESSED_DATASET/
    user = os.environ.get('USER')
    BASE_DIR = Path(f'/leonardo_scratch/large/userexternal/{user}/dataSEC/RAW_DATASET') 
    
    # Formati da processare
    for fmt in ['wav', 'mp3', 'flac']:
        # Percorso dove salvare il file HDF5 (es. nella cartella principale del dataset)
        basedir_format = Path(os.path.join(BASE_DIR, f'raw_{fmt}'))
        OUTPUT_H5_PATH = Path(os.path.join(BASE_DIR, f'SEC_SuperDataset_{fmt}.h5'))
        process_audio_dir_to_hdf5(basedir_format, OUTPUT_H5_PATH, fmt)
        # Nota: L'attuale implementazione sovrascrive il file H5 ad ogni iterazione.
        # Per includere tutti i formati, dovresti modificare 'process_audio_dir_to_hdf5' 
        # per appendere i dati o combinare i formati in un unico dataset.
        # Per la prima implementazione, concentriamoci su un formato alla volta.
        # Adattalo per il tuo uso finale (es. usando f'audio_{fmt}' come nome del dataset HDF5)
