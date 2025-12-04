import os
import h5py
import numpy as np
import tempfile
import shutil
from typing import List, Dict, Tuple

# Definizioni per il dataset giocattolo (potrebbero andare all'inizio del tuo file di test)
TEST_CLASSES: List[str] = ['Music', 'Voices', 'Birds']
TEST_TRACKS_PER_CLASS: Dict[str, int] = {'Music': 3, 'Voices': 3, 'Birds': 2} # 8 tracce totali
TEST_TRACK_DURATION_SECONDS: int = 3 # Durata di ogni traccia finta
SAMPLING_RATE: int = 44100 # Hz
AUDIO_DTYPE: type = h5py.vlen_dtype(np.dtype('float32')) # Tipo di dati per le forme d'onda
TEST_AUDIO_FORMAT: str = "wav" # Usato per la struttura del path e il nome del dataset interno
TEST_N_OCTAVE: int = 1 # Usato per la struttura del path degli embeddings, non raw audio

# Per i cut_secs, verranno usati dopo dalla pipeline di embedding, non nella generazione raw
TEST_CUT_SECS_VALUES: List[int] = [1, 2] # Manteniamo per riferimento futuro

# Definizione del METADATA_DTYPE come fornito
METADATA_DTYPE = np.dtype([
    ('subclass', h5py.string_dtype(encoding='utf-8')), # Sottoclasse
    ('track_name', h5py.string_dtype(encoding='utf-8')), # Nome traccia
    ('hdf5_index', np.int32) # Indice di riga originale (0, 1, 2...)
])

def create_fake_raw_audio_h5(base_raw_dir: str) -> List[str]:
    """
    Crea una struttura di directory e file HDF5 finti contenenti tracce audio RAW.
    Questi file emulano l'output della conversione iniziale di file audio reali.
    Ogni file HDF5 contiene un dataset audio e un dataset 'metadata_{audio_format}'
    per i metadati specifici delle tracce, oltre agli attributi globali richiesti.
    
    Args:
        base_raw_dir (str): La directory radice dove verranno creati i file raw HDF5.
                            Es: temp_dir/raw_data
                                 /raw_data/wav/Music.h5
                                 /raw_data/wav/Voices.h5
                                 /raw_data/wav/Birds.h5
    
    Returns:
        List[str]: Una lista dei percorsi dei file HDF5 creati.
    """
    print(f"Generating fake RAW audio HDF5 datasets in: {base_raw_dir}")

    output_path_base = os.path.join(base_raw_dir, TEST_AUDIO_FORMAT)
    os.makedirs(output_path_base, exist_ok=True)

    generated_files: List[str] = []

    class_to_idx = {cls: idx for idx, cls in enumerate(TEST_CLASSES)}

    for class_name in TEST_CLASSES:
        h5_filename = f'{class_name}_{TEST_AUDIO_FORMAT}_dataset.h5'
        h5_classdir = os.path.join(output_path_base, class_name)
        os.makedirs(h5_classdir, exist_ok=True)
        h5_filepath = os.path.join(h5_classdir, h5_filename)
        
        num_tracks = TEST_TRACKS_PER_CLASS.get(class_name, 0)
        if num_tracks == 0:
            print(f"Warning: No tracks defined for class {class_name}, skipping HDF5 creation.")
            continue

        print(f"Creating {num_tracks} fake tracks for class '{class_name}' in {h5_filepath}...")
        
        samples_per_track = SAMPLING_RATE * TEST_TRACK_DURATION_SECONDS
        
        all_class_audio_data = np.random.rand(num_tracks * samples_per_track).astype(AUDIO_DTYPE) * 0.2 - 0.1

        metadata_records: List[Tuple[bytes, bytes, int]] = [] # Per il nuovo dataset di metadati
        
        current_start_sample = 0
        for i in range(num_tracks):
            end_sample = current_start_sample + samples_per_track
            
            # Creazione del record di metadati per ogni traccia
            # 'subclass' userà il class_name come sottoclasse per semplicità
            # 'track_name' sarà una stringa unica, es: Music_track_0, Music_track_1
            metadata_records.append((
                class_name.encode('utf-8'), 
                f'{class_name}_track_{i}'.encode('utf-8'), 
                i # hdf5_index
            ))
            current_start_sample = end_sample
        
        # Scrivi i dati audio, l'indice e i metadati nel file HDF5
        with h5py.File(h5_filepath, 'w') as f:
            f.create_dataset(f'audio_{TEST_AUDIO_FORMAT}', data=all_class_audio_data, dtype=AUDIO_DTYPE)
            
            # --- AGGIUNTA CHIRURGICA: Dataset dei metadati specifici delle tracce ---
            f.create_dataset(f'metadata_{TEST_AUDIO_FORMAT}', data=np.array(metadata_records, dtype=METADATA_DTYPE))
            # --- FINE AGGIUNTA CHIRURGICA ---
            
            # Aggiungi gli attributi globali al file HDF5
            f.attrs['audio_format'] = TEST_AUDIO_FORMAT
            f.attrs['class'] = class_name
            f.attrs['class_idx'] = class_to_idx[class_name]
            f.attrs['sample_rate'] = SAMPLING_RATE
            f.attrs['description'] = f"Fake audio data for {class_name} class for testing."

        print(f"Created fake RAW HDF5: {h5_filepath} with 'audio_{TEST_AUDIO_FORMAT}' shape {all_class_audio_data.shape} and {len(metadata_records)} metadata entries.")
        generated_files.append(h5_filepath)
            
    return generated_files

# Esempio di utilizzo (solo per test, questo non andrà nel tuo main di test definitivo)
if __name__ == "__main__":
    temp_root = tempfile.mkdtemp()
    fake_raw_dir = os.path.join(temp_root, 'raw_data')
    
    print(f"Temporary root directory: {temp_root}")
    
    created_h5_files = create_fake_raw_audio_h5(fake_raw_dir)
    
    print("\nVerifying created files:")
    for f_path in created_h5_files:
        if os.path.exists(f_path):
            with h5py.File(f_path, 'r') as f:
                print(f"  {f_path}")
                print(f"    Dataset 'audio_{TEST_AUDIO_FORMAT}' shape: {f[f'audio_{TEST_AUDIO_FORMAT}'].shape}, dtype: {f[f'audio_{TEST_AUDIO_FORMAT}'].dtype}")
                # --- VERIFICA CHIRURGICA: Nuova stampa per il dataset di metadati ---
                print(f"    Dataset 'metadata_{TEST_AUDIO_FORMAT}' shape: {f[f'metadata_{TEST_AUDIO_FORMAT}'].shape}, dtype: {f[f'metadata_{TEST_AUDIO_FORMAT}'].dtype}")
                # --- FINE VERIFICA CHIRURGICA ---
                print(f"    Attrs: audio_format={f.attrs['audio_format']}, class={f.attrs['class']}, class_idx={f.attrs['class_idx']}, sample_rate={f.attrs['sample_rate']}, description='{f.attrs['description']}'")
        else:
            print(f"  ERROR: File not found: {f_path}")

    # Pulizia
    print(f"\nCleaning up temporary directory: {temp_root}")
    shutil.rmtree(temp_root)
