import os
import json
import yaml
import glob
import logging
import h5py
import numpy as np

### Get model, training and spectrogram configuration from yaml ###

def get_config_from_yaml(config_file="config0.yaml"):
    """
    Loads configuration from yaml file and yields
    variables to use in desired namespace.

    args:
     - config_file: name of config file, to be attached to its relative path.
    """
    config_path = os.path.join('configs', config_file)
    with open(config_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
    patience = configs["patience"]
    epochs = configs["epochs"]
    batch_size = configs["batch_size"]
    save_log_every = configs["save_log_every"]
    sampling_rate = configs["sampling_rate"]
    ref = configs["ref"]
    noise_perc = configs["noise_perc"]
    seed = configs["seed"]
    center_freqs = np.array(configs["center_freqs"])
    valid_cut_secs = configs["valid_cut_secs"]
    divisions_xc_sizes_names = [("train", configs["train_size"]),
                                ("es", configs["es_size"]),
                                ("valid", configs["valid_size"]),
                                ("test", configs["test_size"])]
    return patience, epochs, batch_size, save_log_every, sampling_rate, ref, noise_perc, \
                            seed, center_freqs, valid_cut_secs, divisions_xc_sizes_names

   
### Files and directory handling functions ###

def extract_all_files_from_dir(source_class_dir, extension='.wav'):
    """
    Extracts recursively files of a certain extension from nested directories.
    
    args:
     - source_class_dir: root directory from which to start the recursive search;
     - extension: the required extension for files. All other files are going to
       be neglected. Has to be an audio format extension like wav, mp3, flac etc.
       
    returns:
     - list of paths to audio files of the said extension.
    """
    subdirs = [source_class_dir]
    audio_fp_list = []
    while len(subdirs) > 0:
        basedir = subdirs.pop(0)
        for fn in os.listdir(basedir):
            full_fn = os.path.join(basedir, fn)
            if full_fn.lower().endswith(extension):
                short_path = full_fn.replace(source_class_dir, '')
                if short_path[0] == '/':
                    short_path = short_path[1:]
                audio_fp_list.append(short_path)
            if os.path.isdir(full_fn):
                subdirs.append(full_fn)
    return sorted(audio_fp_list)

### Log file functions for embedding calculation ###

def gen_log(log_path, cut_secs, ic, di, results, round_, finish_class, \
                    divisions_xc_sizes_names, noise_perc, seed, rank):
    """
    Generates a log file to set up a check point for embedding generation.
    This function is called every #save_log_every embedding creations or
    as a consequence of a keyboard interruption. Args are among the ones
    of the split_audio_tracks function. The log is organised as a dictionary
    and then dumped into a json file.
    
    args:
     - cut_secs: the CLAP cut length for the embeddings;
     - ic: class index
     - di: index of the current dataset split. In order they are: 'train', 
       'es', valid', 'test'; 
     - results: embedding index among the total number to generate per class;
     - round_: round of data augmentation. When the number of tracks if less
       than the required number for each class, split_audio_tracks starts to
       run subsequent data augmentation runs until the desired number is reached;
     - finish_class: whether the current class has been completely swept of not;
     - divisions_xc_sizes_names: a list containing tuples of ('split_name', #samples);
       the usual one is [('train', 500), ('es', 100), ('valid', 100), ('test', 100)];
     - noise_perc: intensity of noise for data augmentation;
     - seed: random seed for track permutations inside the same class;
     - rank: rank of the log generating process to confront different logs and
       sort them lexicographically.
    """
    lengths = list(zip(*divisions_xc_sizes_names))[1]
    results = sum(lengths[:di]) if di > 0 else 0
    log = {
        "cut_secs" : cut_secs,
        "ic" : ic,
        "di" : di,
        "results" : results,
        "round" : round_,
        "finish_class" : finish_class,
        "divisions_xc_sizes_names" : divisions_xc_sizes_names,
        "noise_perc" : noise_perc,
        "seed" : seed
    }
    with open(os.path.join(log_path, f"log_rank_{rank}.json"), 'w') as f:
        json.dump(log, f, indent=4)
    print("Logfile saved successfully.\n"
          f"cut_secs: {cut_secs}\n"
          f"ic: {ic}\n"
          f"di: {di}\n"
          f"results: {results}\n"
          f"round: {round_}\n"
          f"finish_class: {finish_class}\n"
          f"divisions_xc_sizes_names: {divisions_xc_sizes_names}\n"
          f"noise_perc: {noise_perc}\n"
          f"seed: {seed}\n"
    )

def read_log(log_path):
    """
    Reads all log files, finds the one lexicographically earliest, and returns its content.
    returns:
     - best_log_data: a dictionary containing the lexicographically earliest logging information.
    """
    all_logs = glob.glob(os.path.join(log_path, "log_rank_*.json"))
    
    if not all_logs:
        raise FileNotFoundError

    # Inizializza con un valore che sarà sicuramente superato
    best_log_data = None
    best_log_score = (float('inf'), float('inf'), float('inf'), float('inf'), float('inf')) # Imposta il valore iniziale più alto
    
    for log_file in all_logs:
        try:
            with open(log_file, 'r') as f:
                current_log_data = json.load(f)
            
            # Crea un "punteggio" lessicografico per il confronto
            current_log_score = (
                current_log_data.get("cut_secs", float('inf')),
                current_log_data.get("ic", float('inf')),
                current_log_data.get("di", float('inf')),
                current_log_data.get("results", float('inf')),
                current_log_data.get("round", float('inf'))
            )

            # Confronta e trova il log più avanzato
            if current_log_score < best_log_score:
                best_log_score = current_log_score
                best_log_data = current_log_data
        except Exception as e:
            logging.error(f"Errore nella lettura del file di log {log_file}: {e}")
            continue

    if best_log_data:
        logging.info("Trovato il log più recente. Proseguo dal punto di interruzione.")
    else:
        logging.info("Nessun log valido trovato. Avvio da zero.")

    # Opzionale: pulisci i file temporanei dopo la lettura
    for log_file in all_logs:
        try:
            os.remove(log_file)
        except OSError as e:
            logging.error(f"Errore nella rimozione del file {log_file}: {e}")

    return best_log_data

def delete_log(log_path):
    """
    Deletes the log file (if exists) after a complete embedding run.
    """
    try:
        os.remove(os.path.join(log_path, "log.json"))
    except FileNotFoundError:
        return

### hdf5 data generation and appending for embeddings and spectrograms ###

def initialize_hdf5(file_path, embedding_dim, spec_shape):
    """
    Creates HDF5 file with resizable embedding and spectrogram datasets.

    args:
     - file_path: path of HDF5 file;
     - embedding_dim: dimension of single embedding;
     - spec_shape: shape of single spectrogram.
    """
    # 'a' per aprire in modalità append, crea il file se non esiste
    with h5py.File(file_path, 'a') as hf:
        if 'embeddings' not in hf:
            hf.create_dataset(
                'embeddings',
                shape=(0, embedding_dim),  # La prima dimensione è 0, sarà ridimensionata
                maxshape=(None, embedding_dim), # Può crescere indefinitamente lungo la prima dimensione
                dtype='f4', # Float a 32 bit
                chunks=True
            )
        if 'spectrograms' not in hf:
            hf.create_dataset(
                'spectrograms',
                shape=(0,) + spec_shape,
                maxshape=(None,) + spec_shape,
                dtype='f4',
                chunks=True
            )
        if 'names' not in hf:
            dt = h5py.string_dtype(encoding='utf-8')
            hf.create_dataset('names', shape=(0,), maxshape=(None,), dtype=dt, chunks=True)

def append_to_hdf5(file_path, embeddings_buffer, spectrograms_buffer):
    """
    Appends embedding and spectrogram batches to preexisting HDF5 file.

    args:
     - file_path: path of HDF5 file to append data;
     - embeddings_buffer: buffer of embeddings to append;
     - spectrograms_buffer: buffer of spectrograms to append.
    """
    with h5py.File(file_path, 'a') as hf:
        # Aggiungi gli embeddings
        embeddings_ds = hf['embeddings']
        current_size = embeddings_ds.shape[0]
        new_size = current_size + len(embeddings_buffer)
        embeddings_ds.resize(new_size, axis=0) # Ridimensiona il dataset
        embeddings_ds[current_size:] = embeddings_buffer # Scrivi i nuovi dati

        # Aggiungi gli spettrogrammi
        spectrograms_ds = hf['spectrograms']
        current_size = spectrograms_ds.shape[0]
        new_size = current_size + len(spectrograms_buffer)
        spectrograms_ds.resize(new_size, axis=0) # Ridimensiona il dataset
        spectrograms_ds[current_size:] = spectrograms_buffer # Scrivi i nuovi dati

        # Aggiungi i nomi
        names_ds = hf['names']
        current_size = names_ds.shape[0]
        new_size = current_size + len(names_buffer)
        names_ds.resize(new_size, axis=0)
        names_ds[current_size:] = np.array(names_buffer, dtype=h5py.string_dtype(encoding='utf-8'))

### Indexing for embeddings ###

def load_or_create_emb_index(index_path):
    """Carica l'indice da un file JSON o ne crea uno nuovo."""
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return json.load(f)
    return {}

def save_emb_index(index, index_path):
    """Salva l'indice in un file JSON."""
    with open(index_path, 'w') as f:
        json.dump(index, f)

