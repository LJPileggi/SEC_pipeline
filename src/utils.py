import os
import json
import yaml
import glob
import logging
import h5py
import numpy as np
from torch.utils.data import Dataset, DataLoader

__all__ = [
           "get_config_from_yaml",

           "extract_all_files_from_dir",

           "gen_log",
           "read_log",
           "delete_log",

           "initialize_hdf5",
           "append_to_hdf5",
           "combine_hdf5_files",
           "load_octaveband_datasets",

           "load_or_create_emb_index",
           "save_emb_index",

           "setup_rank_and_world_size",
           "setup_distributed_environment",
           "cleanup_distributed_environment"
          ]

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
    splits_xc_sizes_names = [("train", configs["train_size"]),
                                ("es", configs["es_size"]),
                                ("valid", configs["valid_size"]),
                                ("test", configs["test_size"])]
    return patience, epochs, batch_size, save_log_every, sampling_rate, ref, noise_perc, \
                            seed, center_freqs, valid_cut_secs, splits_xc_sizes_names

   
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
                    splits_xc_sizes_names, noise_perc, seed, rank):
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
     - splits_xc_sizes_names: a list containing tuples of ('split_name', #samples);
       the usual one is [('train', 500), ('es', 100), ('valid', 100), ('test', 100)];
     - noise_perc: intensity of noise for data augmentation;
     - seed: random seed for track permutations inside the same class;
     - rank: rank of the log generating process to confront different logs and
       sort them lexicographically.
    """
    lengths = list(zip(*splits_xc_sizes_names))[1]
    results = sum(lengths[:di]) if di > 0 else 0
    log = {
        "cut_secs" : cut_secs,
        "ic" : ic,
        "di" : di,
        "results" : results,
        "round" : round_,
        "finish_class" : finish_class,
        "splits_xc_sizes_names" : splits_xc_sizes_names,
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
          f"splits_xc_sizes_names: {splits_xc_sizes_names}\n"
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

def initialize_hdf5(file_path, embedding_dim, spec_shape, split):
    """
    Creates HDF5 file with resizable embedding and spectrogram datasets.

    args:
     - file_path: path of HDF5 file;
     - embedding_dim: dimension of single embedding;
     - spec_shape: shape of single spectrogram;
     - split: dataset split the embedding belongs to.
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

def append_to_hdf5(file_path, embeddings_buffer, spectrograms_buffer, names_buffer):
    """
    Appends embedding and spectrogram batches to preexisting HDF5 file.

    args:
     - file_path: path of HDF5 file to append data;
     - embeddings_buffer: buffer of embeddings to append;
     - spectrograms_buffer: buffer of spectrograms to append;
     - names_buffer: buffer of embeddings' IDs.
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

def combine_hdf5_files(root_dir, cut_secs_list, splits_list=['train', 'es', 'valid', 'test'], embedding_dim=1024, spec_shape=(128, 1024)):
    """
    Combines individual HDF5 files for each class and split into unified HDF5 files
    for each split.

    Args:
        root_dir (str): The root directory where the cut_secs directories are located.
        cut_secs_list (list): A list of cut_secs values (e.g., [1, 2, 4]).
        splits_list (list): A list of data splits (e.g., ['train', 'valid', 'test']).
        embedding_dim (int): The dimension of the embeddings.
        spec_shape (tuple): The shape of the spectrograms.
    """
    classes_list = sorted([d for d in os.listdir(os.path.join(root_dir, f'{cut_secs_list[0]}_secs')) \
                            if os.path.isdir(os.path.join(root_dir, f'{cut_secs_list[0]}_secs', d))])
    for cut_secs in cut_secs_list:
        logging.info(f"Processing cut_secs: {cut_secs}...")
        
        for split_name in splits_list:
            output_h5_path = os.path.join(root_dir, f'{cut_secs}_secs', f'combined_{split_name}.h5')
            
            # Crea un nuovo file HDF5 per lo split corrente
            with h5py.File(output_h5_path, 'w') as out_h5:
                # Inizializza i dataset nel file unificato
                dt = h5py.string_dtype(encoding='utf-8')
                out_h5.create_dataset('embeddings', shape=(0, embedding_dim), maxshape=(None, embedding_dim), dtype='f4', chunks=True)
                out_h5.create_dataset('spectrograms', shape=(0,) + spec_shape, maxshape=(None,) + spec_shape, dtype='f4', chunks=True)
                out_h5.create_dataset('names', shape=(0,), maxshape=(None,), dtype=dt, chunks=True)
                out_h5.create_dataset('classes', shape=(0,), maxshape=(None,), dtype=dt, chunks=True)

                for class_name in classes_list:
                    class_h5_path = os.path.join(root_dir, f'{cut_secs}_secs', class_name, f'{class_name}_{split_name}.h5')
                    
                    if not os.path.exists(class_h5_path):
                        logging.warning(f"File non trovato per la classe '{class_name}' e split '{split_name}': {class_h5_path}. Salto.")
                        continue
                    
                    logging.info(f"Adding data from class: {class_name}...")
                    
                    try:
                        with h5py.File(class_h5_path, 'r') as in_h5:
                            # Aggiungi i dati al file unificato
                            in_embeddings = in_h5['embeddings'][:]
                            in_spectrograms = in_h5['spectrograms'][:]
                            in_names = in_h5['names'][:]
                            
                            current_size = out_h5['embeddings'].shape[0]
                            new_size = current_size + len(in_embeddings)
                            
                            out_h5['embeddings'].resize(new_size, axis=0)
                            out_h5['spectrograms'].resize(new_size, axis=0)
                            out_h5['names'].resize(new_size, axis=0)
                            out_h5['classes'].resize(new_size, axis=0)
                            
                            out_h5['embeddings'][current_size:] = in_embeddings
                            out_h5['spectrograms'][current_size:] = in_spectrograms
                            out_h5['names'][current_size:] = in_names
                            out_h5['classes'][current_size:] = [class_name] * len(in_embeddings)

                    except Exception as e:
                        logging.error(f"Errore durante l'unione dei file di classe '{class_name}': {e}. Continuo.")

            logging.info(f"Combinazione completata per '{split_name}'. File salvato in: {output_h5_path}")

class HDF5Dataset(Dataset):
    def __init__(self, h5_path, data_types):
        super().__init__()
        if not data_types:
            self._data_types = ['embeddings']
        for data_type in set(data_types):
            if data_type not in ['embeddings', 'spectrograms', 'names']:
                raise ValueError("ValueError: incorrect data type for Dataset.")
        self._data_types = list(set(data_types))
        self.h5_path = h5_path
        # Apri il file HDF5 in modalità di sola lettura
        self.hdf5_file = h5py.File(self.h5_path, 'r')
        
        # Ottieni i riferimenti ai dataset
        self.embeddings = self.hdf5_file['embeddings']
        self.spectrograms = self.hdf5_file['spectrograms']
        self.names = self.hdf5_file['names']
        self.classes = self.hdf5_file['classes']

    def __len__(self):
        # Restituisce il numero totale di elementi nel dataset
        return self.embeddings.shape[0]

    def __getitem__(self, idx):
        # Carica il campione corrispondente all'indice idx
        embedding = torch.from_numpy(self.embeddings[idx]).float()
        spectrogram = torch.from_numpy(self.spectrograms[idx]).float()
        name = self.names[idx]
        class_ = self.classes[idx]
        item = ()
        if 'embeddings' in self._data_types:
            item = item + (embedding,)
        if 'spectrograms' in self._data_types:
            item = item + (spectrogram,)
        if 'names' in self._data_types:
            item = item + (name,)
        
        # Restituisci il campione
        return item + (class_,)

    def close(self):
        # Chiudi il file HDF5
        self.hdf5_file.close()

def load_octaveband_datasets(octaveband_dir, batch_size, data_types):
    """
    Generates lists of DataLoaders for embeddings and/or spectrograms
    of a given octaveband run.
    
    args:
     - octaveband_dir: directory for a given octaveband run;
     - batch_size: batch size for DataLoaders;
     - data_types: list of types of data to be yielded by the Dataset items;
       must choose between a combination of 'embeddings', 'spectrograms' and
       'names'.

    returns:
     - dataloaders: list of DataLoader objects containing the datasets.
    """
    dataloaders = {fp: [torch.utils.data.DataLoader(HDF5Dataset(
        h5_path=os.path.join(octaveband_dir, fp, f'combined_{type_ds}.h5'),
        data_types=data_types
    ), batch_size=batch_size, shuffle=True) for type_ds in ['train', 'es', 'valid', 'test']] for fp in os.listdir(octaveband_dir)}
    return dataloaders

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

### Distributed environment functions ###

def setup_environ_vars(slurm=True):
    if slurm:
        rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 4))
        return rank, world_size
    else:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29500'

def setup_distributed_environment(rank, world_size, slurm=True):
    """
    Setup the distributed environment.

    args:
     - rank: process rank;
     - world_size: n. of distributed units;
     - slurm: whether we are running on a SLURM environment or not; default to True.
    """
    if slurm:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')
        logging.info(f"Processo locale {rank} avviato su {device}.")
    else:
        dist.init_process_group("gloo", rank=rank, world_size=world_size) # Usiamo il backend 'gloo' per la CPU
    
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if torch.cuda.is_available():
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
        logging.info(f"Processo locale {rank} avviato su {device}.")

def cleanup_distributed_environment():
    """Cleanup the distributed environment."""
    dist.barrier()
    dist.destroy_process_group()
    logging.info(f"Processo {rank} ha terminato il suo lavoro.")


