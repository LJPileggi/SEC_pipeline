import os
import json
import yaml
import glob
import logging
import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader

__all__ = [
           "get_config_from_yaml",

           "gen_log",
           "read_log",
           "delete_log",

           "HDF5DatasetManager",
           "HDF5EmbeddingDatasetsManager",
           "combine_hdf5_files",

           "setup_environ_vars",
           "setup_distributed_environment",
           "cleanup_distributed_environment",

           "get_reproducible_permutation"
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

### hdf5 raw dataset class ###

class HDF5DatasetManager:
    """
    Gestisce l'accesso rapido ai dati in un file HDF5 (il Super-Dataset), 
    accedendo ai metadati tramite un DataFrame che mantiene l'indice HDF5.
    """
    
    def __init__(self, h5_file_path: str, audio_format: str = 'wav'):
        self.h5_file_path = h5_file_path
        self.hf = None
        self.audio_format = audio_format
        self.metadata_df = None
        self.metadata_dset_name = f'metadata_{self.audio_format}'
        self.audio_dset_name = f'audio_{self.audio_format}'
        
        try:
            # Apri il file in modalità di sola lettura
            self.hf = h5py.File(self.h5_file_path, 'r')
            self._load_metadata()
            logging.info(f"HDF5 Dataset Manager pronto. Formato: {audio_format}")
        except Exception as e:
            logging.error(f"Errore nell'apertura o caricamento metadati HDF5: {e}")
            raise

    def __getitem__(self, hdf5_index: int) -> np.ndarray:
        """
        Accede rapidamente ai dati audio tramite l'indice HDF5.
        Restituisce un array NumPy [samples,].
        """
        # Accede direttamente al dataset VLEN. L'indice di riga corrisponde all'indice HDF5
        return self.hf[self.audio_dset_name][hdf5_index]

    def get_audio_metadata(self, hdf5_index):
        """
        Restituisce i metadati di una data traccia.
        """
        return self.metadata_df[self.metadata_df['hdf5_index'] == hdf5_index]
  
    def _load_metadata(self):
        """Carica il dataset strutturato dei metadati in un DataFrame Pandas."""
        if self.metadata_dset_name not in self.hf:
            raise KeyError(f"Dataset metadati '{self.metadata_dset_name}' non trovato nel file HDF5.")
            
        # Carica l'intero dataset strutturato come un array NumPy
        metadata_array = self.hf[self.metadata_dset_name][:]
        
        # Converti in DataFrame. L'indice di Pandas è l'indice HDF5 implicito.
        self.metadata_df = pd.DataFrame(metadata_array)
        
        # Rinominare l'indice Pandas per chiarezza (anche se non usato)
        self.metadata_df.index.name = 'hdf5_index'
        
        # Aggiungi una colonna esplicita per l'indice HDF5, utile per il debug e i check
        self.metadata_df['hdf5_index'] = self.metadata_df.index

    def get_reproducible_permutation(self, seed: int) -> pd.DataFrame:
        """
        Applica una permutazione fissa al DataFrame di una classe, 
        basata su un seed univoco per quella classe.
    
        Args:
            seed: Seed base per la permutazione (es. dal config file).
        
        Returns:
            DataFrame con l'ordine delle tracce permutato.
        """
        # Combiniamo il seed fisso con un hash della stringa della classe per univocità
        unique_seed = seed + hash(class_name) % 1000000
    
        # Permuta gli indici (non i valori originali)
        # df.sample(frac=1, random_state=...) è il modo più pulito con Pandas
        to_permute = self.metadata_df.copy()
        return to_permute.sample(frac=1, random_state=unique_seed).reset_index(drop=False)

    def close(self):
        """Chiude il file HDF5."""
        if self.hf:
            self.hf.close()
            print(f"HDF5 Dataset Manager chiuso.")
            
    def __del__(self):
        self.close()

class HDF5EmbeddingDatasetsManager(Dataset):
    def __init__(self, h5_path, mode='r', partitions=set('classes', 'splits')):
        """
        Container to handle the hdf5 files for the embedding dataset.
        The dataset is designed as a set of materialised partitions for the
        combination (classes, splits) and (splits,).

        args:
         - h5_path: path to a .h5 file; has to be of the 2 types of partitions specified above;
         - partitions: tuple or set for the types of supported partitions (either (classes, splits) or (splits,)).
        """
        super().__init__()
        self.h5_path = h5_path
        self.partitions = set(partitions)
            if not ((self.partitions == set('splits',)) or (self.partitions == set('classes', 'splits'))):
                raise ValueError("ValueError: incorrect view type.")
        self.mode = mode
        self.hf = h5py.File(self.h5_path, self.mode)
        if 'embedding_dataset' in self.hf:
            self.dt = self._set_dataset_format(self.hf.attrs['embedding_dim'], self.hf.attrs['spec_shape'])
        else:
            self.dt = None
        if self.mode == 'a':
            self.embeddings_buffer = []
            self.spectrograms_buffer = []
            self.hash_keys_buffer = []
            self.track_names_buffer = []
            if self.partitions == set('splits',):
                self.classes_buffer = []
            self.subclasses_buffer = []

    def __len__(self):
        return self.hf.shape[0]

    def __getitem__(self, idx):
        """
        Valid only if object is in read or append mode.
        """
        if self.mode not in ['r', 'a']:
            raise Exception("Exception: can't use getitem method in mode different than read.")
        return self.hf[hf['embedding_dataset']['ID'] == idx]
        

    def _set_dataset_format(self, embedding_dim, spec_shape):
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

    def initialize_hdf5(self, embedding_dim, spec_shape, audio_format, cut_secs, n_octave, \
                                          sample_rate, noise_perc, split, class_name=None):
    """
    Creates HDF5 file with resizable embedding and spectrogram datasets.
    Must provide split and class name according to the selected partition.

    args:
     - embedding_dim: dimension of single embedding;
     - spec_shape: shape of single spectrogram;
     - audio_format: format of the original audio;
     - split: dataset split the embedding belongs to;
     - class_name: class the embedding belongs to.
    """
    if self.mode == 'a':
        self.hf.attrs['audio_format'] = audio_format
        self.hf.attrs['cut_secs'] = cut_secs
        self.hf.attrs['n_octave'] = n_octave
        self.hf.attrs['sample_rate'] = sample_rate
        self.hf.attrs['noise_perc'] = noise_perc
        self.hf.attrs['split'] = split
        self.hf.attrs['embedding_dim'] = embedding_dim
        self.hf.attrs['spec_shape'] = spec_shape
        self.dt = self._set_dataset_format(embedding_dim, spec_shape)
        if 'classes' in self.partitions:
            self.hf.attrs['class'] = class_name
        self.hf.create_dataset('embedding_dataset', 
                                shape=(0,),
                                maxshape=(None,),
                                dtype=self.dt,
                                chunks=True
                                )
        else:
            raise Exception(f'Invalid privileges for {self.h5_path}.')

    def add_to_data_buffer(self, embedding, spectrogram, hash_keys, track_name, class_=None, subclass=None):
        """
        Extends internal buffers to be then flushed to hdf5 file.
        """
        self.embeddings_buffer.append(embedding)
        self.spectrograms_buffer.append(spectrogram)
        self.hash_keys_buffer.append(hash_keys)
        self.track_names_buffer.append(track_name)
        self.classes_buffer.append(class_)
        self.subclasses_buffer.append(subclass)

    def flush_buffers(self):
        """
        Flushes buffers to hdf5 file.
        """
        data_buffer = list(zip(
                    self.hash_keys_buffer,
                    self.embeddings_buffer,
                    self.spectrograms_buffer,
                    [s.encode('utf-8') for s in self.track_names_buffer],
                    [s.encode('utf-8') for s in self.subclasses_buffer]
            )) if 'classes' in self.partitions else list(zip(
                    self.hash_keys_buffer,
                    self.embeddings_buffer,
                    self.spectrograms_buffer,
                    [s.encode('utf-8') for s in self.track_names_buffer],
                    [s.encode('utf-8') for s in self.classes_buffer],
                    [s.encode('utf-8') for s in self.subclasses_buffer]
            ))
        data_buffer = numpy.array(data_buffer, dtype=self.dt)

        dataset = hf['embedding_dataset']
        current_size = dataset.shape[0]
        new_size = current_size + len(data_buffer)
        dataset.resize(new_size, axis=0)
        dataset[current_size:] = data_buffer

    def extend_dataset(self, new_data):
        """
        Extends dataset content directly without going through the buffers.
        Data has to be compatible with the native dtype of the dataset.
        """
        if new_data.dtype != self.dt:
            raise TypeError(f"TypeError: new data dtype {new_data.dtype} is "
                            f"incompatible with native dataset dtype {self.dt}")
        dataset = self.hf['embedding_dataset']
        current_size = dataset.shape[0]
        new_size = current_size + len(new_data)
        dataset.resize(new_size, axis=0)
        dataset[current_size:] = new_data

    def close(self):
        if self.hf:
            self.hf.close()
            print(f"HDF5 Dataset Manager chiuso.")
            
    def __del__(self):
        self.close()

def combine_hdf5_files(root_dir, cut_secs_list, audio_format, splits_list=['train', 'es', 'valid', 'test'], \
                                                                embedding_dim=1024, spec_shape=(128, 1024)):
    """
    Combines individual HDF5 files for each class and split into unified HDF5 files
    for each split.

    Args:
        root_dir (str): The root directory where the cut_secs directories are located.
        cut_secs_list (list): A list of cut_secs values (e.g., [1, 2, 4]).
        audio_format (str): Original audio format
        splits_list (list): A list of data splits (e.g., ['train', 'es', 'valid', 'test']).
        embedding_dim (int): The dimension of the embeddings.
        spec_shape (tuple): The shape of the spectrograms.
    """
    classes_list = sorted([d for d in os.listdir(os.path.join(root_dir, f'{cut_secs_list[0]}_secs')) \
                            if os.path.isdir(os.path.join(root_dir, f'{cut_secs_list[0]}_secs', d))])
    for cut_secs in cut_secs_list:
        logging.info(f"Processing cut_secs: {cut_secs}...")
        
        for split_name in splits_list:
            output_h5_path = os.path.join(root_dir, f'{cut_secs}_secs', f'combined_{split_name}.h5')
            out_h5 = HDF5EmbeddingDatasetsManager(output_h5_path, mode='a', partitions=set('splits',))
            out_h5.initialize_hdf5(embedding_dim, spec_shape, audio_format, split_name)

            for class_name in classes_list:
                class_h5_path = os.path.join(root_dir, f'{cut_secs}_secs', class_name, f'{class_name}_{split_name}.h5')
                    
                if not os.path.exists(class_h5_path):
                    logging.warning(f"File non trovato per la classe '{class_name}' e split '{split_name}': {class_h5_path}. Salto.")
                    continue
                    
                logging.info(f"Adding data from class: {class_name}...")
                    
                try:
                    in_h5 = HDF5EmbeddingDatasetsManager(class_h5_path, 'r', set('splits', 'classes'))
                    class_data = in_h5['embedding_dataset'][:]
                    class_data_extended = np.empty(class_data.shape, dtype=out_h5.dt)
                    for name in class_data_extended.names:
                        class_data_extended[name] = class_data[name]
                    class_data_extended['classes'] = [class_name] * len(class_data)
                    out_h5.extend_dataset(class_data_extended)
                except Exception as e:
                    logging.error(f"Errore durante l'unione dei file di classe '{class_name}': {e}. Continuo.")

            logging.info(f"Combinazione completata per '{split_name}'. File salvato in: {output_h5_path}")

def get_track_reproducibility_parameters(self, idx):
    """
    Gives a dictionary containing all the information to reconstruct the track
    an embedding was generated from. Needs the ID key for that embedding.

    args:
     - idx: embedding key given in the required format
       ((class_idx)_(track hdf5 index)_(bucket number)_(round_)_(offset seed)_(results number)_(noise seed)).

    returns:
     - rep_params: dictionary containing all the parameters to reconstruct the track.
    """
    rep_params = {}
    params = idx.split('_')
    param_names = [
            "class_idx",
            "hdf5_index",
            "bucket",
            "round_",
            "offset_seed",
            "results",
            "noise_seed"
        ]
    for param, name in zip(params, param_names):
        rep_params[name] = param
    return rep_params

def reconstruct_tracks_from_embeddings(base_tracks_dir, hdf5_emb_path, idx_list):
    """
    Reconstructs a bunch of tracks relative to a given group of embeddings from their unique
    indexing. Indices are written in such a way to parse all the information needed for the
    reconstruction through the function get_track_reproducibility_parameters.

    args:
     - base_tracks_dir (str): base dir containing the various tracks datasets;
       to be combined with the information coming from the metadata to get the
       correct hdf5 file;
     - hdf5_emb_path (str): path to the hdf5 file of the desired embeddings;
       has to be relative to an entire split for completeness;
     - idx_list (list): list containing the embedding indices formatted in
       the appropriate way:
       ((class)_(track hdf5 index)_(bucket number)_(round_)_(offset seed)_(results number)_(noise seed));

    returns:
     - reconstr_tracks (dict): dict of the recontructed tracks
    """
    hdf5_emb = HDF5EmbeddingDatasetsManager(hdf5_emb_path, 'r', ('splits',))
    audio_format = hdf5_emb.hf.attrs['audio_format']
    cut_secs = hdf5_emb.hf.attrs['cut_secs']
    sample_rate = hdf5_emb.hf.attrs['sample_rate']
    noise_perc = hdf5_emb.hf.attrs['noise_perc']
    classes_list = hdf5_emb.hf['embedding_dataset']['classes'].unique().sort()
    repr_params_list = [get_track_reproducibility_parameters(idx) for idx in idx_list]
    repr_params_list = sorted(repr_params_list, key=lambda a : a['class_idx'])
    curr_class = None
    reconstr_tracks = {}

    for track_idx, repr_params in zip(idx_list, repr_params_list):
        if repr_params['class_idx'] != curr_class:
            hdf5_class_tracks.close()
            curr_class = repr_params['class_idx']
            hdf5_class_path = os.path.join(base_tracks_dir, f'raw_{audio_format}', f'{classes_list[curr_class]}_{audio_format}_dataset.h5')
            hdf5_class_tracks = HDF5DatasetManager(hdf5_class_path, audio_format)
        original_track = hdf5_class_tracks[repr_params['hdf5_index']]

        # set random number generator to reconstruct offset and noise
        offset_rng = np.random.default_rng(repr_params['offset_seed'])
        noise_rng = np.random.default_rng(repr_params['noise_seed'])

        # generate right offset
        offset = 0
        window_size = int(cut_secs * sample_rate)
        if repr_params['round_'] > 1 and original_track.shape[0] > window_size:
            max_offset = original_track.shape[0] - window_size
            if max_offset > 0:
                for _ in range(repr_params['round_'] + 1)
                    offset = offset_rng.integers(0, max_offset)

        # generate noise
        start = repr_params['bucket'] * window_size + offset
        end = start + window_size
        cut_track = original_track[start:end]

        if len(cut_track) < window_size:
            pad_length = window_size - len(cut_track)
            cut_track = np.pad(cut_track, (0, pad_length), 'constant')

        abs_cut_track = np.abs(cut_track)
        max_threshold = np.mean(abs_cut_track)
        for _ in range(repr_params['results']):
            noise = noise_rng.uniform(-max_threshold, max_threshold, cut_track.shape)
        reconstr_track = (1 - noise_perc) * cut_track + noise_perc * noise
        reconstr_tracks[track_idx] = reconstr_track

    hdf5_class_tracks.close()
    return reconstr_tracks

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
    return device

def cleanup_distributed_environment():
    """Cleanup the distributed environment."""
    dist.barrier()
    dist.destroy_process_group()
    logging.info(f"Processo {rank} ha terminato il suo lavoro.")

