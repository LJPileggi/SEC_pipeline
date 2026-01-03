import os
import json
import yaml
import glob
import logging
import gc
import h5py
import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
import traceback

__all__ = [
           "get_config_from_yaml",

           "write_log",
           "join_logs",

           "HDF5DatasetManager",
           "HDF5EmbeddingDatasetsManager",
           "combine_hdf5_files",

           "get_track_reproducibility_parameters",
           "reconstruct_tracks_from_embeddings",

           "setup_environ_vars",
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
    classes = configs["classes"]
    patience = configs["patience"]
    epochs = configs["epochs"]
    batch_size = configs["batch_size"]
    sampling_rate = configs["sampling_rate"]
    ref = configs["ref"]
    noise_perc = configs["noise_perc"]
    seed = configs["seed"]
    if configs.get("center_freqs"):
        center_freqs = np.array(configs["center_freqs"])
    else:
        center_freqs = None
    valid_cut_secs = configs["valid_cut_secs"]
    splits_xc_sizes_names = [("train", configs["train_size"]),
                                ("es", configs["es_size"]),
                                ("valid", configs["valid_size"]),
                                ("test", configs["test_size"])]
    return classes, patience, epochs, batch_size, sampling_rate, ref, noise_perc, seed, \
                                  center_freqs, valid_cut_secs, splits_xc_sizes_names

   
### Log file functions for embedding calculation ###

def write_log(log_path, new_cut_secs_class, process_time, n_embeddings_per_run, completed, **kwargs):
    os.makedirs(log_path, exist_ok=True) #
    
    new_cut_secs_class = str(new_cut_secs_class)
    logfile = os.path.join(log_path, f"log_rank_{kwargs['rank']}.json") #
    
    log = {"config": {}}
    if os.path.exists(logfile):
        try:
            with open(logfile, 'r') as f:
                log = json.load(f)
        except (json.JSONDecodeError, Exception):
            pass

    if not log.get("config"): 
        log["config"].update(kwargs)

    # Poiché ora il file log_rank_X.json accoglie più classi, 
    # usiamo la tupla come chiave per non sovrascrivere i dati
    if new_cut_secs_class not in log:
        log[new_cut_secs_class] = {"process_time": [], "n_embeddings_per_run": [], "rank": []}
        
    log[new_cut_secs_class]["process_time"].append(process_time)
    log[new_cut_secs_class]["n_embeddings_per_run"].append(n_embeddings_per_run)
    log[new_cut_secs_class]["rank"].append(kwargs["rank"])
    log[new_cut_secs_class]["completed"] = completed

    with open(logfile, 'w') as f:
        json.dump(log, f, indent=4)
        f.flush() # 🎯 Forza la scrittura fisica
        os.fsync(f.fileno()) # 🎯 Sincronizza col sistema operativo

def join_logs(log_dir):
    """
    Unisce i log dei vari rank presenti in una cartella di cut_secs,
    preservando i dati di tutte le classi processate.
    """
    final_log = {"config": {}}
    final_log_file = os.path.join(log_dir, "log.json")
    pattern = os.path.join(log_dir, "log_rank_*.json")
    log_files = glob.glob(pattern)
    
    if not log_files:
        return

    for log_file in log_files:
        try:
            with open(log_file, 'r') as f:
                current_rank_log = json.load(f)
                
                for key, data in current_rank_log.items():
                    if key == "config":
                        if not final_log["config"]:
                            final_log["config"].update(data)
                        continue

                    # 🎯 LOGICA DI MERGE ROBUSTA
                    # Se la chiave (es. "(1, 'Birds')") esiste già, estendiamo le liste.
                    # Questo è fondamentale se più rank hanno lavorato sulla stessa classe.
                    if key in final_log:
                        for field in ["process_time", "n_embeddings_per_run", "rank"]:
                            if field in data:
                                final_log[key][field].extend(data[field])
                        # Il task è completato solo se lo sono tutti i segmenti
                        final_log[key]["completed"] = final_log[key].get("completed", True) and data.get("completed", False)
                    else:
                        # Se è una nuova classe per questo cut_secs, la aggiungiamo interamente
                        final_log[key] = data
                        
        except Exception as e:
            print(f"Errore durante l'unione del log {log_file}: {e}")

    # Scrittura del file unificato finale per il cut_secs corrente
    with open(final_log_file, 'w') as f:
        json.dump(final_log, f, indent=4)
        
    # Cleanup
    # for log_file in log_files:
    #     os.remove(log_file)

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
        self.metadata_dset_name = f'metadata_{self.audio_format}'
        self.audio_dset_name = f'audio_{self.audio_format}'
        self.n_records = 0 # 🎯 Gestione lazy
        
        try:
            self.hf = h5py.File(self.h5_file_path, 'r', rdcc_nbytes=0)
            self.n_records = self.hf[self.metadata_dset_name].shape[0]
            logging.info(f"HDF5 Manager pronto: {self.n_records} tracce rilevate.")
        except Exception as e:
            logging.error(f"Errore apertura {h5_file_path}: {e}")
            raise

    def get_audio_and_metadata(self, hdf5_index):
        """🎯 Legge direttamente da HDF5 senza passare per Pandas."""
        audio = self.hf[self.audio_dset_name][hdf5_index]
        # Recupera la riga di metadati come numpy void object
        raw_meta = self.hf[self.metadata_dset_name][hdf5_index]
        # Converte in dizionario semplice per minimizzare l'overhead
        meta_dict = {n: (raw_meta[n].decode('utf-8') if isinstance(raw_meta[n], bytes) else raw_meta[n]) 
                     for n in raw_meta.dtype.names}
        return audio, meta_dict

    def get_reproducible_permutation(self, seed: int):
        """🎯 Permuta solo gli indici numerici, non gli oggetti."""
        rng = np.random.default_rng(seed)
        return rng.permutation(np.arange(self.n_records))

    def close(self):
        if self.hf:
            self.hf.close()
            self.hf = None
        gc.collect()

    def __del__(self):
        """Ripristinato per sicurezza, ma l'azione reale è in close()."""
        try:
            if self.hf: self.hf.close()
        except: pass

class HDF5EmbeddingDatasetsManager(Dataset):
    def __init__(self, h5_path, mode='r', partitions=set(('classes', 'splits')), buffer_size=10):
        super().__init__()
        self.h5_path = h5_path
        self.partitions = set(partitions)
        self.mode = mode
        self.hf = None
        self.buffer_size = buffer_size
        self.buffer_count = 0
        self.existing_keys = set()
        self.dt = None           
        self.buffer_array = None 
        
        try:
            # Apertura del file con retry logica interna di h5py se possibile
            self.hf = h5py.File(self.h5_path, self.mode, rdcc_nbytes=0)
        except Exception as e:
            logging.error(f"Impossibile aprire il file {h5_path}: {e}")
            raise

        # 🎯 RECUPERO METADATI E CHIAVI (Fondamentale per la Resumability)
        if self.hf is not None and 'embedding_dataset' in self.hf:
            try:
                # 1. Recuperiamo i parametri strutturali dagli attributi del file
                emb_dim = self.hf.attrs.get('embedding_dim')
                spec_sh = self.hf.attrs.get('spec_shape')
                
                if emb_dim is not None and spec_sh is not None:
                    # Ricostruiamo il formato del dataset (dtype)
                    self.dt = self._set_dataset_format(emb_dim, spec_sh)
                    
                    # 2. Caricamento sicuro delle chiavi ID per il check O(1)
                    dset = self.hf['embedding_dataset']
                    if dset.shape[0] > 0:
                        # Leggiamo gli ID. Usiamo una slice [:] per caricare in memoria
                        raw_ids = dset['ID'][:]
                        self.existing_keys = {k.decode('utf-8') if isinstance(k, bytes) else k for k in raw_ids}
                    
                    # 3. Inizializziamo il buffer NumPy solo se siamo in modalità append
                    if self.mode == 'a':
                        self.buffer_array = np.empty(self.buffer_size, dtype=self.dt)
            except Exception as e:
                # Se il file è corrotto o incompleto, logghiamo ma non crashiamo il worker
                logging.warning(f"Manager: Errore durante il caricamento dei metadati da {h5_path}: {e}")

    def __len__(self):
        if 'embedding_dataset' in self.hf:
            return self.hf['embedding_dataset'].shape[0]
        return 0

    def __contains__(self, emb_pkey: str) -> bool:
        """
        Verifica se un embedding esiste già usando il lookup O(1) in memoria.
        """
        # 🎯 Lookup istantaneo: nessuna lettura da disco
        return emb_pkey in self.existing_keys

    def __getitem__(self, idx):
        """
        Metodo modificato per gestire sia l'accesso per chiave (stringa) che l'accesso per indice (numero).
        """
        if isinstance(idx, str):
            # Se l'indice è una stringa, lo interpreta come una richiesta di esistenza (contains)
            # e delega al metodo __contains__.
            return self.__contains__(idx) 
    
        # Altrimenti, gestisce l'accesso per indice numerico standard (es. riga 0, 1, 2...)
        return self.hf['embedding_dataset'][idx]
        
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

    def initialize_hdf5(self, embedding_dim, spec_shape, audio_format, cut_secs, n_octave, 
                        sample_rate, seed, noise_perc, split, class_name=None):
        """
        Inizializza il file HDF5 e configura i buffer necessari per l'append.
        Salva i metadati come attributi per permettere all'__init__ di ricaricarli in caso di resume.
        """
        if self.mode == 'a':
            # 🎯 SALVATAGGIO ATTRIBUTI (Metadati strutturali)
            self.hf.attrs['audio_format'] = audio_format
            self.hf.attrs['cut_secs'] = cut_secs
            self.hf.attrs['n_octave'] = n_octave
            self.hf.attrs['sample_rate'] = sample_rate
            self.hf.attrs['noise_perc'] = noise_perc
            self.hf.attrs['seed'] = seed
            self.hf.attrs['split'] = split
            self.hf.attrs['embedding_dim'] = embedding_dim
            self.hf.attrs['spec_shape'] = spec_shape # Fondamentale per ricostruire il buffer NumPy
            
            # Setup formato dataset e buffer pre-allocato
            self.dt = self._set_dataset_format(embedding_dim, spec_shape)
            self.buffer_array = np.empty(self.buffer_size, dtype=self.dt)
            
            if 'classes' in self.partitions:
                self.hf.attrs['class'] = class_name
            
            # Creazione fisica del dataset se non esiste
            if 'embedding_dataset' not in self.hf:
                # Usiamo chunks=True per permettere il ridimensionamento dinamico
                self.hf.create_dataset('embedding_dataset', 
                                        shape=(0,),
                                        maxshape=(None,),
                                        dtype=self.dt,
                                        chunks=True)
            
            # 🎯 FORZA SCRITTURA METADATI
            # Questo assicura che se il processo crasha subito dopo, l'__init__ troverà almeno gli attributi
            self.hf.flush()
        else:
            raise Exception(f'Permessi non validi per la scrittura su {self.h5_path}.')

    def add_to_data_buffer(self, embedding, spectrogram, hash_keys, track_name, class_=None, subclass=None):
        """
        Scrive i dati nel buffer NumPy gestendo correttamente la codifica delle stringhe.
        """
        idx = self.buffer_count
        
        # Funzione helper interna per codificare solo se necessario
        def to_bytes(s):
            if isinstance(s, str):
                return s.encode('utf-8')
            return s # È già bytes o None

        # Inserimento con controllo del tipo
        self.buffer_array[idx]['ID'] = to_bytes(hash_keys)
        
        if torch.is_tensor(embedding):
            self.buffer_array[idx]['embeddings'] = embedding.detach().cpu().numpy()
        else:
            self.buffer_array[idx]['embeddings'] = embedding
            
        self.buffer_array[idx]['spectrograms'] = spectrogram
        self.buffer_array[idx]['track_names'] = to_bytes(track_name)
        
        if 'classes' not in self.partitions:
            self.buffer_array[idx]['classes'] = to_bytes(class_) if class_ else b''
        
        self.buffer_array[idx]['subclasses'] = to_bytes(subclass) if subclass else b''
        
        self.buffer_count += 1
        self.existing_keys.add(hash_keys if isinstance(hash_keys, str) else hash_keys.decode('utf-8'))
        
        if self.buffer_count >= self.buffer_size:
            self.flush_buffers()

    def flush_buffers(self):
        """Scrive su disco e pulisce i riferimenti agli oggetti pesanti."""
        if self.buffer_count == 0:
            return

        # 🎯 Ripristinata la logica della variabile d'ambiente
        if os.getenv('NO_EMBEDDING_SAVE', 'False').lower() in ('false', '0', 'f'):
            dataset = self.hf['embedding_dataset']
            current_size = dataset.shape[0]
            dataset.resize(current_size + self.buffer_count, axis=0)
            dataset[current_size:] = self.buffer_array[:self.buffer_count]

        # 🎯 TRUCCO FINALE: Invece di fill(0), azzeriamo il riferimento
        # Questo costringe NumPy a rilasciare il grosso blocco di memoria C
        self.buffer_array = np.empty(self.buffer_size, dtype=self.dt)
        self.buffer_count = 0
        gc.collect()

    def extend_dataset(self, new_data):
        """
        Estende il dataset direttamente bypassando i buffer. 
        Aggiorna il set delle chiavi per mantenere la coerenza della Latenza 1.
        """
        if isinstance(new_data, dict):
            if 'ID' not in new_data:
                 raise ValueError("Missing 'ID' key in dictionary data.")
            num_new_elements = len(new_data['ID'])

            # Pre-allocazione temporanea per la conversione del dizionario
            new_data_array = np.empty(num_new_elements, dtype=self.dt)

            for name in self.dt.names:
                if name in new_data:
                    new_data_array[name] = new_data[name]
                else:
                    raise TypeError(f"Missing {name} in to-be-added data.")
            new_data = new_data_array

        if new_data.dtype != self.dt:
            raise TypeError(f"New data dtype {new_data.dtype} is "
                            f"incompatible with native dataset dtype {self.dt}")
        
        # --- Scrittura su HDF5 ---
        dataset = self.hf['embedding_dataset']
        current_size = dataset.shape[0]
        new_size = current_size + len(new_data)
        
        dataset.resize(new_size, axis=0)
        dataset[current_size:] = new_data # Scrittura efficiente per blocchi
        
        # --- 🎯 AGGIORNAMENTO LATENZA 1 ---
        # Decodifichiamo e aggiungiamo le nuove chiavi al set in memoria
        new_ids = new_data['ID']
        self.existing_keys.update(k.decode('utf-8') if isinstance(k, bytes) else k for k in new_ids)

    def close(self):
        """Assicura l'ultimo scarico dati prima della chiusura."""
        if self.hf:
            self.flush_buffers() 
            try:
                self.hf.close()
            except Exception:
                pass
            self.hf = None
        self.buffer_array = None

    def __del__(self):
        pass

def combine_hdf5_files(root_dir, cut_secs_list, embedding_dim, spec_shape, audio_format, cut_secs, n_octave, 
                                             sample_rate, seed, noise_perc, splits_list):
    """
    Combines individual HDF5 files for each class and split into unified HDF5 files
    for each split.
    # ... (omissione documentazione)
    """
    classes_list = sorted([d for d in os.listdir(os.path.join(root_dir, f'{cut_secs_list[0]}_secs')) 
                            if os.path.isdir(os.path.join(root_dir, f'{cut_secs_list[0]}_secs', d))])
    
    for current_cut_secs in cut_secs_list:
        
        for split_tuple in splits_list:
            split_name = split_tuple[0]
            output_h5_path = os.path.join(root_dir, f'{current_cut_secs}_secs', f'combined_{split_name}.h5')
            
            # Setup Manager di Output (out_h5)
            out_h5 = HDF5EmbeddingDatasetsManager(output_h5_path, mode='a', partitions=set(('splits',)))
            
            out_h5.initialize_hdf5(embedding_dim, spec_shape, audio_format, current_cut_secs, n_octave,
                                             sample_rate, seed, noise_perc, split_name)

            for class_name in classes_list:
                class_h5_path = os.path.join(root_dir, f'{current_cut_secs}_secs', class_name, f'{class_name}_{split_name}.h5')
                        
                if not os.path.exists(class_h5_path):
                    print(f"[WARNING] File non trovato per la classe '{class_name}' e split '{split_name}': {class_h5_path}. Salto.")
                    continue
                        
                print(f"Aggiunta dati dalla classe: {class_name}...")
                        
                try:
                    # Uso HDF5EmbeddingDatasetsManager per la lettura (in_h5)
                    in_h5 = HDF5EmbeddingDatasetsManager(class_h5_path, 'r', set(('splits', 'classes')))
                    
                    # Lettura dei dati. Se fallisce qui, avremo un traceback.
                    class_data = in_h5['embedding_dataset'][:]
                    
                    class_data_extended = np.empty(class_data.shape, dtype=out_h5.dt)
                    
                    # FIX: Copia dei campi usando .dtype.names (corretto nell'ultima iterazione)
                    for name in class_data_extended.dtype.names:
                        if name in class_data.dtype.names: 
                            class_data_extended[name] = class_data[name]

                    # Aggiunta del metadato 'classes'
                    if 'classes' in class_data_extended.dtype.names:
                        class_data_extended['classes'] = np.array([class_name.encode('utf-8')] * len(class_data),
                                                                    dtype=class_data_extended.dtype['classes'])
                        
                    # CHIAMATA CRUCIALE: Se non viene raggiunta, c'è un errore prima.
                    out_h5.extend_dataset(class_data_extended)
                    
                    # Chiusura del manager di input
                    in_h5.close()
                    
                except Exception as e:
                    print("\n" + "="*80)
                    print(f"FATAL ERROR - Eccezione catturata durante l'unione dei file di classe '{class_name}' (Split {split_name}): {e}")
                    # Stampa l'eccezione completa per il debug
                    traceback.print_exc() 
                    print("="*80 + "\n")
                    # Continua l'esecuzione
                    
            # Chiusura del manager di output
            out_h5.close() 

def get_track_reproducibility_parameters(idx):
    """
    Gives a dictionary containing all the information to reconstruct the track
    an embedding was generated from. Needs the ID key for that embedding.

    args:
     - idx: embedding key given in the required format
       ((class_idx)_(track hdf5 index)_(bucket number)_(round_)_(results number)).

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
            "results"
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
       ((class_idx)_(track hdf5 index)_(bucket number)_(round_)_(results number));

    returns:
     - reconstr_tracks (dict): dict of the recontructed tracks
    """
    hdf5_emb = HDF5EmbeddingDatasetsManager(hdf5_emb_path, 'r', ('splits',))
    audio_format = hdf5_emb.hf.attrs['audio_format']
    cut_secs = hdf5_emb.hf.attrs['cut_secs']
    sample_rate = hdf5_emb.hf.attrs['sample_rate']
    noise_perc = hdf5_emb.hf.attrs['noise_perc']
    seed = hdf5_emb.hf.attrs['seed']
    classes_list = hdf5_emb.hf['embedding_dataset']['classes'].unique().sort()
    repr_params_list = [get_track_reproducibility_parameters(idx) for idx in idx_list]
    repr_params_list = sorted(repr_params_list, key=lambda a : a['class_idx'])
    curr_class = None
    reconstr_tracks = {}

    for track_idx, repr_params in zip(idx_list, repr_params_list):
        if repr_params['class_idx'] != curr_class:
            if curr_class:
                hdf5_class_tracks.close()
            curr_class = repr_params['class_idx']
            class_seed = seed + hash(classes_list[curr_class]) % 10000000
            hdf5_class_path = os.path.join(base_tracks_dir, f'raw_{audio_format}', f'{classes_list[curr_class]}_{audio_format}_dataset.h5')
            hdf5_class_tracks = HDF5DatasetManager(hdf5_class_path, audio_format)
        original_track = hdf5_class_tracks[repr_params['hdf5_index']]

        # set random number generator to reconstruct offset and noise
        offset_rng = np.random.default_rng(class_seed)
        noise_rng = np.random.default_rng(class_seed)

        # generate right offset
        offset = 0
        window_size = int(cut_secs * sample_rate)
        if repr_params['round_'] > 1 and original_track.shape[0] > window_size:
            max_offset = original_track.shape[0] - window_size
            if max_offset > 0:
                for _ in range(repr_params['round_'] + 1):
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

    hdf5_emb.close()
    hdf5_class_tracks.close()
    return reconstr_tracks

### Distributed environment functions ###

def setup_environ_vars(slurm=True):
    if slurm:
        rank = int(os.environ.get("SLURM_PROCID", 0))
        world_size = int(os.environ.get("SLURM_NTASKS", 4))
        
        # 🎯 1. MASTER_ADDR: Mai usare localhost su SLURM. 
        # Usiamo il nome del nodo che ospita il Rank 0.
        if "MASTER_ADDR" not in os.environ:
            # Se siamo su un nodo di login, hostname reale può dare problemi di routing
            # Forziamo 127.0.0.1 che è l'indirizzo più sicuro per processi sullo stesso nodo
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        
        # 🎯 2. MASTER_PORT: Usiamo l'ID del Job SLURM come "seme" per la porta.
        # In questo modo tutti i task dello STESSO JOB avranno la STESSA PORTA,
        # ma job diversi avranno porte diverse. Geniale e sicuro.
        if "MASTER_PORT" not in os.environ:
            job_id = os.environ.get("SLURM_JOB_ID", "29500")
            # Prendiamo le ultime 4 cifre del job_id e sommiamo a 20000
            base_port = 20000 + (int(job_id) % 10000)
            os.environ["MASTER_PORT"] = str(base_port)
            
        return rank, world_size
    else:
        # Locale rimane invariato
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = str(np.random.randint(29500, 29999))

def setup_distributed_environment(rank, world_size, slurm=True):
    import datetime, time
    # 🎯 PUNTO DI INCONTRO FISICO: Un file su Scratch che tutti possono leggere/scrivere
    # Rimuovi questo file all'inizio del job se esiste# 🎯 Usa il percorso mappato su Scratch che è scrivibile
    sync_file = "/tmp_data/torch_sync_file" 
    if rank == 0 and os.path.exists(sync_file):
        os.remove(sync_file)

    init_method = f"file://{sync_file}"
    
    print(f"\n[RANK {rank}] >>> TENTATIVO RENDEZVOUS FILE-BASED (WS={world_size})", flush=True)
    print(f"[RANK {rank}] Sync File: {sync_file}", flush=True)

    try:
        # Usiamo GLOO per la massima compatibilità di rete nei test
        dist.init_process_group(
            backend="gloo", 
            init_method=init_method,
            rank=rank, 
            world_size=world_size,
            timeout=datetime.timedelta(seconds=60)
        )
        device = torch.device(f'cuda:{rank}') if torch.cuda.is_available() else torch.device('cpu')
        print(f"[RANK {rank}] ✅ GRUPPO SINCRONIZZATO!", flush=True)
        return device
    except Exception as e:
        print(f"[RANK {rank}] ❌ ERRORE: {e}", flush=True)
        raise e

def cleanup_distributed_environment(rank):
    """Cleanup con gestione errori per evitare crash a catena."""
    try:
        # Riduciamo il timeout del barrier o mettiamolo in un try
        dist.destroy_process_group()
    except Exception as e:
        logging.warning(f"Rank {rank}: Errore durante il cleanup dist: {e}")
    logging.info(f"Processo {rank} ha terminato il suo lavoro.")
