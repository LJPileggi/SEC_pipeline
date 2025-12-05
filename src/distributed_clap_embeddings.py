import os
import sys
import math
import time
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from queue import Empty
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
from tqdm_multiprocess.tqdm_multiprocess import MultiProcessTqdm 
import logging
import traceback
import random
import pydub

from .models import CLAP_initializer, spectrogram_n_octaveband_generator
from .utils import *
from .dirs_config import *


### Embedding generation ###


def process_class_with_cut_secs(clap_model, audio_embedding, class_to_process, cut_secs, n_octave, config):
    """
    Main job to submit to GPU workers to generate CLAP embeddings for a given class and
    cut_secs value.

    This function is highly optimized for performance and robustness, employing several
    key strategies:
    1.  **In-Memory Processing:** Audio files are loaded from a fast, temporary filesystem
        (like a RAM disk) to eliminate I/O bottlenecks during processing.
    2.  **HDF5 Batch Saving:** Instead of saving individual files, embeddings, spectrograms,
        and their corresponding names are buffered in memory and written in large batches
        to a single HDF5 file. This drastically reduces I/O overhead and network contention.
    3.  **Robust Error Handling:** The process uses nested `try...except` blocks to gracefully
        handle errors at both the file and individual bucket level. Any corrupted data or
        failed operations for a specific audio segment are discarded, preventing file
        corruption.
    4.  **Resumable Workflow:** A separate JSON file serves as a persistent index, allowing
        the process to quickly check for already processed items with an O(1) lookup cost.
        This enables the job to be resumed seamlessly from where it left off after
        interruptions (including system failures or manual KeyboardInterrupts),
        without duplicating work.
    5.  **Clean Shutdown:** A `try...finally` block ensures that any remaining buffered data
        and the updated index are saved to disk upon completion or interruption, guaranteeing
        no work is lost.

    args:
     - clap_model (CLAP_model): The pre-trained CLAP model instance;
     - audio_embedding (torch.nn.Module): The audio embedding module from the model;
     - class_to_process (str): The name of the audio class to process;
     - cut_secs (float): The duration in seconds of each audio segment;
     - n_octave (int): The number of octave bands for the spectrogram;
     - config (dict): The configuration dictionary containing all necessary parameters.

    returns:
      - n_embeddings_per_run: # embeddings actually created during a run (excluding the
        already existing ones);
      - True/False, according to whether the execution terminated successfully or not.
    """
    root_source = config['dirs']['root_source']
    root_target = config['dirs']['root_target']
    target_class_dir = os.path.join(root_target, f'{cut_secs}_secs', class_to_process)
    os.makedirs(target_class_dir, exist_ok=True)
    
    audio_format = config['audio']['audio_format']
    sr = config['spectrogram']['sr']
    ref = config['spectrogram']['ref']
    center_freqs = config['spectrogram']['center_freqs']
    noise_perc = config['audio']['noise_perc']
    seed = config['audio']['seed']
    device = config['device']

    # Identify a unique seed for each class
    class_seed = seed + hash(class_to_process) % 10000000
    offset_rng = np.random.default_rng(class_seed)
    noise_rng = np.random.default_rng(class_seed)

    division_names = [d[0] for d in config['data']['divisions_xc_sizes_names']]
    target_counts_list = [d[1] for d in config['data']['divisions_xc_sizes_names']]
    target_counts_list = np.cumsum(target_counts_list)

    di = 0
    results = 0
    round_ = 0
    n_embeddings_per_run = 0

    ### DEBUG ###
    log_limit_reached = False # Flag per limitare i log di creazione/skip
    rank = dist.get_rank() # Ottieni il rank per tracciare il processo
    ### FINE DEBUG ###

    audio_dataset_manager = HDF5DatasetManager(os.path.join(root_source, class_to_process,
                                                f'{class_to_process}_{audio_format}_dataset.h5'))

    # Specifica le dimensioni dei dati
    embedding_dim = 1024 # Esempio, metti la dimensione corretta del tuo embedding
    spec_shape = (2*n_octave+1, 1024) # Esempio, metti la forma corretta dello spettrogramma

    split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(os.path.join(target_class_dir,
                        f'{class_to_process}_{division_names[di]}_{audio_format}_emb.h5'), 'a')
    split_emb_dataset_manager.initialize_hdf5(embedding_dim, spec_shape, audio_format, cut_secs,
                                n_octave, sr, seed, noise_perc, division_names[di], class_to_process)

    try:
        perms_metadata = audio_dataset_manager.get_reproducible_permutation(class_seed)
        n_records = len(perms_metadata)
        while True:
            round_ += 1
            ### DEBUG ###
            if not log_limit_reached: 
                logging.info(f"[{rank}] --- INIZIO ROUND {round_} --- Risultati attuali: {results}. Target split: {target_counts_list[di]}")
            ### FINE DEBUG ###
            
            for i in range(n_records):
            # for metadata in perms_metadata:
                metadata = perms_metadata.iloc[i]
                track_idx = metadata['hdf5_index']
                ### DEBUG ###
                if not log_limit_reached:
                    logging.info(f"[{rank}] -> Elaborazione Traccia {track_idx}, Round {round_}")
                ### FINE DEBUG ###
                
                track = audio_dataset_manager[track_idx]
                window_size = int(cut_secs * sr)
                # Determina l'offset per l'elaborazione di questo file in questo round
                offset = 0
                # print(track)
                if round_ > 1 and track.shape[0] > window_size:
                    # Applica un offset casuale se non è il primo round
                    max_offset = track.shape[0] - window_size
                    if max_offset > 0:
                        offset = offset_rng.integers(0, max_offset)
                n_buckets = math.ceil((track.shape[0] - offset) / window_size)
                
                ### DEBUG ###
                if not log_limit_reached:
                    logging.info(f"[{rank}] -> Track {track_idx}, Round {round_}. Buckets: {n_buckets}, Offset: {offset}")
                ### FINE DEBUG ###
                
                for b in range(n_buckets):
                    if results >= target_counts_list[di]:
                        ### DEBUG ###
                        if not log_limit_reached: 
                            logging.info(f"[{rank}] TARGET RAGGIUNTO/SUPERATO. Transizione split, results={results}")
                        ### FINE DEBUG ###
                        
                        logging.info(f"Split '{division_names[di]}' completato con {results} elementi. Avvio flush...")
                        # Passa al prossimo split e reinizializza
                        di += 1
                        if di >= len(division_names):
                            # flush existing buffers and close hdf5 files
                            split_emb_dataset_manager.flush_buffers()
                            split_emb_dataset_manager.close()
                            audio_dataset_manager.close()
                            logging.info(f"Classe '{class_to_process}' elaborata. Creazioni totali: {results}")
                            return n_embeddings_per_run, True
                        else:
                            # flush existing buffers and initialise new split file
                            split_emb_dataset_manager.flush_buffers()
                            split_emb_dataset_manager.close()
                            split_emb_dataset_manager = HDF5EmbeddingDatasetsManager(os.path.join(target_class_dir,
                                                                                                  f'{class_to_process}_{division_names[di]}_{audio_format}_emb.h5'), 'a')
                            split_emb_dataset_manager.initialize_hdf5(embedding_dim, spec_shape, audio_format, cut_secs,
                                                                    n_octave, seed, sr, noise_perc, division_names[di], class_to_process)

                    # the primary keys for the embedding follow the following format:
                    # (class_idx)_(track hdf5 index)_(bucket number)_(round_)_(results number)
                    emb_pkey = f"{audio_dataset_manager.hf.attrs['class_idx']}_{track_idx}_{b}_{round_}_{results}"

                    # --- Check di esistenza (Usa __contains__) ---
                    if emb_pkey in split_emb_dataset_manager:
                    # OPPURE, se preferisci mantenere la sintassi originale:
                    # if split_emb_dataset_manager[emb_pkey]:
                        ### DEBUG ###
                        if not log_limit_reached:
                            logging.info(f"[{rank}] ---> SKIP: Embedding {emb_pkey} già esistente. Risultati: {results + 1}")
                        ### FINE DEBUG ###
                        results += 1
                        continue

                    start = b * window_size + offset
                    end = start + window_size
                    cut_data = track[start:end]

                    if len(cut_data) < window_size:
                        pad_length = window_size - len(cut_data)
                        cut_data = np.pad(cut_data, (0, pad_length), 'constant')

                    abs_cutdata = np.abs(cut_data)
                    max_threshold = np.mean(abs_cutdata)
                    noise = noise_rng.uniform(-max_threshold, max_threshold, cut_data.shape)
                    new_audio = (1 - noise_perc) * cut_data + noise_perc * noise

                    spec_n_o = spectrogram_n_octaveband_generator(new_audio, sr, integration_seconds=0.1,
                                                    n_octave=n_octave, center_freqs=center_freqs, ref=ref)

                    # preprocessed_audio = clap_model.preprocess_audio([new_audio])
                    # ----------------------------------------------------------------------------------------
                    # PATCH FINALE: Calcolo Spettrogramma Log-Mel con Librosa (Valori Fissi CLAP)
                    # ----------------------------------------------------------------------------------------

                    # PARAMETRI UFFICIALI DEL MODELLO CLAP
                    N_FFT = 1024
                    HOP_LENGTH = 480
                    N_MELS = 64
                    # La Frequenza di campionamento (sr) è 48000 Hz, ma devi usare il 'sr' 
                    # che hai già nel tuo loop (presumo sia 48000 o 44100 a seconda del tuo audio). 
                    # Se il tuo audio è già risampolato a 48kHz, usa quel valore.

                    # 1. Calcola lo spettrogramma Mel di potenza (in-memory)
                    S_mel = librosa.feature.melspectrogram(
                        y=new_audio, # L'audio già caricato e risampolato
                        sr=sr,       # Usa il Sample Rate corretto (dovrebbe essere 48000)
                        n_fft=N_FFT, 
                        hop_length=HOP_LENGTH, 
                        n_mels=N_MELS
                    )

                    S_db_tensor = torch.tensor(S_db, dtype=torch.float32)
                    # Devi ripristinare i due unsqueeze (Batch e Canale)
                    preprocessed_audio = S_db_tensor.unsqueeze(0).unsqueeze(0).to(device) # -> [1, 1, 64, 218]

                    # Prova a chiamare l'encoder più in profondità per saltare la pre-elaborazione automatica
                    try:
                        x = preprocessed_audio # Il tensore 4D corretto
                        embedding = audio_embedding(x)[0][0] # <--- QUI STA IL PROBLEMA
    
                    except AttributeError:
                        # Se non è possibile accedere direttamente, devi fare un'ulteriore ispezione
                        print("Tentativo di chiamata diretta fallito. Riprova con la chiamata originale.")

                    preprocessed_audio = S_db_tensor.unsqueeze(0).unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = audio_embedding(preprocessed_audio)[0][0]
                    # preprocessed_audio = S_db_tensor.unsqueeze(0).unsqueeze(0).to(device)
                    # preprocessed_audio = preprocessed_audio.reshape(preprocessed_audio.shape[0], preprocessed_audio.shape[2])
                    # x = preprocessed_audio.to(device)
                    # with torch.no_grad():
                    #     embedding = audio_embedding(x)[0][0]

                    split_emb_dataset_manager.add_to_data_buffer(embedding, spec_n_o, emb_pkey,
                                metadata['track_name'], class_to_process, metadata['subclass'])
                        
                    results += 1
                    n_embeddings_per_run += 1
                        
                    ### DEBUG ###
                    if not log_limit_reached:
                        logging.info(f"[{rank}] ---> CREATO: Embedding {emb_pkey}. Nuovo results: {results}")
                    if results >= 20 and not log_limit_reached:
                        logging.info(f"[{rank}] DEBUG LIMIT: Raggiunti 20 embeddings, i log di tracciamento interni verranno soppressi.")
                        log_limit_reached = True
                    ### FINE DEBUG ###


    except Exception as e:
        split_emb_dataset_manager.flush_buffers()
        split_emb_dataset_manager.close()
        audio_dataset_manager.close()
        logging.error(f"{traceback.format_exc()}")
        logging.info(f"{e}. Flushing existing buffers")
        logging.info(f"Classe '{class_to_process}' elaborata. Creazioni totali: {results}")
        return n_embeddings_per_run, False

### Workers ###

def worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, pbar_instance=None):
    """
    Worker process for distributed training.

    This function is executed by each process (on its dedicated GPU) in the distributed setup. It is responsible for:
    - Initializing the PyTorch distributed backend and setting up the GPU.
    - Continuously fetching tasks (defined as a combination of `cut_secs` and `class_name`) from a shared `task_queue`.
    - Calling the `process_class_with_cut_secs` function to handle the data generation for the assigned task.
    - Handling `Queue.Empty` exceptions to gracefully stop when no more tasks are available.
    - Providing clear logging messages to track the progress of each worker.

    args:
     - audio_format: format of the audio to embed from shell;
     - n_octave: octave band split for the spectrogram from shell;
     - config: incomplete config dictionary; info about audio format and n octave is added inside the function;
     - rank (int): unique rank (ID) of current process;
     - world_size (int): total number of processes in the distributed group;
     - task_queue (mp.Queue): shared queue containing tuples of (cut_secs, class_name, tracks_to_run) to be processed;
     - start_log_data (dict): dictionary containing log data to resume processing from a specific checkpoint;
     - pbar_instance: MultiProcessTqdm instance to implement a progress bar on rank 0.
    """
    device = setup_distributed_environment(rank, world_size)

    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)

    config['dirs']['root_source'] = os.path.join(basedir_raw, f'{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    if not os.path.exists(config['dirs']['root_target']):
        os.makedirs(config['dirs']['root_target'])
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    config['device'] = str(device)
    
    logging.info(f"Processo {rank} avviato su GPU {rank}.")

    # Itera sulla lista di task che competono a questo rank
    for cut_secs, class_name in my_tasks:
        # Stampa un messaggio per il task corrente (opzionale)
        if rank == 0:
            print(f"[{rank}] Elaborando {cut_secs}, classe {class_name}", flush=True)

        # Esegui la funzione di elaborazione degli embedding
        start_time = time.time()
        n_embeddings_per_run, completed = process_class_with_cut_secs(clap_model, audio_embedding, class_name, cut_secs, n_octave, config)
        process_time = time.time() - start_time
        write_log(config['dirs']['root_target'], (cut_secs, class_name), process_time, n_embeddings_per_run, rank, completed, **config)

        # Aggiorna la barra di avanzamento dopo aver completato un task
        if pbar_instance:
            pbar_instance.update(1)

    cleanup_distributed_environment(rank)


# Funzione Worker per l'ambiente locale (richiamata da mp.Process)
def local_worker_process(audio_format, n_octave, config, rank, world_size, my_tasks, pbar_instance=None):
    """
    Funzione worker per l'esecuzione parallela in ambiente locale.
    """
    device = setup_distributed_environment(rank, world_size, False)

    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)

    config['dirs']['root_source'] = os.path.join(basedir_raw, f'{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    if not os.path.exists(config['dirs']['root_target']):
        os.makedirs(config['dirs']['root_target'])
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    config['device'] = str(device)

    # Dividi i task (come prima)
    logging.info(f"Processo {rank} avviato su CPU {rank}.")

    # Itera sui task assegnati
    for cut_secs, class_name in my_tasks:
        # Esegui la funzione di elaborazione degli embedding
        start_time = time.time()
        n_embeddings_per_run, completed = process_class_with_cut_secs(clap_model, audio_embedding, class_name, cut_secs, n_octave, config)
        process_time = time.time() - start_time
        write_log(config['dirs']['root_target'], (cut_secs, class_name), process_time, n_embeddings_per_run, rank, completed, **config)
        # Aggiorna la barra di avanzamento locale (che invia il messaggio alla coda)
        if pbar_instance:
            pbar_instance.update(1)

    cleanup_distributed_environment(rank)


### Executions ###

def run_distributed_slurm(config_file, audio_format, n_octave):
    """
    Sets up and launches the distributed data generation pipeline.

    This is the main entry point for the entire distributed process. Its responsibilities include:
    - Loading the configuration from the YAML file.
    - Reading the log file to check for a previous checkpoint and resume if necessary.
    - Populating a shared `task_queue` with all the tasks (cut_secs and class combinations) that need to be processed.
    - Spawning multiple worker processes, one for each available GPU, using `mp.Process`.
    - Managing the main process's lifecycle, waiting for all worker processes to complete their tasks.
    - Cleaning up the log file upon successful completion of all tasks.

    args:
     - config_file: name of the config file from shell;
     - audio_format: format of the audio to embed from shell;
     - n_octave: octave band split for the spectrogram from shell.
    """
    # Questo è ora il punto di ingresso per OGNI processo SLURM (rank)

    # Recupera rank e world_size dalle variabili d'ambiente di SLURM
    # Assicurati che SLURM_PROCID e SLURM_NTASKS siano impostati nel tuo script .sbatch
    rank, world_size = setup_environ_vars()

    # Inizializza il logging una volta per processo
    embed_folder = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    os.makedirs(embed_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                             handlers=[logging.StreamHandler(),
           logging.FileHandler(filename=os.path.join(embed_folder, 'log.txt'))])

    config = {
            'dirs' : {},
            'audio' : {},
            'spectrogram' : {},
            'log' : {},
            'data' : {}
        }

    classes_list, _, _, _, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
                                    divisions_xc_sizes_names = get_config_from_yaml(config_file)

    config['spectrogram']['sr'] = sampling_rate
    config['spectrogram']['ref'] = ref
    config['spectrogram']['center_freqs'] = center_freqs
    config['audio']['noise_perc'] = noise_perc
    config['audio']['seed'] = seed
    config['data']['divisions_xc_sizes_names'] = divisions_xc_sizes_names

    basedir_raw_format = os.path.join(basedir_raw, f'{audio_format}')

    log_data = {}
    log_path = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    try:
        with open(os.path.join(log_path, 'log.json', 'r')) as f:
            log_data = json.load(f)
            logging.info(f"Ripresa da log: {log_data}")
    except FileNotFoundError:
        logging.info("Nessun log trovato, avvio una nuova esecuzione.")

    all_tasks = []
    for cut_secs in cut_secs_list:
        for class_name in classes_list:            
            # FIX: Conversione della chiave in stringa per la lookup nel log JSON e check di esistenza.
            log_key_str = str((cut_secs, class_name))

            # L'accesso al log è ora robusto: controlla se il task non è stato eseguito
            if log_key_str not in log_data or not log_data[log_key_str]:
                all_tasks.append((cut_secs, class_name))
            else:
                # Salta i task già eseguiti
                logging.info(f"Skipping task: cut_secs={cut_secs}, class_name={class_name} (already completed)")
    
    # Dividi i task per il rank corrente
    my_tasks = all_tasks[rank::world_size]
    
    # Crea il manager e la coda di messaggi per MultiProcessTqdm solo nel rank 0
    manager = mp.Manager()
    message_queue = manager.Queue() if world_size > 1 else None

    pbar = None
    if rank == 0:
        if len(all_tasks) > 0:
            pbar = MultiProcessTqdm(message_queue, "main_pbar", desc="Progresso Totale", total=len(all_tasks))

    # Esegui la logica del worker con la fetta di task
    try:
        worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, pbar)
    except Exception as e:
        logging.error(f"Errore critico nel processo principale: {e}")
        raise e
    finally:
        join_logs(log_path)

        # Assicurati che il rank 0 chiuda la pbar dopo che tutti hanno finito
        if rank == 0 and pbar:
            pbar.close()
        logging.info(f"Rank {rank}: tutti i processi hanno terminato il loro lavoro.")


def run_local_multiprocess(config_file, audio_format, n_octave, world_size):
    """
    Funzione per l'esecuzione locale in parallelo.
    """
    # Carica la configurazione e prepara la lista completa di tutti i task
    # ... (codice simile a run_distributed_slurm) ...
    # Calcola la lista completa all_tasks
    
    # Prepara l'ambiente DDP locale (un solo processo parent)
    setup_environ_vars(False)

    embed_folder = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    os.makedirs(embed_folder, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                             handlers=[logging.StreamHandler(),
                   logging.FileHandler(filename=os.path.join(embed_folder, f'log.txt'))])

    config = {
            'dirs' : {},
            'audio' : {},
            'spectrogram' : {},
            'log' : {},
            'data' : {}
        }

    classes_list, _, _, _, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
                                    divisions_xc_sizes_names = get_config_from_yaml(config_file)

    config['spectrogram']['sr'] = sampling_rate
    config['spectrogram']['ref'] = ref
    config['spectrogram']['center_freqs'] = center_freqs
    config['audio']['noise_perc'] = noise_perc
    config['audio']['seed'] = seed
    config['data']['divisions_xc_sizes_names'] = divisions_xc_sizes_names

    basedir_raw_format = os.path.join(basedir_raw, f'{audio_format}')

    log_data = {}
    log_path = os.path.join(basedir_preprocessed, f'{audio_format}', f'{n_octave}_octave')
    try:
        with open(os.path.join(log_path, 'log.json', 'r')) as f:
            log_data = json.load(f)
            logging.info(f"Ripresa da log: {log_data}")
            if log_data:
                first_key = next(iter(log_data.keys()))
    except FileNotFoundError:
        logging.info("Nessun log trovato, avvio una nuova esecuzione.")

    all_tasks = []
    for cut_secs in cut_secs_list:
        for class_name in classes_list:
            # FIX: Conversione della chiave in stringa per la lookup nel log JSON e check di esistenza.
            log_key_str = str((cut_secs, class_name))

            # L'accesso al log è ora robusto: controlla se il task non è stato eseguito
            if log_key_str not in log_data or not log_data[log_key_str]:
                all_tasks.append((cut_secs, class_name))
            else:
                # Salta i task già eseguiti
                logging.info(f"Skipping task: cut_secs={cut_secs}, class_name={class_name} (already completed)")

    # Avvia i processi worker (come nel tuo codice iniziale)
    processes = []
    # Crea un Manager per il logging o le comunicazioni tra processi
    manager = mp.Manager()
    message_queue = manager.Queue()

    # Crea l'istanza di MultiProcessTqdm nel processo principale
    total_tasks = len(all_tasks)
    pbar = None
    if total_tasks > 0:
        pbar = MultiProcessTqdm(message_queue, f"main_pbar", desc="Progresso Totale", total=total_tasks)

    try:
        for rank in range(world_size):
            # Passa i task, il lock e le code a ogni processo
            my_tasks = all_tasks[rank::world_size]
            p = mp.Process(target=local_worker_process, args=(audio_format, n_octave, config, rank,
                                                                        world_size, my_tasks, pbar))
            p.start()
            processes.append(p)
            
        # Aspetta che tutti i processi finiscano
        for p in processes:
            p.join()

    # Non c'è bisogno di un 'except', il 'finally' gestisce sia l'errore che il successo
    except Exception as e:
        # Questo cattura le eccezioni NEL PROCESSO PADRE. 
        # (Le eccezioni nei processi figli devono essere gestite dentro local_worker_process)
        logging.error(f"Errore critico nel processo principale: {e}")
        # Rilancia l'eccezione dopo aver chiamato join_logs (opzionale, ma utile per il debug)
        raise e 
        
    finally:
        # Chiudi la barra di avanzamento dopo che tutti i processi hanno terminato/fallito
        if pbar:
            pbar.close()
            
        # 7. Finalize & join_logs
        # join_logs viene chiamato in ogni caso (successo, eccezione nel padre)
        # Il log si aggiornerà con tutti i task completati fino a quel punto
        join_logs(log_path)

