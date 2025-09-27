import os
import sys
import math
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


def process_class_with_cut_secs(clap_model, audio_embedding, config, cut_secs, n_octave, \
                                        device, rank, start_log_data, class_to_process):
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
     - config (dict): The configuration dictionary containing all necessary parameters;
     - cut_secs (float): The duration in seconds of each audio segment;
     - n_octave (int): The number of octave bands for the spectrogram;
     - device (torch.device): The device (CPU or CUDA) to run the computation on;
     - rank (int): The rank of the current process in a distributed setup;
     - start_log_data (dict): Dictionary with logging data to resume a previous job;
     - class_to_process (str): The name of the audio class to process.

    returns:
        None. The function saves the generated embeddings and spectrograms as a side effect
        to the HDF5 file on disk.
    """
    root_source = config['dirs']['root_source']
    root_target = config['dirs']['root_target']
    cut_secs_dir = os.path.join(root_target, f'{cut_secs}_secs')
    
    audio_format = config['audio']['audio_format']
    sr = config['spectrogram']['sr']
    ref = config['spectrogram']['ref']
    center_freqs = config['spectrogram']['center_freqs']
    noise_perc = config['audio']['noise_perc']
    seed = config['audio']['seed']
    save_log_every = config['log']['save_log_every']
    
    # Inizializzazione della logica di ripresa
    division_names = [d[0] for d in config['data']['divisions_xc_sizes_names']]
    target_counts_list = [d[1] for d in config['data']['divisions_xc_sizes_names']]

    di = 0
    results = 0
    round_ = 0
    finish_class = False

    if start_log_data and start_log_data.get('cut_secs') == cut_secs and start_log_data.get('class_name') == class_to_process:
        di = start_log_data.get('di', 0)
        results = start_log_data.get('results', 0)
        round_ = start_log_data.get('round', 0)
        finish_class = start_log_data.get('finish_class', False)
        
    source_class_dir = os.path.join(root_source, class_to_process)
    audio_fp_list = extract_all_files_from_dir(source_class_dir, extension=f'.{audio_format}')
    
    target_class_dir = os.path.join(cut_secs_dir, class_to_process)
    os.makedirs(target_class_dir, exist_ok=True)

    # Inizializzazione dei buffer globali e dei percorsi dei file
    embeddings_buffer = []
    spectrograms_buffer = []
    names_buffer = []
    buffer_size_limit = 100 # Regola questo valore in base alla RAM disponibile
    
    # Specifica le dimensioni dei dati
    embedding_dim = 1024 # Esempio, metti la dimensione corretta del tuo embedding
    spec_shape = (128, 1024) # Esempio, metti la forma corretta dello spettrogramma
    
    # Carica l'indice esistente e inizializza il file HDF5 per lo split corrente
    index_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}.json')
    existing_names_index = load_or_create_emb_index(index_path)
    
    output_file_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}.h5')
    initialize_hdf5(output_file_path, embedding_dim, spec_shape)

    # Questo blocco 'try...finally' garantisce il salvataggio dei dati in caso di interruzione manuale
    try:
        if len(audio_fp_list) > 0:
            perms = np.random.RandomState(seed=seed).permutation(len(audio_fp_list))
            
            while not finish_class:
                round_ += 1
                n_corrupt_files = 0
                for p in perms:
                    audio_fp = audio_fp_list[p]
                    filepath = os.path.join(source_class_dir, audio_fp)
                    
                    try:
                        data, current_sr = librosa.load(filepath, sr=sr, mono=True)
                        window_size = int(cut_secs * sr)
                        n_buckets = math.ceil(len(data) / window_size)

                        local_embeddings_buffer = []
                        local_spectrograms_buffer = []
                        local_names_buffer = []

                        for b in range(n_buckets):
                            # VEDI MODIFICA: La logica per il cambio split è ora qui
                            if results >= target_counts_list[di]:
                                logging.info(f"Split '{division_names[di]}' completato con {results} elementi. Avvio flush...")

                                # 1. TRASFERIMENTO FORZATO: Sposta i dati processati finora (che non hanno causato l'interruzione)
                                # dai buffer locali a quelli globali PRIMA del flush.
                                # L'elemento che ha fatto sforare il limite non è ancora nel buffer locale.
                                # L'elemento che HA FATTO SFORARE il limite è l'ULTIMO elaborato nel ciclo 'b'.
                                # Dato che il check è PRIMA dell'elaborazione del bucket 'b' corrente:
                                # - Se il check è TRUE, l'elemento che ha completato lo split è l'ULTIMO del bucket precedente (b-1), 
                                #   e la sua elaborazione è stata completata (si è aggiornato 'results').
                                # - **Quindi, l'elemento finale è già nei buffer locali.**
                                embeddings_buffer.extend(local_embeddings_buffer)
                                spectrograms_buffer.extend(local_spectrograms_buffer)
                                names_buffer.extend(local_names_buffer)

                                # 2. PULIZIA DEI BUFFER LOCALI DOPO IL TRASFERIMENTO
                                local_embeddings_buffer = []
                                local_spectrograms_buffer = []
                                local_names_buffer = []

                                # 3. FLUSH DEL BUFFER GLOBALE
                                if len(embeddings_buffer) > 0:
                                    append_to_hdf5(output_file_path, embeddings_buffer, spectrograms_buffer, names_buffer)
                                    save_index(existing_names_index, index_path)
                                    embeddings_buffer = []
                                    spectrograms_buffer = []
                                    names_buffer = []
                                
                                # Passa al prossimo split e reinizializza
                                di += 1
                                if di >= len(division_names):
                                    finish_class = True
                                    break
                                else:
                                    results = 0
                                    index_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}.json')
                                    existing_names_index = load_or_create_emb_index(index_path)
                                    output_file_path = os.path.join(target_class_dir, f'{class_to_process}_{division_names[di]}.h5')
                                    initialize_hdf5(output_file_path, embedding_dim, spec_shape)


                            try:
                                new_fp_base = f'{audio_fp}_{cut_secs}s_({b}_{round_})'

                                if new_fp_base in existing_names_index:
                                    continue

                                start = b * window_size + offset
                                end = start + window_size
                                cut_data = data[start:end]

                                if len(cut_data) < window_size:
                                    pad_length = window_size - len(cut_data)
                                    cut_data = np.pad(cut_data, (0, pad_length), 'constant')

                                abs_cutdata = np.abs(cut_data)
                                max_threshold = np.mean(abs_cutdata)
                                noise = (np.random.rand(*cut_data.shape) * 2 - 1) * max_threshold
                                new_audio = (1 - noise_perc) * cut_data + noise_perc * noise
                            
                                preprocessed_audio = clap_model.preprocess_audio([new_audio], is_path=False)
                                preprocessed_audio = preprocessed_audio.reshape(preprocessed_audio.shape[0], preprocessed_audio.shape[2])
                                x = preprocessed_audio.to(device)
                                with torch.no_grad():
                                    embedding = audio_embedding(x)[0][0]

                                local_embeddings_buffer.append(embedding.cpu().numpy())
                                local_spectrograms_buffer.append(spec3o)
                                local_names_buffer.append(new_fp_base)

                                existing_names_index[new_fp_base] = True

                                results += 1

                            except Exception as e:
                                logging.error(f"Errore durante l'elaborazione del bucket {b} da "
                                              f"{filepath}: {e}. Salto il resto di questo file.")
                                traceback.print_exc(file=sys.stderr)
                                continue

                    except Exception as e:
                        logging.error(f"Errore durante il caricamento del file {filepath}: {e}. Salto il file.")
                        traceback.print_exc(file=sys.stderr)
                        n_corrupt_files += 1
                        if n_corrupt_files >= len(perms):
                            logging.error(f"Tutti i {n_corrupt_files} file sono corrotti. Uscita.")
                            sys.exit(0)
                        continue

                    finally:
                        embeddings_buffer.extend(local_embeddings_buffer)
                        spectrograms_buffer.extend(local_spectrograms_buffer)
                        names_buffer.extend(local_names_buffer)

                        if finish_class:
                            break
                
                # Dopo il ciclo esterno, se il buffer globale è pieno, salvalo su disco
                if len(embeddings_buffer) >= buffer_size_limit:
                    append_to_hdf5(output_file_path, embeddings_buffer, spectrograms_buffer, names_buffer)
                    save_emb_index(existing_names_index, index_path)
                    embeddings_buffer = []
                    spectrograms_buffer = []
                    names_buffer = []

    except KeyboardInterrupt:
        logging.info("Interruzione manuale rilevata. Avvio del salvataggio finale.")
    finally:
        if len(embeddings_buffer) > 0:
            append_to_hdf5(output_file_path, embeddings_buffer, spectrograms_buffer, names_buffer)
            save_emb_index(existing_names_index, index_path)
            logging.info("Dati e indice rimanenti salvati con successo.")
        
        classes_list = sorted([d for d in os.listdir(root_source) if os.path.isdir(os.path.join(root_source, d))])
        ic = classes_list.index(class_to_process)
        gen_log(root_target, cut_secs, ic, di, results, round_, finish_class,
                config['data']['divisions_xc_sizes_names'], noise_perc, seed, rank)
        
        logging.info(f"Classe '{class_to_process}' elaborata. Creazioni totali: {sum(target_counts_list[:di]) + results}")


### Workers ###

def worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, \
                                    start_log_data, pbar_instance=None, test=False):
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
     - task_queue (mp.Queue): shared queue containing tuples of (cut_secs, class_name) to be processed;
     - start_log_data (dict): dictionary containing log data to resume processing from a specific checkpoint;
     - pbar_instance: MultiProcessTqdm instance to implement a progress bar on rank 0;
     - test (bool): whether to execute process for dummy testing dataset; defaul to False.
    """
    setup_distributed_environment(rank, world_size)

    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)

    config['dirs']['root_source'] = os.path.join(basedir_raw if not test else basedir_raw_test, f'{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed if not test else basedir_preprocessed_test,
                                                                          f'{audio_format}', f'{n_octave}_octave')
    if not os.path.exists(config['dirs']['root_target']):
        os.makedirs(config['dirs']['root_target'])
    config['audio']['audio_format'] = audio_format
    config['audio']['n_octave'] = n_octave
    
    logging.info(f"Processo {rank} avviato su GPU {rank}.")

    # Itera sulla lista di task che competono a questo rank
    for cut_secs, class_name in my_tasks:
        try:
            # Stampa un messaggio per il task corrente (opzionale)
            if rank == 0:
                print(f"[{rank}] Elaborando {cut_secs}, classe {class_name}", flush=True)

            # Esegui la funzione di elaborazione degli embedding
            process_class_with_cut_secs(clap_model, audio_embedding, config, cut_secs, n_octave,
                                                      device, rank, start_log_data, class_name)
            
            # Aggiorna la barra di avanzamento dopo aver completato un task
            if pbar_instance:
                pbar_instance.update(1)

        except Exception as e:
            logging.error(f"Errore critico nel processo {rank} per task ({cut_secs}, {class_name}): {e}")

    # Sincronizza i processi prima di distruggere il gruppo
    cleanup_distributed_environment()


# Funzione Worker per l'ambiente locale (richiamata da mp.Process)
def local_worker_process(audio_format, n_octave, config, rank, world_size, my_tasks, \
                                    start_log_data, pbar_instance=None, test=False):
    """
    Funzione worker per l'esecuzione parallela in ambiente locale.
    """
    setup_distributed_environment(rank, world_size, False)

    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)

    # Dividi i task (come prima)
    logging.info(f"Processo {rank} ha {len(my_tasks)} task da elaborare.")

    # Itera sui task assegnati
    for cut_secs, class_name in my_tasks:
        try:
            # Esegui la funzione di elaborazione degli embedding
            process_class_with_cut_secs(clap_model, audio_embedding, config, cut_secs, n_octave,
                                                      device, rank, start_log_data, class_name)
            # Aggiorna la barra di avanzamento locale (che invia il messaggio alla coda)
            if pbar_instance:
                pbar_instance.update(1)
        except Exception as e:
            logging.error(f"Errore critico nel processo {rank}: {e}")

    # Sincronizza i processi prima di distruggere il gruppo
    cleanup_distributed_environment()


### Executions ###

def run_distributed_slurm(config_file, audio_format, n_octave, test=False):
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
     - n_octave: octave band split for the spectrogram from shell;
     - test (bool): whether to execute pipeline for dummy testing dataset; default to False.
    """
    # Questo è ora il punto di ingresso per OGNI processo SLURM (rank)

    # Recupera rank e world_size dalle variabili d'ambiente di SLURM
    # Assicurati che SLURM_PROCID e SLURM_NTASKS siano impostati nel tuo script .sbatch
    rank, world_size = setup_environ_vars()

    # Inizializza il logging una volta per processo
    embed_folder = os.path.join(basedir_preprocessed if not test else basedir_preprocessed_test,
                                                        f'{audio_format}', f'{n_octave}_octave')
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

    _, _, _, save_log_every, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
                                    divisions_xc_sizes_names = get_config_from_yaml(config_file)

    config['spectrogram']['sr'] = sampling_rate
    config['spectrogram']['ref'] = ref
    config['spectrogram']['center_freqs'] = center_freqs
    config['audio']['noise_perc'] = noise_perc
    config['audio']['seed'] = seed
    config['log']['save_log_every'] = save_log_every
    config['data']['divisions_xc_sizes_names'] = divisions_xc_sizes_names

    basedir_raw_format = os.path.join(basedir_raw if not test else basedir_raw_test, f'{audio_format}')
    classes_list = sorted([d for d in os.listdir(basedir_raw_format) if os.path.isdir(os.path.join(basedir_raw_format, d))])

    log_data = {}
    log_path = os.path.join(basedir_preprocessed if not test else basedir_preprocessed_test,
                                                    f'{audio_format}', f'{n_octave}_octave')
    try:
        log_data = read_log(log_path)
        logging.info(f"Ripresa da log: {log_data}")
    except FileNotFoundError:
        logging.info("Nessun log trovato, avvio una nuova esecuzione.")

    log_cut_secs = log_data.get('cut_secs', None)
    log_class_name = log_data.get('class_name', None)

    # Questo flag indica se abbiamo raggiunto il punto di ripresa
    found_resume_point = False

    all_tasks = []
    for cut_secs in cut_secs_list:
        for class_name in classes_list:
            if log_data:
                # Se siamo già oltre il punto di ripresa, mettiamo tutto in coda
                if found_resume_point:
                    all_tasks.append((cut_secs, class_name))
                    continue

                # Confronto lessicografico per trovare il punto di ripresa
                if cut_secs > log_cut_secs:
                    found_resume_point = True
                    all_tasks.append((cut_secs, class_name))
                    continue
                elif cut_secs == log_cut_secs and class_name >= log_class_name:
                    found_resume_point = True
                    # Inserisci il task in corso
                    all_tasks.append((cut_secs, class_name))
                    continue
                else:
                    # Salta i task precedenti al punto di ripresa
                    logging.info(f"Skipping task: cut_secs={cut_secs}, class_name={class_name} (already completed)")
                    continue
            else:
                # Nessun log, aggiungi tutti i task normalmente
                all_tasks.append((cut_secs, class_name))
    
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
    worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, log_data, pbar, test)

    # Assicurati che il rank 0 chiuda la pbar dopo che tutti hanno finito
    if rank == 0 and pbar:
        pbar.close()

    # delete_log (se vuoi una singola pulizia finale) dovrebbe essere chiamato solo dal rank 0
    if rank == 0:
        delete_log(log_path)

    logging.info(f"Rank {rank}: tutti i processi hanno terminato il loro lavoro.")


def run_local_multiprocess(config_file, audio_format, n_octave, world_size, test=False):
    """
    Funzione per l'esecuzione locale in parallelo.
    """
    # Carica la configurazione e prepara la lista completa di tutti i task
    # ... (codice simile a run_distributed_slurm) ...
    # Calcola la lista completa all_tasks
    
    # Prepara l'ambiente DDP locale (un solo processo parent)
    setup_environ_vars(False)

    embed_folder = os.path.join(basedir_preprocessed if not test else basedir_preprocessed_test,
                                                        f'{audio_format}', f'{n_octave}_octave')
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

    _, _, _, save_log_every, sampling_rate, ref, noise_perc, seed, center_freqs, cut_secs_list, \
                                    divisions_xc_sizes_names = get_config_from_yaml(config_file)

    config['spectrogram']['sr'] = sampling_rate
    config['spectrogram']['ref'] = ref
    config['spectrogram']['center_freqs'] = center_freqs
    config['audio']['noise_perc'] = noise_perc
    config['audio']['seed'] = seed
    config['log']['save_log_every'] = save_log_every
    config['data']['divisions_xc_sizes_names'] = divisions_xc_sizes_names

    basedir_raw_format = os.path.join(basedir_raw if not test else basedir_raw_test, f'{audio_format}')
    classes_list = sorted([d for d in os.listdir(basedir_raw_format) if os.path.isdir(os.path.join(basedir_raw_format, d))])

    log_data = {}
    log_path = os.path.join(basedir_preprocessed if not test else basedir_preprocessed_test,
                                                    f'{audio_format}', f'{n_octave}_octave')
    try:
        log_data = read_log(log_path)
        logging.info(f"Ripresa da log: {log_data}")
    except FileNotFoundError:
        logging.info("Nessun log trovato, avvio una nuova esecuzione.")

    log_cut_secs = log_data.get('cut_secs', None)
    log_class_name = log_data.get('class_name', None)

    all_tasks = []
    for cut_secs in cut_secs_list:
        for class_name in classes_list:
            if log_data:
                # Se siamo già oltre il punto di ripresa, mettiamo tutto in coda
                if found_resume_point:
                    all_tasks.append((cut_secs, class_name))
                    continue

                # Confronto lessicografico per trovare il punto di ripresa
                if cut_secs > log_cut_secs:
                    found_resume_point = True
                    all_tasks.append((cut_secs, class_name))
                    continue
                elif cut_secs == log_cut_secs and class_name >= log_class_name:
                    found_resume_point = True
                    # Inserisci il task in corso
                    all_tasks.append((cut_secs, class_name))
                    continue
                else:
                    # Salta i task precedenti al punto di ripresa
                    logging.info(f"Skipping task: cut_secs={cut_secs}, class_name={class_name} (already completed)")
                    continue
            else:
                # Nessun log, aggiungi tutti i task normalmente
                all_tasks.append((cut_secs, class_name))

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

    for rank in range(world_size):
        # Passa i task, il lock e le code a ogni processo
        my_tasks = all_tasks[rank::world_size]
        p = mp.Process(target=local_worker_process, args=(audio_format, n_octave, config, rank,
                                                    world_size, my_tasks, log_data, pbar, test))
        p.start()
        processes.append(p)
        
    # Aspetta che tutti i processi finiscano
    for p in processes:
        p.join()

    # Chiudi la barra di avanzamento dopo che tutti i processi hanno terminato
    if pbar:
        pbar.close()

