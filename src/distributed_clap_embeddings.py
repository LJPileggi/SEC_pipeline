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
import logging
import traceback
import random
import pydub

from .models import CLAP_initializer, spectrogram_n_octaveband_generator
from .utils import get_config_from_yaml, gen_log, read_log, delete_log, extract_all_files_from_dir
from .utils_directories import *


### Saving functions ###

def save_audio_segment(data, sr, path, audio_format="wav"):
    """
    Saves audio segment to be then used by CLAP to generate the embedding.
    Distinguishes between different audio formats.

    args:
     - data (np.array): audio data in the form of monophonic time series;
     - sr (float): sampling rate for the to-save audio;
     - path (str): path to export the audio in;
     - audio_format (str): final audio format of the saved audio; have to
       choose valid audio format (wav, mp3, flac etc.).
    """
    if audio_format == "wav":
        sf.write(path, data, sr, subtype='PCM_24')
    elif audio_format in ("mp3", "flac"):
        audio_segment = pydub.AudioSegment(
            data.astype("float32").tobytes(),
            frame_rate=sr,
            sample_width=4,
            channels=1
        )
        audio_segment.export(path, format=audio_format, bitrate="128k")
    else:
        raise NotImplementedError(f"Formato audio {audio_format} non implementato.")

def save_spectrogram(spectrogram, path):
    """
    Saves spectrogram in .npy file.
    """
    np.save(path, spectrogram)

def save_embedding(embedding, path):
    """
    Saves embedding in .pt file.
    """
    torch.save(embedding.cpu(), path)

### Embedding generation ###

def process_class_with_cut_secs(clap_model, audio_embedding, config, cut_secs, n_octave, \
                        device, rank, start_log_data, delete_segments, class_to_process):
    """
    Main job to submit to GPU workers to generate CLAP embeddings for a given class and
    cut_secs value. Each track gets split into multiple segments of length cut_secs * sr
    and applied random noise for data augmentation. If the required number of embeddings
    per set is not met, mutliple runs of the class tracks are performed, varying offset
    start for segments and noise. Spectrograms for the audio segments are generated as well.

    args:
     - clap_model: instance of CLAP model used for embedding generation;
     - audio_embedding: CLAP audio encoder for embedding generation;
     - config (dict): dictionary containing configuration subdictionaries for 'dirs',
       'audio', 'spectrogram',  'log';
     - cut_secs (int): duration of audio segments to be generated, in seconds;
     - n_octave (int): number of octave bands for the spectrogram;
     - device (torch.device): PyTorch device (e.g., 'cuda:0') to use for computation;
     - rank (int): rank of the current distributed process. Used for logging and progress bars;
     - start_log_data (dict): dictionary containing log data to resume processing from a specific checkpoint;
     - delete_segments: whether or not to delete audio segments created as a by-product of
       embedding generation;
     - class_to_process (str): the name of the audio class to be processed.
    """
    root_source = config['dirs']['root_source']
    root_target = config['dirs']['root_target']
    cut_secs_dir = os.path.join(root_target, f'{cut_secs}_secs')
    if not os.path.exists(root_target):
        os.makedirs(root_target)
    audio_format = config['audio']['audio_format']
    sr = config['spectrogram']['sr']
    ref = config['spectrogram']['ref']
    center_freqs = config['spectrogram']['center_freqs']
    noise_perc = config['audio']['noise_perc']
    seed = config['audio']['seed']
    save_log_every = config['log']['save_log_every']

    # if (start_log_data.get("divisions_xc_sizes_names")) and \
    #   ((start_log_data.get("divisions_xc_sizes_names")) != config['data']['divisions_xc_sizes_names']):
    #     raise ValueError("ValueError: divisions_xc_sizes_names between config and log doesn't match.")
    division_names = [d[0] for d in config['data']['divisions_xc_sizes_names']]
    target_counts_list = [d[1] for d in config['data']['divisions_xc_sizes_names']]

    di = 0
    results = 0
    round_ = 0
    finish_class = False

    # Se stiamo riprendendo da un log, carichiamo i valori salvati
    if start_log_data and start_log_data.get('cut_secs') == cut_secs and start_log_data.get('class_name') == class_to_process:
        di = start_log_data.get('di', 0)
        results = start_log_data.get('results', 0)
        round_ = start_log_data.get('round', 0)
        finish_class = start_log_data.get('finish_class', False)
        
    target_class_dir = os.path.join(cut_secs_dir, class_to_process)
    dir_trvlts_paths = [os.path.join(target_class_dir, p) for p in division_names]
    
    for path in [target_class_dir] + dir_trvlts_paths:
        os.makedirs(path, exist_ok=True)
        
    source_class_dir = os.path.join(root_source, class_to_process)
    audio_fp_list = extract_all_files_from_dir(source_class_dir, extension=f'.{audio_format}')
    
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
                    
                    offset = 0
                    if round_ > 1:
                        offset = random.randint(0, min(len(data) // 2, window_size))

                    for b in range(n_buckets):
                        # Controllo per passare alla prossima divisione
                        if results >= target_counts_list[di]:
                            di += 1
                            if di >= len(division_names):
                                finish_class = True
                                break
                            else:
                                results = 0
                        
                        trg_audio_path, trg_pt_path, trg_spec3o_path = None, None, None
                        try:
                            base_name = os.path.splitext(os.path.basename(audio_fp))[0]
                            new_fp_base = f'{base_name}_{cut_secs}s_({b}_{round_})'
                            trg_audio_path = os.path.join(dir_trvlts_paths[di], f'{new_fp_base}.{audio_format}')
                            trg_pt_path = os.path.join(dir_trvlts_paths[di], f'{new_fp_base}.pt')
                            trg_spec3o_path = os.path.join(dir_trvlts_paths[di], f'{new_fp_base}_spec3o.npy')

                            if os.path.exists(trg_pt_path) and os.path.exists(trg_spec3o_path):
                                results += 1
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
                            
                            save_audio_segment(new_audio, sr, trg_audio_path, audio_format)
                            spec3o = spectrogram_n_octaveband_generator(new_audio, sr, integration_seconds=0.1,
                                                        n_octave=n_octave, center_freqs=center_freqs, ref=ref)
                            save_spectrogram(spec3o, trg_spec3o_path)

                            preprocessed_audio = clap_model.preprocess_audio([trg_audio_path], True)
                            preprocessed_audio = preprocessed_audio.reshape(preprocessed_audio.shape[0], preprocessed_audio.shape[2])
                            x = preprocessed_audio.to(device)
                            with torch.no_grad():
                                embedding = audio_embedding(x)[0][0]
                            
                            save_embedding(embedding, trg_pt_path)
                            if delete_segments.lower() in ["y", "yes"]:
                                os.remove(trg_audio_path)
                            
                            results += 1

                            if results % save_log_every == 0:
                                # Il log è gestito solo dal rank 0
                                classes_list = sorted([d for d in os.listdir(root_source) if os.path.isdir(os.path.join(root_source, d))])
                                ic = classes_list.index(class_to_process)
                                gen_log(root_target, cut_secs, ic, di, results, round_, finish_class,
                                    config['data']['divisions_xc_sizes_names'], noise_perc, seed, rank)
                                logging.info(f"Log salvato. Stato attuale per classe {class_to_process}: results={results}")

                        except KeyboardInterrupt:
                            # Salva lo stato corrente prima di uscire
                            classes_list = sorted([d for d in os.listdir(root_source) if os.path.isdir(os.path.join(root_source, d))])
                            ic = classes_list.index(class_to_process)
                            gen_log(root_target, cut_secs, ic, di, results, round_, finish_class,
                                config['data']['divisions_xc_sizes_names'], noise_perc, seed, rank)
                            if os.path.exists(trg_audio_path):
                                os.remove(trg_audio_path)
                            sys.exit(0)

                        except Exception as e:
                            logging.error(f"Errore durante l'elaborazione del bucket {b} da {filepath}: {e}. Tentativo di pulizia.")
                            traceback.print_exc(file=sys.stderr)
                            for file_path in [trg_audio_path, trg_pt_path, trg_spec3o_path]:
                                if file_path and os.path.exists(file_path):
                                    os.remove(file_path)
                            continue

                    if finish_class:
                        break

                except Exception as e:
                    logging.error(f"Errore durante il caricamento del file {filepath}: {e}. Skippo il file.")
                    n_corrupt_files += 1
                    if n_corrupt_files >= len(perms):
                        classes_list = sorted([d for d in os.listdir(root_source) if os.path.isdir(os.path.join(root_source, d))])
                        ic = classes_list.index(class_to_process)
                        gen_log(root_target, cut_secs, ic, di, results, round_, finish_class,
                            config['data']['divisions_xc_sizes_names'], noise_perc, seed, rank)
                        sys.exit(0)
                    continue

                if finish_class:
                    break
        logging.info(f"Classe '{class_to_process}' elaborata. Creazioni totali: {results}")

### Workers ###

def worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, start_log_data,
                                              delete_segments, pbar_instance=None, test=False):
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
     - delete_segments: whether or not to delete audio segments created as a by-product of
       embedding generation;
     - pbar_instance: MultiProcessTqdm instance to implement a progress bar on rank 0;
     - test (bool): whether to execute process for dummy testing dataset; defaul to False.
    """
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

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
            process_class_with_cut_secs(clap_model, audio_embedding, config, cut_secs, n_octave
                                      device, rank, start_log_data, delete_segments, class_name)
            
            # Aggiorna la barra di avanzamento dopo aver completato un task
            if pbar_instance:
                pbar_instance.update(1)

        except Exception as e:
            logging.error(f"Errore critico nel processo {rank} per task ({cut_secs}, {class_name}): {e}")

    # Sincronizza i processi prima di distruggere il gruppo
    dist.barrier()
    dist.destroy_process_group()
    logging.info(f"Processo {rank} ha terminato il suo lavoro.")


# Funzione Worker per l'ambiente locale (richiamata da mp.Process)
def local_worker_process(audio_format, n_octave, config, rank, world_size, my_tasks, start_log_data,
                                                    delete_segments, pbar_instance=None, test=False):
    """
    Funzione worker per l'esecuzione parallela in ambiente locale.
    """
    # L'inizializzazione DDP è diversa per i processi lanciati con mp.Process
    dist.init_process_group("gloo", rank=rank, world_size=world_size) # Usiamo il backend 'gloo' per la CPU
    
    # Non usiamo torch.cuda.set_device qui, dato che siamo su CPU (o se usi una GPU locale, gestiscila manualmente)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        device = torch.device(f'cuda:{rank}')

    logging.info(f"Processo locale {rank} avviato su {device}.")

    clap_model, audio_embedding, _ = CLAP_initializer(device, use_cuda=True)

    # Dividi i task (come prima)
    my_tasks = all_tasks[rank::world_size]
    logging.info(f"Processo {rank} ha {len(my_tasks)} task da elaborare.")

    # Itera sui task assegnati
    for cut_secs, class_name in my_tasks:
        try:
            # Esegui la funzione di elaborazione degli embedding
            process_class_with_cut_secs(clap_model, audio_embedding, config, cut_secs, n_octave
                                      device, rank, start_log_data, delete_segments, class_name)
        except Exception as e:
            logging.error(f"Errore critico nel processo {rank}: {e}")

    # Sincronizza i processi prima di distruggere il gruppo
    dist.barrier()
    dist.destroy_process_group()
    logging.info(f"Processo {rank} ha terminato il suo lavoro.")


### Executions ###

def run_distributed_slurm(config_file, audio_format, n_octave, delete_segments, test=False):
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
     - delete_segments: whether or not to delete audio segments created as a by-product of
       embedding generation;
     - test (bool): whether to execute pipeline for dummy testing dataset; default to False.
    """
    # Questo è ora il punto di ingresso per OGNI processo SLURM (rank)

    # Recupera rank e world_size dalle variabili d'ambiente di SLURM
    # Assicurati che SLURM_PROCID e SLURM_NTASKS siano impostati nel tuo script .sbatch
    rank = int(os.environ.get("SLURM_PROCID", 0))
    world_size = int(os.environ.get("SLURM_NTASKS", 1))

    # Inizializza il logging una volta per processo
    embed_folder = os.path.join(basedir_preprocessed, f'{args.audio_format}', f'{args.n_octave}_octave')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                                             handlers=[logging.StreamHandler(),
                   logging.FileHandler(filename=os.path.join(embed_folder, f'log_rank_{rank}.txt'))])

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

    all_tasks = []
    for cut_secs in cut_secs_list:
        for class_name in classes_list:
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
    worker_process_slurm(audio_format, n_octave, config, rank, world_size, my_tasks, start_log_data, delete_segments, pbar, test)

    # Assicurati che il rank 0 chiuda la pbar dopo che tutti hanno finito
    if rank == 0 and pbar:
        pbar.close()

    # delete_log (se vuoi una singola pulizia finale) dovrebbe essere chiamato solo dal rank 0
    if rank == 0:
        delete_log(log_path)

    logging.info(f"Rank {rank}: tutti i processi hanno terminato il loro lavoro.")


def run_local_multiprocess(config_file, audio_format, n_octave, delete_segments, world_size, test=False):
    """
    Funzione per l'esecuzione locale in parallelo.
    """
    # Carica la configurazione e prepara la lista completa di tutti i task
    # ... (codice simile a run_distributed_slurm) ...
    # Calcola la lista completa all_tasks
    
    # Prepara l'ambiente DDP locale (un solo processo parent)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500' # Scegli una porta non in uso

    embed_folder = os.path.join(basedir_preprocessed, f'{args.audio_format}', f'{args.n_octave}_octave')
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

    all_tasks = []
    for cut_secs in cut_secs_list:
        for class_name in classes_list:
            all_tasks.append((cut_secs, class_name))

    # Avvia i processi worker (come nel tuo codice iniziale)
    processes = []
    # Crea un Manager per il logging o le comunicazioni tra processi
    manager = mp.Manager()
    log_lock = manager.Lock()
    message_queue = manager.Queue()
    
    for rank in range(world_size):
        # Passa i task, il lock e le code a ogni processo
        p = mp.Process(target=local_worker_process, args=(audio_format, n_octave, config, rank,
                world_size, my_tasks, start_log_data, delete_segments, f'pbar_id_{rank}', test))
        p.start()
        processes.append(p)
        
    # Aspetta che tutti i processi finiscano
    for p in processes:
        p.join()

