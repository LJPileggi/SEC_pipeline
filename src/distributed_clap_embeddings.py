import os
import sys
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import numpy as np
import soundfile as sf
import librosa
from tqdm import tqdm
import logging
import random
import pydub

from .models import CLAP_initializer
from .utils import get_config_from_yaml, gen_log, read_log, delete_log, extract_all_files_from_dir, spectrogram_n_octaveband_generator
from .utils_directories import *


# TODO: change log file name and path to allow for multiple loggings relative to different configurations (n octave bands,
#    audio formats) to exist; save them in appropriate directory

### Saving functions ###
# TODO: add support for other audio files like flac etc.

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
    elif audio_format == "mp3":
        audio_segment = pydub.AudioSegment(
            data.astype("float32").tobytes(),
            frame_rate=sr,
            sample_width=4,
            channels=1
        )
        audio_segment.export(path, format="mp3", bitrate="128k")
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
# TODO: add support for other audio files like flac etc.

def process_class_with_cut_secs(clap_model, config, cut_secs, n_octave, device, rank, start_log_data, class_to_process):
    """
    Main job to submit to GPU workers to generate CLAP embeddings for a given class and
    cut_secs value. Each track gets split into multiple segments of length cut_secs * sr
    and applied random noise for data augmentation. If the required number of embeddings
    per set is not met, mutliple runs of the class tracks are performed, varying offset
    start for segments and noise. Spectrograms for the audio segments are generated as well.

    args:
     - clap_model: instance of CLAP model used for embedding generation;
     - config (dict): dictionary containing configuration subdictionaries for 'dirs',
       'audio', 'spectrogram',  'log';
     - cut_secs (int): duration of audio segments to be generated, in seconds;
     - n_octave (int): number of octave bands for the spectrogram;
     - device (torch.device): PyTorch device (e.g., 'cuda:0') to use for computation;
     - rank (int): rank of the current distributed process. Used for logging and progress bars;
     - start_log_data (dict): dictionary containing log data to resume processing from a specific checkpoint;
     - class_to_process (str): the name of the audio class to be processed.
    """
    root_source = config['dirs']['root_source']
    root_target = config['dirs']['root_target']
    audio_format = config['audio']['audio_format']
    sr = config['spectrogram']['sr']
    ref = config['spectrogram']['ref']
    center_freqs = config['spectrogram']['center_freqs']
    noise_perc = config['audio']['noise_perc']
    seed = config['audio']['seed']
    save_log_every = config['log']['save_log_every']
    
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
        
    target_class_dir = os.path.join(root_target, class_to_process)
    dir_trvlts_paths = [os.path.join(target_class_dir, p) for p in division_names]
    
    for path in [target_class_dir] + dir_trvlts_paths:
        os.makedirs(path, exist_ok=True)
        
    source_class_dir = os.path.join(root_source, class_to_process)
    audio_fp_list = extract_all_files_from_dir(source_class_dir, extension=f'.{audio_format}')
    
    if len(audio_fp_list) > 0:
        perms = np.random.RandomState(seed=seed).permutation(len(audio_fp_list))
        
        while not finish_class:
            round_ += 1
            for p in tqdm(perms, desc=f'GPU {rank} Processing {class_to_process} ({round_})', disable=(rank != 0)):
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
                                continue

                            abs_cutdata = np.abs(cut_data)
                            max_threshold = np.mean(abs_cutdata)
                            noise = (np.random.rand(*cut_data.shape) * 2 - 1) * max_threshold
                            new_audio = (1 - noise_perc) * cut_data + noise_perc * noise
                            
                            save_audio_segment(new_audio, sr, trg_audio_path, audio_format)
                            spec3o = spectrogram_n_octaveband_generator(new_audio, sr, integration_seconds=0.1, n_octave=n_octave)
                            save_spectrogram(spec3o, trg_spec3o_path)

                            preprocessed_audio = clap_model.preprocess_audio([trg_audio_path], True)
                            preprocessed_audio = preprocessed_audio.reshape(preprocessed_audio.shape[0], preprocessed_audio.shape[2])
                            x = preprocessed_audio.to(device)
                            with torch.no_grad():
                                embedding = audio_embedding(x)[0][0]
                            
                            save_embedding(embedding, trg_pt_path)
                            # os.remove(trg_audio_path)
                            
                            results += 1

                            if rank == 0 and results % save_log_every == 0:
                                # Il log è gestito solo dal rank 0
                                classes_list = sorted([d for d in os.listdir(root_source) if os.path.isdir(os.path.join(root_source, d))])
                                ic = classes_list.index(class_to_process)
                                gen_log(cut_secs, ic, di, results, round_, finish_class, config['data']['divisions_xc_sizes_names'], noise_perc, seed)
                                logging.info(f"Log salvato. Stato attuale per classe {class_to_process}: results={results}")

                        except Exception as e:
                            logging.error(f"Errore durante l'elaborazione del bucket {b} da {filepath}: {e}. Tentativo di pulizia.")
                            for file_path in [trg_audio_path, trg_pt_path, trg_spec3o_path]:
                                if file_path and os.path.exists(file_path):
                                    os.remove(file_path)
                            continue

                    if finish_class:
                        break
                if finish_class:
                    break
        
        logging.info(f"Classe '{class_to_process}' elaborata. Creazioni totali: {results}")


def worker_process(audio_format, n_octave, rank, world_size, task_queue, start_log_data, test=False):
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
     - rank (int): unique rank (ID) of current process;
     - world_size (int): total number of processes in the distributed group;
     - task_queue (mp.Queue): shared queue containing tuples of (cut_secs, class_name) to be processed;
     - start_log_data (dict): dictionary containing log data to resume processing from a specific checkpoint;
     - test (bool): whether to execute process for dummy testing dataset; defaul to False.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    clap_model, _, _ = CLAP_initializer()

    config = {
            'dirs' : {},
            'audio' : {},
            'spectrogram' : {},
            'log' : {}
        }
    config['dirs']['root_source'] = os.path.join(basedir_raw, , f'{audio_format}')
    config['dirs']['root_target'] = os.path.join(basedir_preprocessed if not test else basedir_preprocessed_test,
                                                                f'{audio_format}', f'{n_octave}_octave')
    if not os.path.exists(config['dirs']['root_target']):
        os.makedir(config['dirs']['root_target'])
    config['audio']['audio_format'] = audio_format
    config['spectrogram']['sr'] = sr
    config['spectrogram']['ref'] = ref
    config['spectrogram']['center_freqs'] = center_freqs
    config['audio']['noise_perc'] = noise_perc
    config['audio']['seed'] = seed
    config['log']['save_log_every'] = save_log_every

    config['audio']['n_octave'] = n_octave
    audio_embedding = clap_model.clap.audio_encoder
    audio_embedding.to(device)
    
    logging.info(f"Processo {rank} avviato su GPU {rank}.")

    while not task_queue.empty():
        try:
            cut_secs, class_name = task_queue.get(timeout=1)
            logging.info(f"Processo {rank} sta elaborando: cut_secs={cut_secs}, classe={class_name}")
            process_class_with_cut_secs(clap_model, config, cut_secs, n_octave, device, rank, start_log_data, class_name)
        except mp.queues.Empty:
            logging.info(f"Processo {rank} ha terminato, coda vuota.")
            break
        except Exception as e:
            logging.error(f"Errore critico nel processo {rank}: {e}")
            sys.exit(1)

    dist.destroy_process_group()


def setup_and_run(config_file, audio_format, n_octave, world_size, test=False):
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
     - world_size (int): total number of GPU devices to use for parallel processing;
     - test (bool): whether to execute pipeline for dummy testing dataset; default to False.
    """
    mp.set_start_method('spawn', force=True)

    log_data = {}
    try:
        log_data = read_log()
        logging.info(f"Ripresa da log: {log_data}")
    except FileNotFoundError:
        logging.info("Nessun log trovato, avvio una nuova esecuzione.")

    get_config_from_yaml(config_file)
    cut_secs_list = valid_cut_secs
    basedir_raw_format = os.path.join(basedir_raw, f'{audio_format}')
    classes_list = sorted([d for d in os.listdir(basedir_raw_format) if os.path.isdir(os.path.join(basedir_raw_format, d))])

    # Popola la coda dei task
    task_queue = mp.Queue()
    for cut_secs in cut_secs_list:
        for class_name in classes_list:
            task_queue.put((cut_secs, class_name))
            
    # Avvia i processi worker
    processes = []
    for rank in range(world_size):
        p = mp.Process(target=worker_process, args=(audio_format, n_octave, rank, world_size, task_queue, log_data, test))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
    
    delete_log()
    logging.info("Tutti i processi hanno terminato il loro lavoro.")

