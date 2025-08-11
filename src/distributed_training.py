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

from models import CLAP_initializer
from utils import get_config_from_yaml, gen_log, read_log, delete_log, extract_all_files_from_dir, spectrogram_n_octaveband_generator


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


def split_audio_tracks(clap_model, config, cut_secs, **kwargs):
    """
    Funzione principale per il preprocessing e la generazione di embedding,
    adattata per il calcolo distribuito.
    """
    logging.info(f"Avvio split_audio_tracks per {cut_secs} secondi, su GPU {rank}")

    root_source = config['dirs']['root_source']
    root_target = config['dirs']['root_target']
    audio_format = config['audio']['audio_format']
    window_size = int(cut_secs * config['audio']['sr'])
    noise_perc = config['audio']['noise_perc']
    seed = config['audio']['seed']
    save_log_every = config['log']['save_log_every']
    
    division_names = ["train". "es", "valid", "test"]
    target_counts = config['data']['target_counts']
    target_counts_dict = {name: count for name, count in zip(division_names, target_counts)}

    n_octave = kwargs["n_octave"]

    # kwargs relative a parallelismo
    device = kwargs["device"]
    rank = kwargs["rank"]
    world_size = kwargs["world_size"]

    # Recupero delle variabili dal log
    ic = kwargs["start_ic"]
    di = kwargs["start_di"]
    round_ = kwargs["start_round"]
    results = kwargs["start_results"]
    finish_class = kwargs["start_finish_class"]
    
    # ---------------------------------------------------------------------
    # 📍 USO DELLE VARIABILI RECUPERATE
    # Le variabili 'ic', 'di', 'round_', etc., vengono usate come stato iniziale
    # per il ciclo principale, permettendo di riprendere da dove ci si era interrotti.
    # ---------------------------------------------------------------------

    all_files = extract_all_files_from_dir(root_source)
    random.seed(seed)
    random.shuffle(all_files)
    
    sampler = DistributedSampler(all_files, num_replicas=world_size, rank=rank, shuffle=False)
    
    all_divisions_filled = False
    
    # Questo ciclo principale ora continua da `start_round`
    while not all_divisions_filled:
        files_to_process = [all_files[i] for i in list(sampler)]
        
        with tqdm(files_to_process, desc=f"GPU {rank} Processing (Round {round_})", disable=(rank != 0)) as pbar:
            for filepath in pbar:
                audio_fp_name = filepath.replace(root_source, "")
                
                try:
                    data, sr = librosa.load(filepath, sr=None, mono=True)
                    if len(data) < window_size:
                        continue
                    
                    n_buckets = len(data) // window_size
                    n_buckets_with_offset = n_buckets
                    offset = 0
                    
                    for b in range(int(n_buckets_with_offset)):
                        current_division_name = None
                        current_division_idx_for_this_cut = -1
                        for idx, div_name in enumerate(division_names):
                            if results < target_counts_dict[div_name]:
                                current_division_name = div_name
                                current_division_idx_for_this_cut = idx
                                break

                        if current_division_name is None:
                            all_divisions_filled = True
                            break

                        base_name = os.path.splitext(audio_fp_name)[0].replace("/", "_")
                        new_fp_base = f'{base_name}_{cut_secs}s_({b}_{round_})'
                        
                        target_division_dir = os.path.join(root_target, current_division_name)

                        os.makedirs(target_division_dir, exist_ok=True)
                        
                        trg_pt_path = os.path.join(target_division_dir, f'{new_fp_base}.pt')
                        trg_spec3o_path = os.path.join(target_division_dir, f'{new_fp_base}_spec3o.npy')
                        trg_audio_path = os.path.join(target_division_dir, f'{new_fp_base}.{audio_format}')

                        if os.path.exists(trg_pt_path) and os.path.exists(trg_spec3o_path):
                            results += 1
                            continue

                        start = b * window_size + offset
                        cut_data = data[start:start + window_size]

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
                            embedding = clap_model.clap.audio_encoder(x)[0][0]
                        
                        save_embedding(embedding, trg_pt_path)
                        
                        os.remove(trg_audio_path)
                        
                        results += 1

                        if rank == 0:
                            if results % save_log_every == 0:
                                # ---------------------------------------------------------------------
                                # 📍 PUNTO DI SALVATAGGIO 2: Salvataggio periodico del log
                                # ---------------------------------------------------------------------
                                gen_log(cut_secs, ic, di, results, round_, finish_class, list(zip(division_names, target_counts)), noise_perc, seed)
                                logging.info(f"Log salvato. Stato attuale: results={results}")

                except KeyboardInterrupt:
                    print("\nProcesso interrotto dall'utente.")
                    if rank == 0:
                        gen_log(cut_secs, ic, di, results, round_, finish_class, list(zip(division_names, target_counts)), noise_perc, seed)
                    sys.exit(0)
                except Exception as e:
                    logging.error(f"Errore durante l'elaborazione del file {filepath} (Round {round_}): {e}. Salto questo file.")
                    continue
            
            if all_divisions_filled:
                break
        
        round_ += 1
        
    logging.info(f"Tutte le divisioni per {cut_secs}s sono state riempite. Round {round_} completato.")

def get_embeddings_for_n_octaveband(clap_model, config, device, rank, world_size):
    """
    Loop più esterno che itera sui diversi valori di cut_secs.
    """
    cut_secs_list = config['data']['cut_secs']
    n_octave = config['audio']['n_octave']
    noise_perc = config['audio']['noise_perc']
    seed = config['audio']['seed']
    division_names = ["training", "validation", "test"]
    target_counts = config['data']['target_counts']

    start_cut_secs = None
    start_ic, start_di, start_round, start_results, start_finish_class = 0, 0, 0, 0, False

    # ---------------------------------------------------------------------
    # 📍 PUNTO DI RECUPERO 1: Recupero delle variabili dal log
    # Questo blocco viene eseguito solo dal rank 0
    # ---------------------------------------------------------------------
    if rank == 0:
        try:
            log_data = read_log()
            # Recupera i parametri dal log
            start_cut_secs = log_data.get("cut_secs")
            start_ic = log_data.get("ic")
            start_di = log_data.get("di")
            start_results = log_data.get("results")
            start_round = log_data.get("round")
            start_finish_class = log_data.get("finish_class")
            logging.info(f"Ripresa da log. cut_secs={start_cut_secs}, round={start_round}")
        except FileNotFoundError:
            logging.info("Nessun log trovato, avvio una nuova esecuzione.")
            
    # Sincronizza i parametri del log tra tutti i processi
    start_cut_secs_tensor = torch.tensor([start_cut_secs if start_cut_secs is not None else -1.0]).to(device)
    dist.broadcast(start_cut_secs_tensor, src=0)
    if start_cut_secs is None:
        start_cut_secs = -1.0 if start_cut_secs_tensor.item() == -1.0 else start_cut_secs_tensor.item()
    
    start_params = torch.tensor([start_ic, start_di, start_results, start_round, int(start_finish_class)], dtype=torch.float32).to(device)
    dist.broadcast(start_params, src=0)
    start_ic, start_di, start_results, start_round, start_finish_class = start_params.int().tolist()
    start_finish_class = bool(start_finish_class)
    
    start_index = 0
    if start_cut_secs is not None:
        try:
            start_index = cut_secs_list.index(start_cut_secs)
        except ValueError:
            logging.warning(f"Il cut_secs dal log ({start_cut_secs}) non si trova nella lista. Avvio da capo.")
    
    for i in range(start_index, len(cut_secs_list)):
        cut_secs = cut_secs_list[i]
        
        # Inizializza i parametri per il nuovo `cut_secs` se non stiamo riprendendo
        current_ic, current_di, current_results, current_round, current_finish_class = (0, 0, 0, 0, False)
        
        if cut_secs == start_cut_secs and i == start_index:
            current_ic = start_ic
            current_di = start_di
            current_results = start_results
            current_round = start_round
            current_finish_class = start_finish_class

        split_audio_tracks_kwargs = {
                    "n_octave"            : n_octave,
                    "device"              : device,
                    "rank"                : rank,
                    "world_size"          : world_size,
                    "start_ic"            : current_ic,
                    "start_di"            : current_di,
                    "start_round"         : current_round,
                    "start_finish_class"  : current_finish_class
            }
        split_audio_tracks(clap_model, config, cut_secs, **split_audio_tracks_kwargs)

# TODO: change log file name and path to allow for multiple loggings relative to different configurations (n octave bands,
#    audio formats) to exist; save them in appropriate directory
def setup_and_run(rank, world_size):
    """
    Funzione di avvio per ogni processo.
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    device = torch.device(f'cuda:{rank}')

    log_file_path = 'log.json'
    if rank == 0:
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler(log_file_path)])
    else:
        logging.basicConfig(level=logging.WARNING)

    logging.info(f"Processo {rank} avviato su GPU {rank}.")

    clap_model, _, _ = CLAP_initializer()
    config = get_config_from_yaml('config.yaml')

    get_embeddings_for_n_octaveband(clap_model, config, device, rank, world_size)
    
    dist.barrier()
    if rank == 0:
        logging.info("Tutti i processi hanno terminato il loro lavoro.")
        delete_log()

    dist.destroy_process_group()

# TODO: put main in appropriate main file
if __name__ == "__main__":
    world_size = 4
    mp.spawn(setup_and_run, args=(world_size,), nprocs=world_size, join=True)

