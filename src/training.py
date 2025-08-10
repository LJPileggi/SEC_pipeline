import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import pydub
import librosa
import soundfile as sf
from tqdm import tqdm
import random
import sys

# from .utils import patience, epochs, batch_size, device, save_log_every, \
#                         sampling_rate, ref, center_freqs, valid_cut_secs
from .utils import get_config_from_yaml
from .utils import basedir_preprocessed # basedir, basedir_raw, results_validation_filepath_project
from .utils import extract_all_files_from_dir, gen_log, read_log, delete_log
from .models import CLAP_initializer, spectrogram_n_octaveband_generator
from .losses import *


### Embedding generation ###
# TODO: configure embedding generation for multi-GPU run
# TODO: add support for other audio files like flac etc.

def split_audio_tracks(
    audio_format: str,
    sampling_rate: str,
    center_freqs: np.array,
    ref: float,
    root_source: str,
    root_target: str,
    clap_model: CLAP,
    cut_secs: int = 3,
    divisions_xc_sizes_names: list[tuple[str, int]] = [('train', 500), ('es', 100), ('valid', 100), ('test', 100)],
    noise_perc: float = 0.3,
    seed: int = 0,
    initial_counts: dict = None, # Conteggi iniziali da log
    initial_round: int = 0 # Round iniziale da log
):
    """
    Divide le tracce audio in segmenti di dimensione fissa, genera embedding,
    e li distribuisce in divisioni specificate (train, es, valid, test)
    con conteggi esatti, supportando la ripresa da un log.

    Args:
        audio_format (str): Formato dell'audio da analizzare.
        sampling_rate: sampling rate da config file.
        center_freqs: frequenze centrali delle bande d'ottava.
        ref: valore di ref da config file.
        root_source (str): Percorso della directory radice contenente le classi audio.
        root_target (str): Percorso della directory dove verranno salvati audio ed embedding elaborati.
        clap_model (CLAP): Un'istanza del modello CLAP per l'elaborazione audio.
        cut_secs (int): Durata in secondi per ogni segmento audio.
        divisions_xc_sizes_names (list[tuple[str, int]]): Una lista di tuple,
            dove ogni tupla contiene (nome_divisione, conteggio_target).
            Esempio: [('train', 500), ('es', 100), ('valid', 100), ('test', 100)]
        noise_perc (float): Percentuale di rumore da aggiungere ai segmenti audio.
        seed (int): Seed per la generazione di numeri casuali per la riproducibilità.
        initial_counts (dict, optional): Dizionario dei conteggi attuali per ciascuna divisione
            se si riprende da un log. Default a None.
        initial_round (int, optional): Il numero di round da cui riprendere. Default a 0.
    """
    # Estrai i nomi delle divisioni e i loro conteggi target
    division_names, _ = zip(*divisions_xc_sizes_names)
    target_counts_dict = dict(divisions_xc_sizes_names)

    # Inizializza i conteggi attuali per ciascuna divisione, riprendendo dal log se fornito
    current_counts = initial_counts if initial_counts is not None else {name: 0 for name in division_names}

    # Sposta l'encoder audio su CPU per liberare memoria GPU se non attivamente utilizzata
    audio_embedding = clap_model.clap.audio_encoder
    audio_embedding.cpu()

    # Inizializza lo stato casuale per la riproducibilità
    np.random.RandomState(seed=seed)
    random.seed(seed)

    # Crea la directory target radice se non esiste
    if not os.path.exists(root_target):
        os.makedirs(root_target)

    # Determina se tutte le divisioni sono già state riempite dai conteggi iniziali
    all_divisions_filled = all(current_counts[name] >= target_counts_dict[name] for name in division_names)

    # Se tutte le divisioni erano già piene nel log, abbiamo finito.
    if all_divisions_filled:
        print("Tutte le divisioni erano già state riempite nella precedente esecuzione. Uscita.")
        audio_embedding.to(device)
        delete_log()
        return current_counts

    # Ottieni la lista di tutte le classi (sottodirectory) nella sorgente
    all_classes_in_source = os.listdir(root_source)

    round_ = initial_round # Inizializza il round dal log

    # Ciclo esterno per continuare a generare tagli finché tutte le divisioni non sono piene
    while not all_divisions_filled:
        round_ += 1
        print(f"\n--- Avvio del Round {round_} per la generazione degli embedding ---")

        # Mescola le classi per ogni round per ottenere nuove combinazioni/ordine
        random.shuffle(all_classes_in_source)

        # Itera attraverso ogni classe (sottodirectory) nella sorgente
        for class_name in tqdm(all_classes_in_source, desc=f'Elaborazione Classi (Round {round_})'):
            if all_divisions_filled:
                break # Tutti i target globali raggiunti, interrompi l'elaborazione delle classi

            source_class_dir = os.path.join(root_source, class_name)
            if not os.path.isdir(source_class_dir):
                continue # Salta se non è una directory

            target_class_dir = os.path.join(root_target, class_name)
            if not os.path.exists(target_class_dir):
                os.makedirs(target_class_dir)

            # Crea sottodirectory per ogni divisione all'interno della directory della classe corrente
            dir_trvlts_paths_for_class = []
            for div_name in division_names:
                div_path = os.path.join(target_class_dir, div_name)
                if not os.path.exists(div_path):
                    os.makedirs(div_path)
                dir_trvlts_paths_for_class.append(div_path)

            # Ottieni tutti i file audio per la classe corrente
            audio_fp_list = extract_all_files_from_dir(source_class_dir, extension=audio_format) # O '.wav'
            if not audio_fp_list:
                # print(f"Nessun file audio trovato in {source_class_dir}. Salto.")
                continue

            # Mescola i file audio all'interno della classe per randomizzare la selezione
            random.shuffle(audio_fp_list)

            # Itera attraverso i file audio per generare tagli ed embedding
            for audio_fp_name in tqdm(audio_fp_list, desc=f'File in {class_name} (Round {round_})', leave=False):
                if all_divisions_filled:
                    break # Tutti i target globali raggiunti, interrompi l'elaborazione dei file in questa classe

                # Trova la divisione corrente da riempire prima di ogni tentativo di taglio
                current_division_name = None
                current_division_idx_for_this_cut = -1
                for idx, div_name in enumerate(division_names):
                    if current_counts[div_name] < target_counts_dict[div_name]:
                        current_division_name = div_name
                        current_division_idx_for_this_cut = idx
                        break

                if current_division_name is None: # Tutte le divisioni piene
                    all_divisions_filled = True
                    break # Interrompi il ciclo dei bucket

                filepath = os.path.join(source_class_dir, audio_fp_name)
                try:
                    data, sr = librosa.load(filepath, sr=sampling_rate) # Carica con SR originale
                    target_sr = 52100
                    if sr != target_sr:
                        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                    sr = target_sr
                except Exception as e:
                    print(f'Salta file: {filepath} a causa di errore: {e}')
                    continue

                data = librosa.to_mono(data)
                window_size = round(cut_secs * sr)

                # Determina l'offset per l'elaborazione di questo file in questo round
                offset = 0
                if round_ > 1 and data.shape[0] > window_size:
                    # Applica un offset casuale se non è il primo round
                    max_offset = data.shape[0] - window_size
                    if max_offset > 0:
                        offset = random.randint(0, max_offset)

                # Ricalcola n_buckets in base all'offset corrente
                n_buckets_with_offset = math.floor((data.shape[0] - offset) / window_size)

                if n_buckets_with_offset == 0:
                    # print(f"Avviso: File audio {filepath} (Round {round_}) troppo corto per tagli di {cut_secs}s con offset. Salto.")
                    continue

                # Genera tagli da questo file audio per il round corrente
                for b in range(int(n_buckets_with_offset)):
                    # Trova la divisione corrente da riempire prima di ogni nuovo tentativo di taglio
                    current_division_name = None
                    current_division_idx_for_this_cut = -1
                    for idx, div_name in enumerate(division_names):
                        if current_counts[div_name] < target_counts_dict[div_name]:
                            current_division_name = div_name
                            current_division_idx_for_this_cut = idx
                            break

                    if current_division_name is None: # Tutte le divisioni piene
                        all_divisions_filled = True
                        break # Interrompi il ciclo dei bucket

                    # Costruisci i nuovi percorsi dei file, includendo round_ nel nome per garantire l'unicità
                    base_name = os.path.splitext(audio_fp_name)[0].replace("/", "_")
                    new_fp_base = f'{base_name}_{cut_secs}s_({b}_{round_})' # Aggiunto round_ al nome del file
                    trg_audio_path = os.path.join(dir_trvlts_paths_for_class[current_division_idx_for_this_cut],
                                                                                    f'{new_fp_base}.{audio_format}')
                    trg_pt_path = os.path.join(dir_trvlts_paths_for_class[current_division_idx_for_this_cut],
                                                                                    f'{new_fp_base}.pt')
                    trg_spec3o_path = os.path.join(dir_trvlts_paths_for_class[current_division_idx_for_this_cut],
                                                                                    f'{new_fp_base}_spec3o.npy')

                    # Controlla se l'embedding e lo spettrogramma esistono già
                    if os.path.exists(trg_pt_path) and os.path.exists(trg_spec3o_path):
                        current_counts[current_division_name] += 1
                        continue # Salta la generazione, conta l'esistente e prosegui

                    try:
                        start = b * window_size + offset # Applica l'offset qui
                        cut_data = data[start:start + window_size]

                        # Aggiungi rumore
                        abs_cutdata = np.abs(cut_data)
                        max_threshold = np.mean(abs_cutdata)
                        noise = (np.random.rand(*cut_data.shape) * 2 - 1) * max_threshold
                        new_audio = (1 - noise_perc) * cut_data + noise_perc * noise

                        # Esporta il segmento audio temporaneamente
                        if audio_format == "mp3":
                            audio_segment = pydub.AudioSegment(
                                new_audio.astype("float32").tobytes(),
                                frame_rate=sr,
                                sample_width=4, # float32 è 4 byte
                                channels=1
                            )
                            audio_segment.export(trg_audio_path, format="mp3", bitrate="128k")
                        elif audio_format == "wav":
                            sf.write(trg_path, new_audio, sr, subtype='PCM_24')
                        elif audio_format == "flac":
                            raise NotImplementedError(f"NotImplementedError: {audio_format} audio format "
                                                      f"not yet implemented.")
                        else:
                            raise NotImplementedError(f"NotImplementedError: {audio_format} audio format "
                                                      f"not yet implemented.")

                        # Genera lo spettrogramma
                        spec3o = spectrogram_n_octaveband_generator(new_audio, sr, integration_seconds=0.1,
                                                                        center_freqs=center_freqs, ref=ref)
                        np.save(trg_spec3o_path, spec3o)

                        # Preprocessing audio per CLAP e generazione embedding
                        preprocessed_audio = clap_model.preprocess_audio([trg_audio_path], True)
                        preprocessed_audio = preprocessed_audio.reshape(preprocessed_audio.shape[0], preprocessed_audio.shape[2])
                        x = preprocessed_audio
                        with torch.no_grad():
                            h = audio_embedding(x)[0][0] # Assumendo che restituisca un tensore
                        torch.save(h.cpu(), trg_pt_path)

                        # Pulisci il file audio temporaneo
                        os.remove(trg_audio_path)

                        current_counts[current_division_name] += 1

                        # Logga lo stato periodicamente
                        if sum(current_counts.values()) % save_log_every == 0:
                            gen_log(cut_secs, current_counts, round_, divisions_xc_sizes_names, noise_perc, seed)

                    except KeyboardInterrupt:
                        print("\nProcesso interrotto dall'utente. Pulizia ed uscita.")
                        # Salva lo stato corrente prima di uscire
                        gen_log(cut_secs, current_counts, round_, divisions_xc_sizes_names, noise_perc, seed)
                        if os.path.exists(trg_audio_path):
                            os.remove(trg_audio_path)
                        sys.exit(0)
                    except Exception as e:
                        print(f"Errore durante l'elaborazione del taglio {b} da {filepath} (Round {round_}): {e}. Salto questo taglio.")
                        # Pulisci eventuali file parzialmente creati
                        if os.path.exists(trg_audio_path):
                            os.remove(trg_audio_path)
                        if os.path.exists(trg_pt_path):
                            os.remove(trg_pt_path)
                        if os.path.exists(trg_spec3o_path):
                            os.remove(trg_spec3o_path)
                        continue # Prova il prossimo taglio o il prossimo file

                if all_divisions_filled:
                    break # Interrompi il ciclo dei file audio se tutte le divisioni sono piene

            if all_divisions_filled:
                break # Interrompi il ciclo delle classi se tutte le divisioni sono piene

        # Dopo un round completo di elaborazione di tutte le classi, controlla se tutte le divisioni sono piene.
        if not all_divisions_filled:
            print(f"Round {round_} completato. Non tutte le divisioni sono piene. Conteggi attuali: {current_counts}")
            # Logga lo stato dopo ogni round completo se non è finito.
            gen_log(cut_secs, current_counts, round_, divisions_xc_sizes_names, noise_perc, seed)

    # Log finale quando tutte le divisioni sono piene
    audio_embedding.to(device)
    delete_log() # Assumendo che questa funzione gestisca la pulizia del log globale

    print("\n--- Conteggi Finali degli Embedding ---")
    for name, count in current_counts.items():
        target = target_counts_dict.get(name, "N/A")
        print(f"  {name}: {count} / {target}")
    print("--------------------------------------")

    return current_counts

def get_embeddings_for_n_octaveband(basedir_raw, n_octave_dir, audio_format, sampling_rate, center_freqs, ref):
    clap_model, _, _ = CLAP_initializer(device)

    # Definisci le dimensioni delle divisioni desiderate
    my_divisions = [('train', 500), ('val', 100), ('es', 100), ('test', 100)]

    # Il ciclo di esecuzione principale dal tuo codice originale, adattato
    try:
        pars_split = read_log()
        print("Riprendendo da log precedente.")
        # Inizializza current_counts e round_ dai dati del log
        initial_counts_from_log = pars_split.get("current_counts", {name: 0 for name, _ in my_divisions})
        initial_round_from_log = pars_split.get("round", 0)
        # Assicurati che divisions_xc_sizes_names sia coerente
        logged_divisions = pars_split.get("divisions_xc_sizes_names", my_divisions)
        if logged_divisions != my_divisions:
            print("Avviso: Le divisioni nel log non corrispondono a quelle attuali. Potrebbe causare un comportamento inaspettato.")
            # Potresti voler gestire questo caso, ad esempio, azzerando i conteggi o chiedendo all'utente.
            # Per ora, useremo le divisioni dal log per la ripresa.
            my_divisions = logged_divisions # Usa le divisioni dal log per la ripresa
    except FileNotFoundError:
        print("Nessun log precedente trovato. Avvio da zero.")
        initial_counts_from_log = {name: 0 for name, _ in my_divisions}
        initial_round_from_log = 0

    # Questo ciclo itera attraverso diversi valori di cut_secs.
    # La logica del conteggio esatto si applicherà per ogni esecuzione di `cut_secs`.
    # Assicurati che `basedir_raw` e `clap_model` siano correttamente inizializzati prima di questo ciclo.

    # Esempio di configurazione fittizia per la dimostrazione:
    # basedir_raw = './dummy_raw_audio'
    # n_octave_dir = './dummy_preprocessed_embeddings'
    # os.makedirs(os.path.join(basedir_raw, 'class_1'), exist_ok=True)
    # os.makedirs(os.path.join(basedir_raw, 'class_2'), exist_ok=True)
    # # Crea alcuni file audio fittizi (file vuoti, solo per la simulazione del percorso)
    # for i in range(10):
    #     with open(os.path.join(basedir_raw, 'class_1', f'audio_{i}.mp3'), 'w') as f: f.write('dummy')
    #     with open(os.path.join(basedir_raw, 'class_2', f'audio_{i+10}.mp3'), 'w') as f: f.write('dummy')
    # clap_model = CLAP()

    # Determina da quale `cut_secs` iniziare
    start_cut_secs = pars_split.get("cut_secs", 1) if 'pars_split' in locals() else 1

    for cut_secs in range(start_cut_secs, 31): # Esegui per i valori di cut_secs desiderati
        if cut_secs in valid_cut_secs:
            print(f'\nAVVIO TAGLIO SUONI CON SECONDI {cut_secs}')
            # new_dir composta secondo formato audio e n octave
            new_dir = os.path.join(basedir_preprocessed, audio_format, n_octave_dir, f'{cut_secs}_secs')
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)

            # Se si riprende per lo stesso cut_secs, usa i parametri loggati
            if cut_secs == start_cut_secs and 'pars_split' in locals():
                split_audio_tracks(
                    audio_format=audio_format,
                    sampling_rate=sampling_rate,
                    center_freqs=center_freqs,
                    ref=ref,
                    root_source=basedir_raw,
                    root_target=new_dir,
                    clap_model=clap_model,
                    cut_secs=cut_secs,
                    divisions_xc_sizes_names=logged_divisions, # Usa le divisioni dal log
                    noise_perc=pars_split.get("noise_perc", 0.3),
                    seed=pars_split.get("seed", 0),
                    initial_counts=initial_counts_from_log,
                    initial_round=initial_round_from_log
                )
            else:
                # Per i nuovi valori di cut_secs, avvia da zero
                split_audio_tracks(
                    audio_format=audio_format,
                    sampling_rate=sampling_rate,
                    center_freqs=center_freqs,
                    ref=ref,
                    root_source=basedir_raw,
                    root_target=new_dir,
                    clap_model=clap_model,
                    cut_secs=cut_secs,
                    divisions_xc_sizes_names=my_divisions, # Usa le divisioni predefinite per le nuove esecuzioni
                    noise_perc=0.3, # Percentuale di rumore predefinita
                    seed=0, # Seed predefinito
                    initial_counts={name: 0 for name, _ in my_divisions},
                    initial_round=0
                )
            print(f'FINITO TAGLIO SUONI CON SECONDI {cut_secs}')


### Finetuned classifier training ###

def build_model():
    return FinetunedModel(classes, device=device)

def train(tr_set, es_set, config, epochs, callback=None):
    """
    Training container function to train finetuned classifier
    with the different loss functions.

    args:
     - tr_set: container for training set's x-s and y-s;
     - es_set: container for validation set's x-s and y-s;
     - config: optimiser configuration;
     - epochs: training epochs (only if optimiser != RR);
     - callback (optional): training callback.

    returns:
     - model: trained torch model for finetuned classifier.
    """
    model = build_model()
    optimizer, with_epochs = build_optimizer(config['optimizer'], model)
    with_epochs = config['optimizer']['builder'] != 'RR'
    if not with_epochs:
        epochs = 1
    best_es_accuracy = None
    best_params = model.state_dict()
    counter_es = 0
    for epoch in range(epochs):
        model.train()
        for x, y in tqdm(tr_set, desc=f'{epoch + 1}/{epochs} training'):
            if with_epochs:
                h = model(x)
                loss = criterion(h, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            else:
                optimizer(x, y)
        if not with_epochs:
            optimizer.set_readout()
        model.eval()
        _, es_accuracy, _ = get_scores(model, es_set)
        if best_es_accuracy is None or es_accuracy > best_es_accuracy:
            best_es_accuracy = es_accuracy
            best_params = model.state_dict()
            counter_es = 0
        else:
            counter_es += 1
            if counter_es > patience:
                model.load_state_dict(best_params)
                break
        if callback is not None:
            callback(model)
    print(f'Best ES accuracy: {best_es_accuracy} after {epoch + 1} epochs')
    model.eval()
    return model

def select_optim(configs, validation_filepath, dataloaders, epochs=10, patience=0):
    """
    Model selection pipeline to get best configuration for a given n-octaveband
    dataset. The hyperparameters explored are cut_secs, optimiser and learning rate
    (only inf optimiser != RR). Saves metrics and relative plots.

    args:
     - configs: list of hyperparameter configurations;
     - dataloaders: list of dataset splits for each cut_sec value;
     - epochs: #epochs;
     - patience: n/a.

    returns:
     - results: list of result dictionaries.
    """
    results = {}
    for k, (tr_set, es_set, vl_set, _) in dataloaders.items():
        results[k] = []
        criterion = torch.nn.CrossEntropyLoss()
        o_vl_loss, o_vl_accuracy, o_cm = get_scores(OriginalModel(classes, clap_model.get_text_embeddings, device=device), vl_set)
        results[k].append(dict(metrics=dict(type_learning='original', accuracy=o_vl_accuracy, loss=o_vl_loss, cm=o_cm.tolist())))
        for config in configs:
            config_label = str(config['optimizer'])
            model = train(tr_set, es_set, config, epochs)
            vl_loss, vl_accuracy, cm = get_scores(model, vl_set)
            # PLOT START
            disp_orig = ConfusionMatrixDisplay(confusion_matrix=o_cm, display_labels=classes)
            disp_ft = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
            fig, axs = plt.subplots(2, figsize=(15, 20))

            axs[0].set_title(f'Original {k} acc={round(o_vl_accuracy * 100, 2)}% loss={round(o_vl_loss, 4)}')
            disp_orig.plot(xticks_rotation='vertical', ax=axs[0])

            axs[1].set_title(f'Finetuning {k} acc={round(vl_accuracy * 100, 2)}% loss={round(vl_loss, 4)} config={config_label}')
            disp_ft.plot(xticks_rotation='vertical', ax=axs[1])

            plt.tight_layout()

            plt.savefig(os.path.join(validation_filepath, f'{k}_{config_label}.png'))

            results[k].append(dict(metrics=dict(type_learning='finetuning', accuracy=vl_accuracy,
                                                loss=vl_loss, cm=cm.tolist()), hyperparams=config))
    return results

def select_optim_mainloop(validation_filepath):
    """
    Runs the model selection loop with desired hyperparameters.
    Dumps results into json file.

    args:
     - validation_filepath: path where to store validation results.
    """
    ms_results = select_optim([
            *[
                {
                    'optimizer': {
                            'builder': 'RR',
                            'reg': reg,
                    }
                } for reg in [0, 0.001, 0.01, 0.1, 1, 10, 50, 100, 150]
            ],
            *[
                {
                    'optimizer': {
                            'builder': builder,
                            'lr': lr,
                    }
                } for builder in ['SGD', 'Adam'] for lr in [0.1, 0.01, 0.001, 0.0001]
            ],
            ],
            validation_filepath,
            dataloaders,
            epochs=epochs,
            patience=patience,
    )

    json.dump(ms_results, open(os.path.join(validation_filepath, 'validation_ms_results.json'), 'w'))

    for k, results in ms_results.items():
        results.sort(key=lambda x: x['metrics']['accuracy'], reverse=True)
    ms_results = {k: ms_results[k] for k in sorted(ms_results, key=lambda x: ms_results[x][0]['metrics']['accuracy'], reverse=True)}
    
    values = []
    for k, v in ms_results.items():
        for t in v:
            values.append(dict(
                time=k,
                **(t['hyperparams']['optimizer'] if 'hyperparams' in t else {}),
                **t['metrics'],
            ))

    df = pd.DataFrame(values)

    df = df.sort_values('accuracy', ascending=False)

    df.to_csv(os.path.join(validation_filepath, 'validation_ms_results.csv'))


    for k, results in ms_results.items():
        print(k, results)

