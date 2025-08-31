import os
import json
import yaml
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

def gen_log(log_path, cut_secs, ic, di, results, round_, finish_class, divisions_xc_sizes_names, noise_perc, seed):
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
     - seed: random seed for track permutations inside the same class.
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
    with open(os.path.join(log_path, "log.json"), 'w') as f:
        json.dump(log, f)
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
    Reads the log generated through gen_log.
    
    returns:
     - log: a dictionary containing the logging information.
    """
    with open(os.path.join(log_path, "log.json"), 'r') as f:
        log = json.load(f)
    return log

def delete_log(log_path):
    """
    Deletes the log file (if exists) after a complete embedding run.
    """
    try:
        os.remove(os.path.join(log_path, "log.json"))
    except FileNotFoundError:
        return
