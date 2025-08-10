import os
import json
import yaml
import numpy as np

### TODOs list: ###

### Directory organisation ###
# insert cineca base directory
# change basedir_preprocessed dinamically according to audio format to embed;
# have to create specific subfolders to basedir_preprocessed
#     to account for different octave bands embeddings

###  ###

### Model and sampling parameters ###
patience = 10
epochs = 100
batch_size = 128
# TODO: change the device to cineca GPUs
device = 'cpu'
save_log_every = 1000
sampling_rate = 52100
ref = 2e-5
center_freqs = np.array([6.30e+00, 8.00e+00, 1.00e+01, 1.25e+01, 1.60e+01, 2.00e+01,
                            2.50e+01, 3.15e+01, 4.00e+01, 5.00e+01, 6.30e+01, 8.00e+01,
                            1.00e+02, 1.25e+02, 1.60e+02, 2.00e+02, 2.50e+02, 3.15e+02,
                            4.00e+02, 5.00e+02, 6.30e+02, 8.00e+02, 1.00e+03, 1.25e+03,
                            1.60e+03, 2.00e+03, 2.50e+03, 3.15e+03, 4.00e+03, 5.00e+03,
                            6.30e+03, 8.00e+03, 1.00e+04, 1.25e+04, 1.60e+04, 2.00e+04])
valid_cut_secs = list(range(10)) + [15, 20, 30]
                            
### Directory organisation ###
# cineca base directory     
# TODO: insert cineca base directory
basedir = ''
# raw audio files directory
basedir_raw = os.path.join(basedir, 'RAW_DATASET')
# directory for audio embeddings
# TODO: change basedir_preprocessed dinamically according to audio format to embed;
# TODO: have to create specific subfolders to basedir_preprocessed
#     to account for different octave bands embeddings
basedir_preprocessed = os.path.join(basedir, 'PREPROCESSED_DATASET') 
if not os.path.exists(basedir_preprocessed):
    os.mkdir(basedir_preprocessed)
# diretory with model selection results
results_filepath_project = os.path.join(basedir, 'results')
# validation results directory
results_validation_filepath_project = os.path.join(results_filepath_project, 'validation')
# directory for the saved model
model_filepath = os.path.join(results_filepath_project, 'finetuned_model')
for dir in [results_filepath_project, results_validation_filepath_project]:
    if not os.path.exists(dir):
        os.mkdir(dir)


### Get model, training and spectrogram configuration from yaml ###

def get_config_from_yaml(config_file="config0.yaml"):
    """
    Loads configuration from yaml file and yields
    variables to use in desired namespace.

    args:
     - config_file: name of config file, to be attached to its relative path.
    """
    config_path = os.path.join('..', 'configs', config_file)
    with open(config_path, 'r') as f:
        configs = yaml.load(f, Loader=yaml.SafeLoader)
    global patience
    global epochs
    global batch_size
    global device
    global save_log_every
    global sampling_rate
    global ref
    global center_freqs
    global valid_cut_secs
    patience = data["patience"]
    epochs = data["epochs"]
    batch_size = data["batch_size"]
    device = data["device"]
    save_log_every = data["save_log_every"]
    sampling_rate = data["sampling_rate"]
    ref = data["ref"]
    center_freqs = np.array(data["center_freqs"])
    valid_cut_secs = data["valid_cut_secs"]

   
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

def gen_log(cut_secs, ic, di, results, round_, finish_class, divisions_xc_sizes_names, noise_perc, seed):
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
        "round" : round,
        "finish_class" : finish_class,
        "divisions_xc_sizes_names" : divisions_xc_sizes_names,
        "noise_perc" : noise_perc,
        "seed" : seed
    }
    with open(os.path.join(basedir, "log.json"), 'w') as f:
        json.dump(log, f)
    print("Logfile saved successfully.\n"
          f"cut_secs: {cut_secs}\n"
          f"ic: {ic}\n"
          f"di: {di}\n"
          f"results: {results}\n"
          f"round: {round}\n"
          f"finish_class: {finish_class}\n"
          f"divisions_xc_sizes_names: {divisions_xc_sizes_names}\n"
          f"noise_perc: {noise_perc}\n"
          f"seed: {seed}\n"
    )

def read_log():
    """
    Reads the log generated through gen_log.
    
    returns:
     - log: a dictionary containing the logging information.
    """
    with open(os.path.join(basedir, "log.json"), 'r') as f:
        log = json.load(f)
    return log

def delete_log():
    """
    Deletes the log file (if exists) after a complete embedding run.
    """
    try:
        os.remove(os.path.join(basedir, "log.json"))
    except FileNotFoundError:
        return
