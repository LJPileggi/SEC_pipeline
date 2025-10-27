import sys
sys.path.append('.')

import os
import yaml

def main():
    print("Generate new configuration file.")
    config_file = input("Insert config file name.")
    classes = []
    class_ = input("Insert classes")
    while class_ != None:
        classes.append(class_)
        class_ = input("Insert classes")
    patience = input("Insert patience")
    epochs = input("Insert epochs")
    batch_size = input("Insert batch_size")
    sampling_rate = input("Insert sampling_rate")
    ref = input("Insert ref")
    noise_perc = input("Insert noise_perc")
    seed = input("Insert seed")
    print("Insert center_freqs")
    center_freqs = []
    freq = input()
    while freq != None:
        center_freqs.append(freq)
        freq = input()
    print("Insert valid_cut_secs")
    valid_cut_secs = []
    cut_secs = input()
    while cut_secs != None:
        valid_cut_secs.append(cut_secs)
        cut_secs = input()
    train_size = input("Insert train set size")
    es_size = input("Insert es set size")
    valid_size = input("Insert valid set size")
    test_size = input("Insert test set size")
    config = {
        "classes" : classes
        "patience" : patience,
        "epochs" : epochs,
        "batch_size" : batch_size,
        "sampling_rate" : sampling_rate,
        "ref" : ref,
        "noise_perc" : noise_perc,
        "seed" : seed,
        "center_freqs" : center_freqs,
        "valid_cut_secs" : valid_cut_secs
        "train_size": train_size,
        "es_size": es_size,
        "valid_size": valid_size,
        "test_size": test_size
    }
    with open(os.path.join("configs", config_file), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    print("Config file saved")

if __name__ == "__main__":
    main()

