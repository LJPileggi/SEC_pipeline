import os
import yaml

def main():
    print("Generate new configuration file.")
    config_file = input("Insert config file name.")
    patience = input("Insert patience")
    epochs = input("Insert epochs")
    batch_size = input("Insert batch_size")
    save_log_every = input("Insert save_log_every")
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
    valid_cut_secs = []
    cut_secs = input()
    while cut_secs != None:
        valid_cut_secs.append(cut_secs)
        cut_secs = input()
    config = {
        "patience" : patience,
        "epochs" : epochs,
        "batch_size" : batch_size,
        "save_log_every" : save_log_every,
        "sampling_rate" : sampling_rate,
        "ref" : ref,
        "noise_perc" : noise_perc,
        "seed" : seed,
        "center_freqs" : center_freqs,
        "valid_cut_secs" : valid_cut_secs
    }
    with open(os.path.join("configs", config_file), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    print("Config file saved")

if __name__ == "__main__":
    main()

