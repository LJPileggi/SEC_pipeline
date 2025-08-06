### TODO file

### 05/08/2025

### /scr/utils.py

# Model and sampling parameters
- change the device to cineca GPUs

# Directory organisation
- insert cineca base directory
- change basedir_preprocessed dinamically according to audio format to embed;
- have to create specific subfolders to basedir_preprocessed
      to account for different octave bands embeddings



### 06/08/2025

### /src/models.py

# CLAP_initializer
- configure CLAP to multi-GPU training



### /src/training.py

# Embedding generation
- configure embedding generation for multi-GPU run

# select_optim_mainloop
- add saving of results in csv format



### src/explainability.py

# listenable_wav_from_n_octaveband
- set correct directory to save explanations from listenable_wav_from_n_octaveband

# LMAC
# generate_listenable_interpretation
- set correct reconstructed audio path for generate_listenable_interpretations

# LMAC_explainer
- finish fixing and honing the pipeline
- correctly set up multi-GPU mode
- finish writing comments
