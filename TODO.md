# TODO file

### Pending

- /scr/utils.py ; Directory organisation ; 05/08/2025 - pending ; insert cineca base directory
- /scr/utils.py ; Directory organisation ; 05/08/2025 - pending ; change basedir_preprocessed
    dinamically according to audio format to embed
- /scr/utils.py ; Directory organisation ; 05/08/2025 - pending ; have to create specific subfolders to basedir_preprocessed
    to account for different octave bands embeddings
--------------------------------------------------------------------------
- /src/models.py ; CLAP_initializer ; 06/08/2025 - pending ; configure CLAP to multi-GPU training
- /src/models.py ; Set configuration ; 10/08/2025 - pending ; change configuration file dynamically
--------------------------------------------------------------------------
- /src/training.py ; Embedding generation ; 06/08/2025 - pending ; configure embedding generation for multi-GPU run
- /src/training.py ; Set configuration ; 10/08/2025 - pending ; change configuration file dynamically
--------------------------------------------------------------------------
- /src/explainability.py ; listenable_wav_from_n_octaveband ; 06/08/2025 - pending ; set correct directory to save explanations
    from listenable_wav_from_n_octaveband
- /src/explainability.py ; LMAC -- generate_listenable_interpretation ; 06/08/2025 - pending ; set correct reconstructed audio
    path for generate_listenable_interpretations
- /src/explainability.py ; LMAC_explainer ; 06/08/2025 - pending ; finish fixing and honing the pipeline
- /src/explainability.py ; LMAC_explainer ; 06/08/2025 - pending ; correctly set up multi-GPU mode
- /src/explainability.py ; LMAC_explainer ; 06/08/2025 - pending ; finish writing comments
- /src/explainability.py ; Set configuration ; 10/08/2025 - pending ; change configuration file dynamically
--------------------------------------------------------------------------
- /configs/config0.yaml ; ; 10/08/2025 - pending ; change the device to cineca GPUs

### Resolved

- /src/training.py ; select_optim_mainloop ; 06/08/2025 - 10/08/2025 ; add saving of results in csv format
