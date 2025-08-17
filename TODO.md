
# TODO file

### Pending

```diff
+ /scr/utils.py ; Directory organisation ; 05/08/2025 - pending ; insert cineca base directory
+ /scr/utils.py ; Directory organisation ; 05/08/2025 - pending ; change basedir_preprocessed
+    dinamically according to audio format to embed
! /scr/utils.py ; Directory organisation ; 05/08/2025 - pending ; have to create specific subfolders to basedir_preprocessed
!    to account for different octave bands embeddings
! /src/utils.py ; Log file functions for embedding calculation ; 11/08/2025 - pending ; change log file name and path to
!    allow for multiple loggings relative to different configurations (n octave bands, audio formats) to exist; save them
!    in appropriate directory
+ /scr/utils.py ; Model and sampling parameters ; 11/08/2025 - delete old parameters already yielded by config.yaml
+ /scr/utils.py ; Directory organisation ; 17/08/2025 - pending ; implement a script that generates correct directory tree
+    for project when installing the repo
```
--------------------------------------------------------------------------
```diff
+ /src/training.py ; Embedding generation -- split_audio_tracks ; 10/08/2025 - pending ; add support for other audio files like flac etc.
```
--------------------------------------------------------------------------
```diff
+ /src/distributed_training.py ; Saving functions -- save_audio_segment ; 11/08/2025 - pending ; add support for other audio
+    files like flac etc.
+ /src/distributed_training.py ; Embedding generation -- split_audio_tracks ; 11/08/2025 - pending ; add support for
+    other audio files like flac etc.
! /src/distributed_training.py ; main ; 11/08/2025 - pending ; put main in appropriate main file
! /src/distributed_training.py ; main ; 11/08/2025 - pending ; change log file name and path to allow for multiple loggings
!    relative to different configurations (n octave bands, audio formats) to exist; save them in appropriate directory
```
--------------------------------------------------------------------------
```diff
! /src/explainability.py ; listenable_wav_from_n_octaveband ; 06/08/2025 - pending ; set correct directory to save explanations
!    from listenable_wav_from_n_octaveband
! /src/explainability.py ; LMAC -- generate_listenable_interpretation ; 06/08/2025 - pending ; set correct reconstructed audio
!    path for generate_listenable_interpretations
+ /src/explainability.py ; LMAC_explainer ; 06/08/2025 - pending ; finish fixing and honing the pipeline
+ /src/explainability.py ; LMAC_explainer ; 06/08/2025 - pending ; correctly set up multi-GPU mode
+ /src/explainability.py ; LMAC_explainer ; 06/08/2025 - pending ; finish writing comments
+ /src/explainability.py ; Set configuration ; 10/08/2025 - pending ; change configuration file dynamically
```
--------------------------------------------------------------------------
```diff
+ /configs/config0.yaml ; ; 10/08/2025 - pending ; change the device to cineca GPUs
```

### Resolved

- /src/training.py ; select_optim_mainloop ; 06/08/2025 - 10/08/2025 ; add saving of results in csv format
- /src/training.py ; Set configuration ; 10/08/2025 - 10/08/2025 ; change configuration file dynamically
- /scripts/get_clap_embeddings.py ; main ; 10/08/2025 - pending ; add args.audio_format to get_embeddings_for_n_octaveband
- /src/training.py ; Embedding generation -- split_audio_tracks ; 10/08/2025 - 10/08/2025 ; rewrite to support different audio
-    formats (wav, mp3, flac etc.)
- /src/training.py ; Embedding generation -- get_embeddings_for_n_octaveband ; 10/08/2025 - 10/08/2025 ; pass different audio
-    formats (wav, mp3, flac etc.) to split_audio_tracks
- /src/training.py ; Finetuned classifier training -- select_optim_mainloop ; 10/08/2025 - 10/08/2025 ; pass
-    results_validation_filepath_project dynamically to account for different octaveband folders
- /scripts/classifier_finetuning.py ; main ; 10/08/2025 - 10/08/2025 ; add args.validation_filepath argument to
-    select_optim_mainloop
- /src/models.py ; Set configuration ; 10/08/2025 - 10/08/2025 ; change configuration file dynamically
! /src/training.py ; Embedding generation -- split_audio_tracks ; 06/08/2025 - 11/08/2025 ; configure embedding generation for
!    multi-GPU run
+ /src/models.py ; CLAP_initializer ; 06/08/2025 - 11/08/2025 ; configure CLAP to multi-GPU training
