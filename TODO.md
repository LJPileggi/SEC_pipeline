
# TODO file

### Pending

```diff
- /src/tests ;  27/10/2025 - pending ; write new tests for all new utils functions and classes
```
---------------------------------------------------------------------------------------------------------------
```diff
- /src/distributed_training.py ; 27/10/2025 - pending ; rewrite functions according to new data framework
```
---------------------------------------------------------------------------------------------------------------
```diff
! /src/explainability/LMAC.py ; listenable_wav_from_n_octaveband ; 06/08/2025 - pending ; set correct directory to save explanations
!    from listenable_wav_from_n_octaveband
! /src/explainability/LMAC.py ; LMAC -- generate_listenable_interpretation ; 06/08/2025 - pending ; set correct reconstructed audio
!    path for generate_listenable_interpretations
+ /src/explainability/LMAC.py ; LMAC_explainer ; 06/08/2025 - pending ; finish fixing and honing the pipeline
+ /src/explainability/LMAC.py ; LMAC_explainer ; 06/08/2025 - pending ; correctly set up multi-GPU mode
+ /src/explainability/LMAC.py ; LMAC_explainer ; 06/08/2025 - pending ; finish writing comments
+ /src/explainability/LMAC.py ; Set configuration ; 10/08/2025 - pending ; change configuration file dynamically
```

### Resolved

- /src/training.py ; select_optim_mainloop ; 06/08/2025 - 10/08/2025 ; add saving of results in csv format
- /src/training.py ; Set configuration ; 10/08/2025 - 10/08/2025 ; change configuration file dynamically
- /scripts/get_clap_embeddings.py ; main ; 10/08/2025 - pending ; add args.audio_format to get_embeddings_for_n_octaveband
- /src/training.py ; Embedding generation -- split_audio_tracks ; 10/08/2025 - 10/08/2025 ; rewrite to support different audio
     formats (wav, mp3, flac etc.)
- /src/training.py ; Embedding generation -- get_embeddings_for_n_octaveband ; 10/08/2025 - 10/08/2025 ; pass different audio
     formats (wav, mp3, flac etc.) to split_audio_tracks
- /src/training.py ; Finetuned classifier training -- select_optim_mainloop ; 10/08/2025 - 10/08/2025 ; pass
     results_validation_filepath_project dynamically to account for different octaveband folders
- /scripts/classifier_finetuning.py ; main ; 10/08/2025 - 10/08/2025 ; add args.validation_filepath argument to
     select_optim_mainloop
- /src/models.py ; Set configuration ; 10/08/2025 - 10/08/2025 ; change configuration file dynamically
- /src/training.py ; Embedding generation -- split_audio_tracks ; 06/08/2025 - 11/08/2025 ; configure embedding generation for
     multi-GPU run
- /src/models.py ; CLAP_initializer ; 06/08/2025 - 11/08/2025 ; configure CLAP to multi-GPU training
- /scr/utils.py ; Directory organisation ; 05/08/2025 - 18/08/2025 ; insert cineca base directory
- /scr/utils.py ; Directory organisation ; 05/08/2025 - 18/08/2025 ; change basedir_preprocessed
     dinamically according to audio format to embed
- /scr/utils.py ; Directory organisation ; 05/08/2025 - 18/08/2025 ; have to create specific subfolders to basedir_preprocessed
     to account for different octave bands embeddings
- /scr/utils.py ; Model and sampling parameters ; 11/08/2025 - 18/08/2025 ; delete old parameters already yielded by config.yaml
- /scr/utils.py ; Directory organisation ; 17/08/2025 - 18/08/2025 ; implement a script that generates correct directory tree
     for project when installing the repo
- /src/distributed_training.py ; main ; 11/08/2025 - 18/08/2025 ; put main in appropriate main file
- /configs/config0.yaml ; ; 10/08/2025 - 18/08/2025 ; change the device to cineca GPUs
- /src/training.py ; Embedding generation -- split_audio_tracks ; 10/08/2025 - 18/08/2025 ; add support for other audio files like flac etc.
- /src/utils.py ; Log file functions for embedding calculation ; 11/08/2025 - 20/08/2025 ; change log file name and path to
     allow for multiple loggings relative to different configurations (n octave bands, audio formats) to exist; save them
     in appropriate directory
- /src/distributed_training.py ; main ; 11/08/2025 - 20/08/2025 ; change log file name and path to allow for multiple loggings
     relative to different configurations (n octave bands, audio formats) to exist; save them in appropriate directory
- /script/test_embeddings.py ; ; 18/08/2025 - 20/08/2025 ; expand testing by checking file generation and well functioning of functions
- /script/test_embeddings.py ; ; 20/08/2025 - 20/08/2025 ; complete testing
- /setup.sh ; ; 17/08/2025 - 29/08/2025 ; adjust filepath to correct cineca filepath
- /.gitignore ; ; 18/08/2025 - 29/08/2025 ; insert data, embeddings and results folders and files to .gitignore
- /src/distributed_training.py ; Saving functions -- save_audio_segment ; 11/08/2025 - 29/08/2025 ; add support for other audio
     files like flac etc.
- /src/distributed_training.py ; Embedding generation -- split_audio_tracks ; 11/08/2025 - 29/08/2025 ; add support for
     other audio files like flac etc.
- /src/distributed_clap_embeddings.py ; process_class_with_cut_secs ; 31/08/2025 - 31/08/2025 ; find solution for conflicting logs
- /src/distributed_clap_embeddings.py ; process_class_with_cut_secs ; 01/09/2025 - 01/09/2025 ; fix parallel stout print
- /src/distributed_clap_embeddings.py ; process_class_with_cut_secs ; 01/09/2025 - 02/09/2025 ; fix folder generation and
     create cut_secs subfolders
- /src/distributed_clap_embeddings.py ; process_class_with_cut_secs ; 01/09/2025 - 02/09/2025 ; add flag to choose whether
     or not to delete audio cuts after embedding generation
- /src/distributed_finetuning.py ; ; 18/08/2025 - 09/09/2025 ; implement test mode
- /src/distributed_finetuning.py ; ; 20/08/2025 - 09/09/2025 ; develop support for other classifier models
- /run_tests.sh ; ; 18/08/2025 - 09/09/2025 ; add other tests and implement correctness check
- src/distributed_clap_embeddings.py ; process_class_with_cut_secs ; 23/09/2025 - 23/09/2025 ; update cancelling logic according to new hdf5 framework
- /src/distributed_clap_embeddings.py ; worker_process_slurm ; 27/10/2025 - 27/10/2025 ; modify logging logic to take into
account class only (no more inner loop counters)
- /src/distributed_clap_embeddings.py ; run_distributed_slurm ; 27/10/2025 - 27/10/2025 ; fix function according to new logic
- /src/distributed_clap_embeddings.py ; local_worker_process ; 27/10/2025 - 27/10/2025 ; readapt function according to new
worker_process_slurm version
- /src/distributed_clap_embeddings.py ; run_local_multiprocess ; 27/10/2025 - 27/10/2025 ; readapt function according to new
run_distributed_slurm version
- /scr/utils.py ; read_log, write_log, delete_log ; 27/10/2025 - 27/10/2025 ; readapt those functions to take into account
new logging logic (possibly simply use logging library)
