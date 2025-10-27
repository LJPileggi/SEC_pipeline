from .dirs_config import *
from .utils import *
from .losses import accuracy, get_scores, RR, build_optimizer
from .models import CLAP_initializer, spectrogram_n_octaveband_generator, OriginalModel, FinetunedModel
from .distributed_clap_embeddings import run_distributed_slurm, run_local_multiprocess
# from .distributed_finetuning import select_optim_distributed
# from .explainability.LMAC import listenable_wav_from_n_octaveband, Decoder, LMAC, LMAC_explainer

__all__ = ["basedir",
           "basedir_raw",
           "basedir_preprocessed",
           "results_filepath_project",
           "results_validation_filepath_project",
           "model_filepath",

           "basedir_testing",
           "basedir_raw_test",
           "basedir_preprocessed_test",
           "results_filepath_project_test",
           "results_validation_filepath_project_test",
           "model_filepath_test",

           "get_config_from_yaml",
           "write_log",
           "join_logs",
           "HDF5DatasetManager",
           "HDF5EmbeddingDatasetsManager",
           "combine_hdf5_files",
           "setup_environ_vars",
           "setup_distributed_environment",
           "cleanup_distributed_environment",

           "accuracy",
           "get_scores",
           "RR",
           "build_optimizer",
           "CLAP_initializer",
           "spectrogram_n_octaveband_generator",
           "OriginalModel",
           "FinetunedModel",

           "run_distributed_slurm",
           "run_local_multiprocess"#,
           # "select_optim_distributed",

           # "listenable_wav_from_n_octaveband",
           # "Decoder",
           # "LMAC",
           # "LMAC_explainer"
           ]
