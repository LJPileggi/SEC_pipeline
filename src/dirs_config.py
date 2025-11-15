import os

__all__ = ["basedir",
           "basedir_raw",
           "basedir_preprocessed",
           "results_filepath_project",
           "results_validation_filepath_project",
           "model_filepath"
           ]

### standard dirs ###
# cineca base directory     
basedir = os.getenv('NODE_TEMP_BASE_DIR', os.path.join('..', 'dataSEC'))

# raw audio files directory
basedir_raw = os.path.join(basedir, 'RAW_DATASET')

# directory for audio embeddings
basedir_preprocessed = os.path.join(basedir, 'PREPROCESSED_DATASET')

# diretory with model selection results
results_filepath_project = os.path.join(basedir, 'results')

# validation results directory
results_validation_filepath_project = os.path.join(results_filepath_project, 'validation')

# directory for the saved model
model_filepath = os.path.join(results_filepath_project, 'finetuned_model')
