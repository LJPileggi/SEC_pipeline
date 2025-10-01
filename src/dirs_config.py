import os

__all__ = ["basedir",
           "basedir_raw",
           "basedir_preprocessed",
           "results_filepath_project",
           "results_validation_filepath_project",
           "model_filepath",
           "basedir_raw_test",
           "basedir_preprocessed_test",
           "results_filepath_project_test",
           "results_validation_filepath_project_test",
           "model_filepath_test"
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


### testing dirs ###
# testing directory
basedir_testing = os.getenv('NODE_TEMP_BASE_DIR', os.path.join('..', 'dataSEC', 'testing'))

# raw audio files directory
basedir_raw_test = os.path.join(basedir_testing, 'RAW_DATASET')

# directory for audio embeddings
basedir_preprocessed_test = os.path.join(basedir_testing, 'PREPROCESSED_DATASET')

# diretory with model selection results
results_filepath_project_test = os.path.join(basedir_testing, 'results')

# validation results directory
results_validation_filepath_project_test = os.path.join(results_filepath_project_test, 'validation')

# directory for the saved model
model_filepath_test = os.path.join(results_filepath_project_test, 'finetuned_model')
