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

# cineca base directory     
basedir = os.path.join('..', 'dataSEC')

# raw audio files directory
basedir_raw = os.path.join(basedir, 'RAW_DATASET')

# testing directory
basedir_testing = os.path.join(basedir, 'testing')

# directory for audio embeddings
basedir_preprocessed = os.path.join(basedir, 'PREPROCESSED_DATASET')

# diretory with model selection results
results_filepath_project = os.path.join(basedir, 'results')

# validation results directory
results_validation_filepath_project = os.path.join(results_filepath_project, 'validation')

# directory for the saved model
model_filepath = os.path.join(results_filepath_project, 'finetuned_model')

# testing directories
basedir_raw_test = os.path.join(basedir_testing, 'RAW_DATASET')
basedir_preprocessed_test = os.path.join(basedir_testing, 'PREPROCESSED_DATASET')
results_filepath_project_test = os.path.join(basedir_testing, 'results')
results_validation_filepath_project_test = os.path.join(results_filepath_project_test, 'validation')
model_filepath_test = os.path.join(results_filepath_project_test, 'finetuned_model')
