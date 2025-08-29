#!/bin/bash

echo "Creating virtual environment..."
python -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

# TODO: adjust filepath to correct ../dataSEC filepath
echo "Creation of working directories..."
mkdir -p ../dataSEC
mkdir -p ../dataSEC/PREPROCESSED_DATASET
mkdir -p ../dataSEC/results
mkdir -p ../dataSEC/results/validation
mkdir -p ../dataSEC/results/finetuned_model
mkdir -p ../dataSEC/testing
mkdir -p ../dataSEC/testing/PREPROCESSED_DATASET
mkdir -p ../dataSEC/testing/results
mkdir -p ../dataSEC/testing/results/validation
mkdir -p ../dataSEC/testing/results/finetuned_model

echo "Installation completed successfully. Virtual environment correctly set up."
