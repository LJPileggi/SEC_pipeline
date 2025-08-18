#!/bin/bash

echo "Creating virtual environment..."
python -m venv .venv

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

# TODO: adjust filepath to correct cineca filepath
echo "Creation of working directories..."
mkdir -p cineca
mkdir -p cineca/PREPROCESSED_DATASET
mkdir -p cineca/results
mkdir -p cineca/results/validation
mkdir -p cineca/results/finetuned_model
mkdir -p cineca/testing
mkdir -p cineca/testing/PREPROCESSED_DATASET
mkdir -p cineca/testing/results
mkdir -p cineca/testing/results/validation
mkdir -p cineca/testing/results/finetuned_model

echo "Installation completed successfully. Virtual environment correctly set up."
