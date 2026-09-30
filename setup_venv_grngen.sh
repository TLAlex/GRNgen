#!/bin/bash

# Create the virtual environment: 
conda create --name venv_grngen python=3.10 -y

# Switch to the activated environment: 
conda activate venv_grngen

# Install the jupyter notebook kernel: 
pip install --user ipykernel

# Install jupyter notebook in venv_grngen: 
python -m ipykernel install --user --name=venv_grngen

# Install packages
pip install -r ./requirements.txt

exec bash
