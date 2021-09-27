#!/bin/bash
# Create and activate conda environment from yml file
echo "Installing and activating conda environment."

eval "$(conda shell.bash hook)"
conda env create --file stochastic_env.yml --name stochastic_env
