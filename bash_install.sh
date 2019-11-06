#!/bin/bash

conda update conda -y
conda env create -f environment.yml
source activate myenv
jupyter notebook