#!/bin/bash

python generate_data/generate_data.py --setting uniqueness0 --dim_modalities 200 100 150
python generate_data/generate_data.py --setting uniqueness1 --dim_modalities 200 100 150
python generate_data/generate_data.py --setting uniqueness2 --dim_modalities 200 100 150

python generate_data/generate_data.py --setting redundancy --dim_modalities 200 100 150
python generate_data/generate_data.py --setting synergy --dim_modalities 200 100 150
