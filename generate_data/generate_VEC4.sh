#!/bin/bash

# Define the save_path variable
save_path="$1"

# Check if save_path is provided
if [ -z "$save_path" ]; then
    echo "Error: save_path not provided."
    exit 1
fi

python generate_data/generate_data.py --setting uniqueness0 --dim_modalities 200 100 150 100 --save_path "$save_path"
python generate_data/generate_data.py --setting uniqueness1 --dim_modalities 200 100 150 100 --save_path "$save_path"
python generate_data/generate_data.py --setting uniqueness2 --dim_modalities 200 100 150 100 --save_path "$save_path"
python generate_data/generate_data.py --setting uniqueness3 --dim_modalities 200 100 150 100 --save_path "$save_path"
python generate_data/generate_data.py --setting redundancy --dim_modalities 200 100 150 100 --save_path "$save_path"
python generate_data/generate_data.py --setting synergy --dim_modalities 200 100 150 100 --save_path "$save_path"
