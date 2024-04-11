#!/bin/bash

# Path of the file to check whether trained models have already been downloaded
file_path="./pretraining/decoder/pretrained_decoder.pth"

# Check and run application
if [ -e "$file_path" ]; then
	echo "Running gradio application to show translation examples..."
	python appli.py
else
	bash get_files.sh
	echo "Running gradio application to show translation examples..."
        python appli.py
fi 
