#!/bin/bash

# Download files from google drives
echo "Downloading trained models/tokenizers and prepare files:"
echo ""

# Download
python download.py
echo ""

# Prepare files
echo "Preparing files..."
tar -xf models_and_tokenizers.tar.xz
mv ./Translator/english_tokenizer ./pretraining/decoder/
mv ./Translator/pretrained_decoder.pth ./pretraining/decoder
mv ./Translator/french_tokenizer ./pretraining/encoder/
mv ./Translator/pretrained_encoder.pth ./pretraining/encoder
mv ./Translator/trained_transformer.pth ./
rm -r Translator/
rm models_and_tokenizers.tar.xz
echo ""
