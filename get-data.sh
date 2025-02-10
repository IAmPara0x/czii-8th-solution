#!/usr/bin/bash

set -xe

# Create directories
mkdir ./data
mkdir -p ./checkpoints/tiny-unet
mkdir -p ./checkpoints/medium-unet

echo "downloading post processed synthetic data"
kaggle datasets download -d sirapoabchaikunsaeng/czii-synthetic-dataset
unzip ./czii-synthetic-dataset.zip -d ./data/synthetic-data
rm ./czii-synthetic-dataset.zip
echo "downloaded post processed synthetic data"

echo "downloading post processed train data"
kaggle datasets download -d iamparadox/czii-train-data-zip
unzip ./czii-train-data-zip.zip -d ./data/train
rm ./czii-train-data-zip.zip
mv ./data/train/data/train/* ./data/train/
echo "downloaded post processed train data"
