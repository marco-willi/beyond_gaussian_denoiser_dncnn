#!/bin/bash

DOWNLOAD_PATH=data/dcnn/bsds500/images

# Create data directory if it doesn't exist
mkdir -p $DOWNLOAD_PATH

# Download the dataset
echo "Downloading BSDS500 dataset..."
wget -q --show-progress -O ${DOWNLOAD_PATH}/BSDS500.zip https://github.com/BIDS/BSDS500/archive/refs/heads/master.zip

# Unzip the dataset
echo "Unzipping BSDS500 dataset..."
unzip -q ${DOWNLOAD_PATH}/BSDS500.zip -d ${DOWNLOAD_PATH}

# Move the images to the desired directory
echo "Moving images to target directory..."
mv ${DOWNLOAD_PATH}/BSDS500-master/BSDS500/data/images/* ${DOWNLOAD_PATH}/

# Cleanup
echo "Cleaning up..."
rm -rf ${DOWNLOAD_PATH}/BSDS500-master
rm ${DOWNLOAD_PATH}/BSDS500.zip

echo "Download and extraction complete."
