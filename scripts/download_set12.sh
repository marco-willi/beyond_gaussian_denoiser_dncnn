#!/bin/bash

DOWNLOAD_PATH=data/dcnn/Set12

# Create data directory if it doesn't exist
mkdir -p ${DOWNLOAD_PATH}

# Download the Set12 dataset
echo "Downloading Set12 dataset..."
wget -q --show-progress -O ${DOWNLOAD_PATH}/Set12.zip https://github.com/cszn/DnCNN/archive/refs/heads/master.zip

# Unzip the dataset
echo "Unzipping Set12 dataset..."
unzip -q ${DOWNLOAD_PATH}/Set12.zip -d ${DOWNLOAD_PATH}

# Move the images to the test directory
mv ${DOWNLOAD_PATH}/DnCNN-master/testsets/Set12/* ${DOWNLOAD_PATH}/

# Cleanup
echo "Cleaning up..."
rm -rf ${DOWNLOAD_PATH}/DnCNN-master
rm ${DOWNLOAD_PATH}/Set12.zip

echo "Download and extraction complete."
