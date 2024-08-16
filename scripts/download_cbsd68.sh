#!/bin/bash

DOWNLOAD_PATH=data/dcnn/cbsd68

# Create data directory if it doesn't exist
mkdir -p $DOWNLOAD_PATH

# Download the dataset
echo "Downloading CBSD68 dataset..."
wget -q --show-progress -O ${DOWNLOAD_PATH}/CBSD68.zip https://github.com/clausmichele/CBSD68-dataset/archive/refs/heads/master.zip

# Unzip the dataset
echo "Unzipping CBSD68 dataset..."
unzip -q ${DOWNLOAD_PATH}/CBSD68.zip -d ${DOWNLOAD_PATH}

# Move the images to the test directory
mv ${DOWNLOAD_PATH}/CBSD68-dataset-master/CBSD68/* ${DOWNLOAD_PATH}/

# Cleanup
echo "Cleaning up..."
rm -rf ${DOWNLOAD_PATH}/CBSD68-dataset-master
rm ${DOWNLOAD_PATH}/CBSD68.zip

echo "Download and extraction complete."
