#!/bin/bash

DOWNLOAD_PATH=data/RAISE1k

# Create data directory if it doesn't exist
mkdir -p ${DOWNLOAD_PATH}

# Download the dataset
echo "Downloading RAISE dataset..."
wget -q --show-progress -O ${DOWNLOAD_PATH}/RAISE_1k.zip http://loki.disi.unitn.it/RAISE/getFile.php?p=1k

# Unzip the dataset
echo "Unzipping RAISE dataset..."
unzip -q ${DOWNLOAD_PATH}/RAISE_1k.zip -d ${DOWNLOAD_PATH}

# Cleanup
echo "Cleaning up..."
rm -rf ${DOWNLOAD_PATH}/*/  # This assumes that after moving, the original subdirectory should be removed
rm ${DOWNLOAD_PATH}/RAISE_1k.zip

echo "Download and extraction complete."
