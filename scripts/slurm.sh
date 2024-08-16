#!/bin/bash -l
#SBATCH --job-name="train_denoiser"
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=6
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --partition=performance


source ../.env

echo "Start training Denoiser"

# Singularity Image
SIF_DIR=${HOST_DATA_PATH}/singularity
SIF_FILE=${CONTAINER_NAME}_${IMAGE_TAG}.sif


mkdir -p ${SIF_DIR}

# Singularity Bindings and Env variables
export SINGULARITY_DOCKER_USERNAME='$oauthtoken'
export SINGULARITY_DOCKER_PASSWORD=${CONTAINER_REGISTRY_PULL_TOKEN}
export SINGULARITY_BIND="${HOST_DATA_PATH}:${DATA_PATH},${HOST_CODE_PATH}:${CODE_PATH}"
export HF_HOME=${HF_HOME}


# Pull Image
cd $SIF_DIR
singularity pull --name ${SIF_FILE} docker://${CONTAINER_REGISTRY}:${IMAGE_TAG}

singularity exec --nv ${SIF_DIR}/${SIF_FILE} python ${CODE_PATH}/src/gaussian_denoiser/train.py experiment=${EXPERIMENT}
