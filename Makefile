.PHONY: docker-build docker-run docker-push download-cbsd68 download-set12 tests train \
	download-bsds500 download-raise1k extract-raise1k data

#################################################################################
# GLOBALS                                                                       #
#################################################################################

ifneq (,$(wildcard ./.env))
    include .env
    export
endif

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)

#################################################################################
# COMMANDS                                                                      #
#################################################################################


help:	## Show this help.
	@sed -ne '/@sed/!s/## //p' $(MAKEFILE_LIST)


##### DOCKER

docker-build:	## Build docker image
	docker build -t ${CONTAINER_REGISTRY}:${IMAGE_TAG} .

docker-run:	## Run docker container
	docker run \
		-v ${HOST_CODE_PATH}:${CODE_PATH} \
		-v ${HOST_DATA_PATH}:${DATA_PATH} \
		-it -d \
		--gpus=all \
		--shm-size 12G \
		--name ${CONTAINER_NAME} ${CONTAINER_REGISTRY}:${IMAGE_TAG}

docker-push: ## Push image to registry
	docker login -u ${GIT_USER_NAME} -p ${CONTAINER_REGISTRY_PUSH_TOKEN} ${CONTAINER_REGISTRY}
	docker push ${CONTAINER_REGISTRY}:${IMAGE_TAG}

##### DATA

download-cbsd68: ## Download test data Berkeley
	bash scripts/download_cbsd68.sh

download-bsds500: ## Download train data Berkeley
	bash scripts/download_bsds500.sh

download-set12: ## Download test data 12 images
	bash scripts/download_set12.sh

download-raise1k: ## Download csv data
	bash scripts/download_raise1k.sh

extract-raise1k: ## Extract and Download images
	python scripts/download_raise1k.py \
	--csv_path ${CODE_PATH}/data/RAISE1k/RAISE_1k.csv \
	--output_path ${CODE_PATH}/data/RAISE1k \
	--download

data: download-cbsd68 download-set12 download-bsds500 download-raise1k extract-raise1k


##### TRAIN & DEV

tests: ## run tests
	pytest tests/

train: ## run training
	cd ./src/gaussian_denoiser; python train.py experiment=$(experiment)


train-slurm: ## run training on slurm cluster
	cd ./scripts; export EXPERIMENT=$(experiment); sbatch slurm.sh
