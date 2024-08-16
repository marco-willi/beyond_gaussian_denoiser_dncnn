import logging
import os
from datetime import datetime
from pathlib import Path

import hydra
import lightning as L
import pyrootutils
import wandb
from icecream import ic
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from omegaconf import DictConfig, OmegaConf

from gaussian_denoiser import data, dataset, dncnn, transforms

root = pyrootutils.setup_root(
    search_from=__file__,
    indicator="pyproject.toml",
    project_root_env_var=True,
    dotenv=True,
    pythonpath=False,
    cwd=True,
)

PROJECT_ROOT = os.getenv("PROJECT_ROOT")

DATASET_CLASS = {"ImageFolder": dataset.ImageFolderDataset, "LocalHFDataset": dataset.HFDataset}


@hydra.main(config_path=f"{PROJECT_ROOT}/config", config_name="train")
def train_model(cfg: DictConfig):

    cfg_experiment = cfg.experiment

    ic(OmegaConf.to_container(cfg, resolve=True))

    L.seed_everything(cfg_experiment.train.random_seed, workers=True)

    # Logging
    log_dir = Path(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
    ic(log_dir)

    if cfg.debug:
        ic("debug mode, no outputs are created")
    if not log_dir.exists() and not cfg.debug:
        log_dir.mkdir(mode=777)

    loggers = []
    if cfg.wandb.enable and not cfg.debug:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            entity=cfg.wandb.entity,
            save_dir=str(log_dir),
            id=cfg.wandb.run_id,
        )

        wandb_logger.experiment.config.update(
            OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
        )

        loggers.append(wandb_logger)
    else:
        wandb.init(mode="disabled")

    # Data
    dataset_name = cfg.experiment.data.dataset
    dataset_cfg = cfg.datasets[dataset_name]
    DatasetClass = DATASET_CLASS[dataset_cfg.dataset_type]

    ic(f"Preparing: {dataset_cfg.train_path}")
    ds_train = DatasetClass(path=dataset_cfg.train_path, cache_images=dataset_cfg.cache_images)

    ds_val = DatasetClass(path=dataset_cfg.val_path, cache_images=dataset_cfg.cache_images)

    ds_test = DatasetClass(path=dataset_cfg.test_path, cache_images=dataset_cfg.cache_images)

    # Data Transformations
    if cfg_experiment.data.train_noise_type == "awgn_only":
        train_noise = transforms.AWGNOnlyTransform(
            min_variance=cfg_experiment.data.train_noise_level_interval[0],
            max_variance=cfg_experiment.data.train_noise_level_interval[1],
        )

    if cfg_experiment.data.val_noise_type == "awgn_only":
        val_noise = transforms.AWGNOnlyTransform(
            min_variance=cfg_experiment.data.val_noise_level_interval[0],
            max_variance=cfg_experiment.data.val_noise_level_interval[1],
        )

    if cfg_experiment.data.train_noise_type == "combined":
        train_noise = transforms.CombinedTransform(
            min_quality=cfg_experiment.data.train_jpeg_min_max[0],
            max_quality=cfg_experiment.data.train_jpeg_min_max[1],
            factors=cfg_experiment.data.train_up_down_factors,
            min_variance=cfg_experiment.data.train_noise_level_interval[0],
            max_variance=cfg_experiment.data.train_noise_level_interval[1],
        )

    if cfg_experiment.data.val_noise_type == "combined":
        val_noise = transforms.CombinedTransform(
            min_quality=cfg_experiment.data.val_jpeg_min_max[0],
            max_quality=cfg_experiment.data.val_jpeg_min_max[1],
            factors=cfg_experiment.data.val_up_down_factors,
            min_variance=cfg_experiment.data.val_noise_level_interval[0],
            max_variance=cfg_experiment.data.val_noise_level_interval[1],
        )

    data_module = data.DenoisingDataModule(
        ds_train=ds_train,
        ds_val=ds_val,
        ds_test=ds_test,
        batch_size=cfg_experiment.train.batch_size,
        patch_size=cfg_experiment.data.patch_size,
        train_noise=train_noise,
        val_noise=val_noise,
        max_patches_per_image=cfg_experiment.data.max_patches_per_image,
    )

    model = dncnn.DnCNNModule(
        in_channels=cfg_experiment.model.in_channels,
        depth=cfg_experiment.model.depth,
        channels=cfg_experiment.model.channels,
        kernel_size=cfg_experiment.model.kernel_size,
        normalization=cfg_experiment.model.normalization,
    )

    if cfg.debug:
        from torchinfo import summary

        batch_size = 16
        summary(
            model,
            input_size=(
                batch_size,
                3,
                cfg_experiment.data.patch_size,
                cfg_experiment.data.patch_size,
            ),
        )

    # Callbacks
    callbacks = list()
    callbacks.append(ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min"))
    callbacks.append(
        EarlyStopping(
            monitor="val_loss",
            patience=cfg_experiment.train.early_stopping_patience,
            mode="min",
            verbose=True,
        )
    )

    trainer = L.Trainer(
        default_root_dir=f"{PROJECT_ROOT}/logs/",
        accelerator=cfg_experiment.train.device,
        check_val_every_n_epoch=1,
        log_every_n_steps=100,
        max_epochs=cfg_experiment.train.max_epochs,
        logger=loggers if len(loggers) > 0 else False,
        callbacks=callbacks,
        deterministic=True,
    )

    trainer.fit(
        model,
        data_module,
        ckpt_path=cfg_experiment.train.load_from_checkpoint
        if cfg_experiment.train.load_from_checkpoint != ""
        else None,
    )

    os.system(f"chmod -R 777 {log_dir}")


if __name__ == "__main__":
    train_model()
