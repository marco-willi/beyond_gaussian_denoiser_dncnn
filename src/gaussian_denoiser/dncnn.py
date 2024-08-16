import lightning as L
import torch
import torch.nn as nn
import wandb
from huggingface_hub import PyTorchModelHubMixin
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from torchmetrics.regression import MeanSquaredError


class DnCNN(nn.Module, PyTorchModelHubMixin):
    def __init__(self, in_channels, depth, channels=64, kernel_size=3, normalization="BN"):
        super().__init__()

        NORMALIZATION_MAP = {
            "BN": lambda: nn.BatchNorm2d(channels),
            "CN": lambda: ChannelNormalization(channels),
        }

        self.input_layers = nn.ModuleList(
            [
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=kernel_size - 2,
                    bias=True,
                ),
                nn.ReLU(),
            ]
        )

        self.hidden_layers = nn.ModuleList()
        for _ in range(depth - 1):
            self.hidden_layers.append(
                nn.Conv2d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=kernel_size - 2,
                    bias=False,
                )
            )
            self.hidden_layers.append(NORMALIZATION_MAP[normalization]())
            self.hidden_layers.append(nn.ReLU())

        self.output_layer = nn.Conv2d(
            in_channels=channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            padding=kernel_size - 2,
            bias=False,
        )

    def forward(self, x):
        for layer in self.input_layers:
            x = layer(x)

        for layer in self.hidden_layers:
            x = layer(x)
        return self.output_layer(x)


class DnCNNModule(L.LightningModule):
    def __init__(self, in_channels, depth, channels=64, kernel_size=3, normalization: str = "BN"):
        super().__init__()
        self.model = DnCNN(
            in_channels,
            depth,
            channels=channels,
            kernel_size=kernel_size,
            normalization=normalization,
        )
        self.loss_val = MeanSquaredError()
        self.psnr_val = PeakSignalNoiseRatio((0, 1), dim=(1, 2, 3), reduction="elementwise_mean")
        self.psnr_train = PeakSignalNoiseRatio((0, 1), dim=(1, 2, 3), reduction="elementwise_mean")

        self.ssim_val = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.ssim_train = StructuralSimilarityIndexMeasure(data_range=1.0)
        self.num_images_to_log = 9

        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        image, noisy_image, noise = batch
        noise_hat = self(noisy_image)
        loss = nn.functional.mse_loss(noise, noise_hat)
        image_hat = noisy_image - noise_hat
        image_hat_clipped = torch.clip(image_hat, 0, 1)

        # Metrics and logging
        self.psnr_train(image, image_hat_clipped)
        self.ssim_train(image, image_hat_clipped)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_psnr", self.psnr_train, on_step=False, on_epoch=True)
        self.log("train_ssim", self.ssim_train, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image, noisy_image, noise = batch
        noise_hat = self(noisy_image)
        loss = self.loss_val(noise, noise_hat)

        image_hat = noisy_image - noise_hat
        image_hat_clipped = torch.clip(image_hat, 0, 1)

        # loss_naive_baseline = (noise**2).mean()
        self.psnr_val(image, image_hat_clipped)
        self.ssim_val(image, image_hat_clipped)
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_psnr", self.psnr_val, on_step=False, on_epoch=True)
        self.log("val_ssim", self.ssim_val, on_step=False, on_epoch=True)

        # pick images
        if batch_idx < self.num_images_to_log:
            # pick random patch
            self.reconstructed_images.append((image_hat_clipped).cpu()[0])
            self.images.append((image).cpu()[0])
            self.noise_images.append((noise).cpu()[0])
            self.noise_hat_images.append((noise_hat).cpu()[0])
            self.noise_diff_images.append((noise - noise_hat).cpu()[0])
            self.noisy_images.append((noisy_image).cpu()[0])

        return loss

    def on_validation_start(self):
        self.reconstructed_images = list()
        self.images = list()
        self.noise_images = list()
        self.noise_hat_images = list()
        self.noise_diff_images = list()
        self.noisy_images = list()

    def on_validation_end(self):

        for data, text in zip(
            [
                self.reconstructed_images,
                self.noisy_images,
                self.images,
                self.noise_images,
                self.noise_hat_images,
                self.noise_diff_images,
            ],
            ["Rec", "Noisy Image", "Image", "Noise", "Noise_Hat", "Noise_Diff"],
        ):
            wandb.log({f"{text}": [wandb.Image(img, caption=text) for img in data]})

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters())
        return optimizer


class ChannelNormalization(nn.Module):
    def __init__(self, n_channels):
        super().__init__()
        self.n_channels = n_channels
        self.norm = torch.nn.LayerNorm(n_channels, eps=1e-6)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2).contiguous()  # (N, H, W, C) -> (N, C, H, W)
        return x
