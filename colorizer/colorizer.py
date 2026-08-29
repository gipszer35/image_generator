import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sys
import torchvision.transforms as T
from dataclasses import dataclass
import datetime


def is_colab():
    return "COLAB_GPU" in os.environ


@dataclass
class Config:
    root_dir: str
    batch_size: int
    images_dir: str
    work_dir: str
    content_drive: str

    image_size: int = 512
    lr: float = 2e-5

    @property
    def checkpoint_path(self):
        return os.path.join(self.work_dir, "colorizer_512x512.pt")


def create_config() -> Config:
    content_drive: str = "/content/drive"
    if is_colab():
        root_dir = os.path.join(content_drive, "MyDrive")
        work_dir = os.path.join(root_dir, "ImageGenerator", "colorizer")
        images_dir = os.path.join(root_dir, "images")
        batch_size = 10
    else:
        root_dir = "../"
        work_dir = "./"
        batch_size = 2
        images_dir = "../images/my-images/"

    return Config(
        root_dir=root_dir,
        work_dir=work_dir,
        batch_size=batch_size,
        images_dir=images_dir,
        content_drive=content_drive,
    )


config = create_config()

if is_colab():
    if not os.path.ismount(config.content_drive):
        from google.colab import drive

        drive.mount(config.content_drive)

sys.path.append(config.work_dir)
sys.path.append(config.root_dir)
import my_common as my
import colorizer_common as cc

logger = my.create_logger()


class UNetColorizer(nn.Module):
    """
    Standard U-Net architecture extended to 6 stages for 512x512 image colorization.
    Downsamples the input 6 times to reach an 8x8 bottleneck, preserving deep
    structural details and merging them with low-resolution spatial color hints.
    """

    class ConvBlock(nn.Module):
        """Standard convolution block containing Conv2d, BatchNorm2d, and ReLU layers."""

        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=3, padding=1, bias=False
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )

        def forward(self, x):
            return self.conv(x)

    def __init__(self):
        super().__init__()
        c1 = 64
        c2 = 128
        c3 = 256
        c4 = 512
        c5 = 1024
        bn = 1024 + 512  # bottleneck channels size

        # Extended channel configurations for the 6-stage downsampling path
        encoder_channels = [1, c1, c2, c3, c4, c5, bn]

        # Multi-stage feature extraction and downsampling modules
        self.encoders = nn.ModuleList(
            [
                UNetColorizer.ConvBlock(encoder_channels[i], encoder_channels[i + 1])
                for i in range(len(encoder_channels) - 1)
            ]
        )
        # Increased to 6 pools to match the 6-stage encoder path
        self.pools = nn.ModuleList([nn.MaxPool2d(2) for _ in range(6)])

        # Dedicated processing network for the low-resolution color hints
        self.color_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Central processing block that merges encoder features (1024) and color features (128)
        self.bottleneck = UNetColorizer.ConvBlock(bn + 128, bn)

        # Extended channel configurations for the 6-stage upsampling path
        decoder_channels = [c5, c4, c3, c2, c1, c1]

        # Spatial upsampling modules via transposed convolutions
        self.up_samplers = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
                for in_ch, out_ch in zip([bn] + decoder_channels[:-1], decoder_channels)
            ]
        )

        skip_channels = list(reversed(encoder_channels[1:]))

        self.decoders = nn.ModuleList(
            [
                UNetColorizer.ConvBlock(up_ch + skip_ch, out_ch)
                for up_ch, skip_ch, out_ch in zip(
                    decoder_channels,
                    skip_channels,
                    decoder_channels,
                )
            ]
        )

        # Final projection layer to map features back to standard RGB channels
        self.final_conv = nn.Conv2d(c1, 3, kernel_size=1)

    def forward(self, grayscale, color_hint):
        # 1. Feature extraction loop storing intermediate maps for skip connections
        skip_connections = []
        x = grayscale

        for enc, pool in zip(self.encoders, self.pools):
            s = enc(x)
            skip_connections.append(s)
            x = pool(s)

        # 2. Process external color guidance and concatenate at the lowest resolution
        hint_features = self.color_encoder(color_hint)
        bottleneck_input = torch.cat((x, hint_features), dim=1)
        x = self.bottleneck(bottleneck_input)

        # 3. Reconstruction loop combining upsampled features with corresponding skip maps
        for up, dec, s in zip(
            self.up_samplers, self.decoders, reversed(skip_connections)
        ):
            x = up(x)
            x = torch.cat((x, s), dim=1)
            x = dec(x)

        # Output bounded between [-1, 1] via hyperbolic tangent activation
        return torch.tanh(self.final_conv(x))


class ColorizerTrainer(cc.ColorizerTrainerBase):

    def __init__(self, model, num_epochs):
        super().__init__(model, config.lr)
        self.step = 0
        self.criterion = my.SRLoss(perceptual_weight=0.1)
        self.multi_loss_tracker = my.MultiLossTracker()

        dataset = my.cropped_dataset(
            config.images_dir,
            crop_size=config.image_size,
            max_num_patches_per_image=1,
            keep_first_full_scale=False,
        )
        dataloader_factory = cc.DataLoaderFactory(config, dataset)
        self.dataloader = dataloader_factory.get_dataloader()

        self.visualizer = cc.Visualizer()

        self.num_epochs = num_epochs
        self.checkpoint_path = config.checkpoint_path

    def train_step(self, real, epoch):
        real = real.to(my.DEVICE)

        # Extract grayscale target and generate 8x8 low-res color hint
        gray = T.functional.rgb_to_grayscale(real, num_output_channels=1)
        hint = F.interpolate(real, size=(8, 8), mode="area")

        # Predict the full RGB reconstruction using our hint-guided model
        pred_rgb = self.model(gray, hint)
        # Calculate reconstruction mean squared error
        mse_loss = ((pred_rgb - real) ** 2).mean()
        sr_loss = self.criterion(pred_rgb, real)
        loss = mse_loss * 7 + sr_loss

        avg_loss = self.multi_loss_tracker.calculate_loss("loss", loss)
        avg_mse_loss = self.multi_loss_tracker.calculate_loss("mse_loss", mse_loss)
        avg_sr_loss = self.multi_loss_tracker.calculate_loss("sr_loss", sr_loss)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step % 100 == 0:
            if self.step % 7 == 0:
                logger.info("Debug trick enabled: color hint replaced")
                # DEBUG TRICK: Replace the entire color hint of batch index 0 with batch index 1
                # If the model is working correctly, object 0 should take on the colors of object 1
                hint[0] = hint[1].clone()
                pred_rgb = self.model(gray, hint)

            self.save_checkpoint()
            current_lr = self.optimizer.param_groups[0]["lr"]
            now = datetime.datetime.now()
            logger.info(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
            avg_loss = self.multi_loss_tracker.calculate_loss("loss", loss.item())

            logger.info(
                f"Epoch: {epoch + 1}/{self.num_epochs}\n"
                f"Step: {self.step:,}\n"
                f"Loss: {loss.item():.6f}\n"
                f"Avg Loss: {avg_loss:.6f}\n"
                f"Avg MSE Loss: {avg_mse_loss:.6f}\n"
                f"Avg SR Loss: {avg_sr_loss:.6f}\n"
                f"LR: {current_lr:.2e}\n"
                f"Batch: {real.size(0)}"
            )

            with torch.no_grad():
                self.visualizer.debug_colorizer(
                    real[0],
                    gray[0],
                    hint[0],
                    pred_rgb[0],
                )

        self.step += 1
        return

    def train(self):
        for epoch in range(self.num_epochs):
            for real, _ in self.dataloader:
                self.train_step(real, epoch)


def train():
    model = UNetColorizer().to(my.DEVICE)
    trainer = ColorizerTrainer(model, num_epochs=100000)
    trainer.load_or_init()
    trainer.train()


if __name__ == "__main__":
    logger.info(f"Batch size: {config.batch_size}")
    train()
