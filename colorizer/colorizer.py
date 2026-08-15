import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os, sys
import matplotlib.pyplot as plt
import torchvision.transforms as T
from dataclasses import dataclass
from collections import deque
import datetime
import random
import numpy as np


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

sys.path.append(config.root_dir)
sys.path.append(config.work_dir)
import my_common as my

logger = my.create_logger()


class DataLoaderFactory:
    """Factory class to load and augment dataset, then return a configured DataLoader."""

    class Random90Rotation:
        """Custom transform to rotate images into one of the 4 cardinal directions (0, 90, 180, 270 degrees)."""

        def __call__(self, img):
            # Randomly choose to rotate 0, 1, 2, or 3 times by 90 degrees
            k = random.choice([0, 1, 2, 3])
            return T.functional.rotate(img, k * 90)

    def __init__(self, config, data_dir, dataset):
        self.config = config
        self.data_dir = data_dir
        self.dataset = dataset

    def get_dataloader(self) -> DataLoader:
        """
        Applies rotation and color jittering,
        and returns the final PyTorch DataLoader.
        """

        # Override the dataset's transform attribute from the outside
        self.dataset.transform = T.Compose(
            [
                # Converts to PIL ONLY if the input is a NumPy array, otherwise passes it through
                T.Lambda(
                    lambda img: (
                        T.ToPILImage()(img) if isinstance(img, np.ndarray) else img
                    )
                ),
                # Rotate the image into one of the 4 main orientations
                DataLoaderFactory.Random90Rotation(),
                # Carefully jitter color, brightness, contrast, and saturation
                T.ColorJitter(
                    brightness=0.15, contrast=0.15, saturation=0.15, hue=0.05
                ),
                my.normalize_transform(),
            ]
        )

        # 3. Create and return the DataLoader
        return DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)


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


class Visualizer:
    def debug_colorizer(
        self,
        real_image,
        gray_image,
        color_hint,
        colorized_image,
    ):
        def prepare_image(image):
            if isinstance(image, torch.Tensor):
                image = my.denormalize(image)
                image = image.detach().cpu()

                if image.ndim == 3 and image.shape[0] in (1, 3):
                    image = image.permute(1, 2, 0)

                image = image.numpy()

            return image

        class ImageDisplay:
            @staticmethod
            def show(ax, image, title):
                image = prepare_image(image)

                # Handle single channel grayscale mapping for matplotlib
                if image.ndim == 3 and image.shape[2] == 1:
                    ax.imshow(image.squeeze(), cmap="gray")
                else:
                    ax.imshow(image)

                ax.set_title(title)
                ax.axis("off")

        image_display = ImageDisplay()

        # Pass individual axis elements from the subplots array
        _, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)

        image_display.show(axes[0], real_image, "Original")
        image_display.show(axes[1], colorized_image, "Colorized Result")
        image_display.show(axes[2], gray_image, "Grayscale")
        image_display.show(axes[3], color_hint, "Color Hint")


        plt.tight_layout()
        plt.show()

        # Full-size result
        _, ax = plt.subplots(figsize=(8, 8))
        image_display.show(ax, colorized_image, "Colorized Result")
        plt.show()


class ColorizerTrainer:
    class MultiLossTracker:
        def __init__(self, maxlen=1000):
            self.maxlen = maxlen
            self.queues = {}  # stores name -> deque

        def calculate_loss(self, loss_name, loss_value):
            if loss_name not in self.queues:
                self.queues[loss_name] = deque(maxlen=self.maxlen)

            self.queues[loss_name].append(loss_value)

            valid_losses = [x for x in self.queues[loss_name] if x is not None]
            if not valid_losses:
                return 0.0
            return sum(valid_losses) / len(valid_losses)

    def __init__(
        self,
        num_epochs,
    ):
        self.step = 0
        self.criterion = my.SRLoss(perceptual_weight=0.1)
        self.multi_loss_tracker = my.MultiLossTracker()
        data_dir = os.path.join(config.work_dir, "data")
        logger.info(f"Target data location is set to: {data_dir}")

        dataset = my.cropped_dataset(
            config.images_dir,
            crop_size=config.image_size,
            max_num_patches_per_image=1,
            keep_first_full_scale=False,
        )
        dataloader_factory = DataLoaderFactory(config, data_dir, dataset)
        self.dataloader = dataloader_factory.get_dataloader()

        self.visualizer = Visualizer()

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

    def save_checkpoint(self):
        logger.info(f"Save checkpoint: {self.checkpoint_path}")

        checkpoint = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"Checkpoint saved: {self.checkpoint_path}")

    def load_or_init(self):
        logger.info(f"\n:::Colorizer model:::\n")
        self.model = UNetColorizer().to(my.DEVICE)
        if os.path.exists(self.checkpoint_path):
            logger.info("Load model checkpoint")
            checkpoint = torch.load(self.checkpoint_path, map_location=my.DEVICE)
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            checkpoint = None

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=config.lr, weight_decay=0.01
        )

        if checkpoint:
            self.step = checkpoint["step"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = config.lr
            logger.info(f"Checkpoint loaded: {self.checkpoint_path}")
        else:
            logger.info("No checkpoint found — initialized new model and optimizer.")
            self.step = 0

        my.print_parameter_summary(self.model)


def train():

    trainer = ColorizerTrainer(
        num_epochs=100000,
    )
    trainer.load_or_init()
    trainer.train()


if __name__ == "__main__":
    logger.info(f"Batch size: {config.batch_size}")
    train()
