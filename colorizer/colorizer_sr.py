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
    paintings_dir: str
    work_dir: str
    content_drive: str

    image_size: int = 512
    lr: float = 1e-4

    @property
    def checkpoint_path(self):
        return os.path.join(self.work_dir, "colorizer_sr.pt")


def create_config() -> Config:
    content_drive: str = "/content/drive"
    if is_colab():
        root_dir = os.path.join(content_drive, "MyDrive")
        work_dir = os.path.join(root_dir, "ImageGenerator", "colorizer")
        images_dir = os.path.join(root_dir, "images")
        paintings_dir = os.path.join(root_dir, "paintings")

        batch_size = 16
    else:
        root_dir = "../"
        work_dir = "./"
        batch_size = 2
        images_dir = "../images/my-images/"
        paintings_dir = "../images/Impressionism/"

    return Config(
        root_dir=root_dir,
        work_dir=work_dir,
        batch_size=batch_size,
        images_dir=images_dir,
        paintings_dir=paintings_dir,
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


class CategoryEmbedding(nn.Module):
    def __init__(self, num_categories, channels, size=8):
        super().__init__()
        self.embedding = nn.Embedding(num_categories, channels)
        self.size = size

    def forward(self, category):
        x = self.embedding(category)
        return x[:, :, None, None].expand(-1, -1, self.size, self.size)


class UNetColorizerSR(nn.Module):
    """
    U-Net architecture for colorizing 256x256 images using 64x64 spatial
    color hints, while simultaneously generating a 512x512 colorized output.
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
        c0 = 64
        c1 = 128
        c2 = 128
        c3 = 256
        c4 = 512
        bn = 1024  # bottleneck channels size
        color_channels = 64
        category_channel = 64

        # Extended channel configurations for the downsampling path
        encoder_channels = [1, c0, c1, c2, c3, c4, bn]

        # Multi-stage feature extraction and downsampling modules
        self.encoders = nn.ModuleList(
            [
                UNetColorizerSR.ConvBlock(encoder_channels[i], encoder_channels[i + 1])
                for i in range(len(encoder_channels) - 1)
            ]
        )

        # Downsample between encoder stages
        self.pools = nn.ModuleList(
            [nn.MaxPool2d(2) for _ in range(len(encoder_channels) - 2)]
        )

        self.color_encoder_64 = self._create_color_encoder(color_channels, downsample=0)
        self.color_encoder_8 = self._create_color_encoder(color_channels, downsample=3)

        # Bottleneck block combining encoder features with color and category conditioning
        self.bottleneck = UNetColorizerSR.ConvBlock(
            bn + color_channels + category_channel, bn
        )

        # Extended channel configurations for the upsampling path
        decoder_channels = [c4, c3, c2, c1, c0, c0]

        decoder_input_channels = [bn, c4, c3, c2 + color_channels, c1, c0]

        self.up_samplers = nn.ModuleList(
            [
                nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
                for in_ch, out_ch in zip(decoder_input_channels, decoder_channels)
            ]
        )

        skip_channels = list(reversed(encoder_channels[:-1]))

        self.decoders = nn.ModuleList(
            [
                UNetColorizerSR.ConvBlock(up_ch + skip_ch, out_ch)
                for up_ch, skip_ch, out_ch in zip(
                    decoder_channels,
                    skip_channels,
                    decoder_channels,
                )
            ]
        )
        self.category_embedding = CategoryEmbedding(2, category_channel)

        self.final_upsample = nn.ConvTranspose2d(
            c0, c0, kernel_size=2, stride=2
        )
        # Final projection layer to map features back to standard RGB channels
        self.final_conv = nn.Conv2d(c0, 3, kernel_size=1)

    def _create_color_encoder(self, color_channels, downsample=0):
        # downsample controls the number of 2× spatial reductions.
        # For example, downsample=3 reduces 64×64 → 32×32 → 16×16 → 8×8.
        layers = [
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, color_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        ]

        for _ in range(downsample):
            layers.append(nn.MaxPool2d(2))

        return nn.Sequential(*layers)

    def forward(self, grayscale, color_hint, category):
        # Feature extraction loop storing intermediate maps for skip connections
        skip_connections = []
        x = grayscale

        for i, enc in enumerate(self.encoders):
            s = enc(x)

            if i < len(self.encoders) - 1:
                skip_connections.append(s)
                x = self.pools[i](s)
            else:
                x = s

        # Process external color guidance at 8×8
        hint_features = self.color_encoder_8(color_hint)

        # Category embedding
        category_features = self.category_embedding(category)

        # Combine encoder, color, and category features
        bottleneck_input = torch.cat((x, hint_features, category_features), dim=1)
        x = self.bottleneck(bottleneck_input)

        # Color features at 64×64
        color_64 = self.color_encoder_64(color_hint)

        # Reconstruction loop combining upsampled features with corresponding skip maps
        for i, (up, dec, s) in enumerate(
            zip(
                self.up_samplers,
                self.decoders,
                reversed(skip_connections),
            )
        ):
            # Add 64×64 color features before the corresponding upsampling
            if i == 3:
                x = torch.cat((x, color_64), dim=1)

            x = up(x)
            x = torch.cat((x, s), dim=1)
            x = dec(x)

        x = self.final_upsample(x)
        # Output bounded between [-1, 1] via hyperbolic tangent activation
        return torch.tanh(self.final_conv(x))


class CategoryDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, category):
        self.dataset = dataset
        self.category = category

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, _ = self.dataset[idx]
        return image, self.category


class ColorizerSRTrainer(cc.ColorizerTrainerBase):

    def __init__(self, model, num_epochs):
        super().__init__(model, config.lr)
        self.step = 0
        self.criterion = my.SRLoss(perceptual_weight=0.1)
        self.multi_loss_tracker = my.MultiLossTracker()
        data_dir = os.path.join(config.work_dir, "data")
        logger.info(f"Target data location is set to: {data_dir}")

        dataset_images = my.cropped_dataset(
            config.images_dir,
            crop_size=config.image_size,
            max_num_patches_per_image=1,
            keep_first_full_scale=False,
        )

        dataset_paintings = my.cropped_dataset(
            config.paintings_dir,
            crop_size=config.image_size,
            max_num_patches_per_image=1,
            keep_first_full_scale=False,
        )

        dataset = torch.utils.data.ConcatDataset(
            [
                CategoryDataset(dataset_images, 0),  # image
                CategoryDataset(dataset_paintings, 1),  # painting
            ]
        )

        dataloader_factory = cc.DataLoaderFactory(config, data_dir, dataset)
        self.dataloader = dataloader_factory.get_dataloader()

        self.visualizer = cc.Visualizer()

        self.num_epochs = num_epochs
        self.checkpoint_path = config.checkpoint_path

    def train_step(self, real, category, epoch):
        real = real.to(my.DEVICE)
        category = category.to(my.DEVICE)

        # Extract grayscale target and generate 8x8 low-res color hint
        gray = T.functional.rgb_to_grayscale(real, num_output_channels=1)
        # Downsample grayscale input to 256×256
        gray = F.interpolate(gray, size=(256, 256), mode="area")

        # Generate 64×64 color hint for the multi-scale color encoders
        hint = F.interpolate(real, size=(64, 64), mode="area")

        # Predict the full RGB reconstruction using our hint-guided model
        pred_rgb = self.model(gray, hint, category)

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
                pred_rgb = self.model(gray, hint, category)


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
            for real, category in self.dataloader:
                self.train_step(real, category, epoch)


def train():
    model = UNetColorizerSR().to(my.DEVICE)
    trainer = ColorizerSRTrainer(model, num_epochs=100000)
    trainer.load_or_init()
    trainer.train()


if __name__ == "__main__":
    logger.info(f"Batch size: {config.batch_size}")
    train()
