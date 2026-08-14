import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
import os, sys
import matplotlib.pyplot as plt
import torchvision.transforms as T
from dataclasses import dataclass
from collections import deque


def is_colab():
    return "COLAB_GPU" in os.environ


@dataclass
class Config:
    root_dir: str
    batch_size: int
    images_dir: str
    work_dir: str
    content_drive: str


    image_size: int = 256

    @property
    def checkpoint_path(self):
        return os.path.join(self.work_dir, "colorizer.pt")


def create_config() -> Config:
    content_drive: str = "/content/drive"
    if is_colab():
        root_dir = os.path.join(content_drive, "MyDrive")
        work_dir = os.path.join(root_dir, "ImageGenerator", "colorizer")
        images_dir = os.path.join(root_dir, "images")
        batch_size = 96
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
        content_drive=content_drive
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


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DiffusionTransformer(nn.Module):
    """
    U-Net based architecture that takes a grayscale image, preserves structural details
    via skip connections, and merges them with spatial color hints from an smaller RGB
    image at the bottleneck.
    """

    def __init__(self):
        super().__init__()

        # Grayscale Encoder
        self.enc1 = ConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)  # Out: 16x16

        self.enc2 = ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)  # Out: 8x8

        self.enc3 = ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)  # Out: 4x4

        # Color Hint Encoder (8x8 RGB -> 4x4)
        self.color_encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # Out: 4x4
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Bottleneck (Concat: 256 gray features + 64 color features = 320 channels)
        self.bottleneck = ConvBlock(256 + 64, 512)

        # Decoder + Skip Connections
        self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = ConvBlock(256 + 256, 256)

        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128 + 128, 128)

        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64 + 64, 64)

        self.final_conv = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, grayscale, color_hint):
        # Encoder forward pass and saving skip connections
        s1 = self.enc1(grayscale)
        p1 = self.pool1(s1)

        s2 = self.enc2(p1)
        p2 = self.pool2(s2)

        s3 = self.enc3(p2)
        p3 = self.pool3(s3)

        # Process color hint
        hint_features = self.color_encoder(color_hint)

        # Concatenate features along the channel dimension at the bottleneck
        bottleneck_input = torch.cat((p3, hint_features), dim=1)
        b = self.bottleneck(bottleneck_input)

        # Decoder forward pass using skip connections
        d3 = self.up3(b)
        d3 = torch.cat((d3, s3), dim=1)
        d3 = self.dec3(d3)

        d2 = self.up2(d3)
        d2 = torch.cat((d2, s2), dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat((d1, s1), dim=1)
        d1 = self.dec1(d1)

        return torch.tanh(self.final_conv(d1))


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
        _, axes = plt.subplots(1, 3, figsize=(9, 3))
        image_display.show(axes[0], real_image, "Original")
        image_display.show(axes[1], gray_image, "Grayscale")
        image_display.show(axes[2], color_hint, "Color Hint")

        plt.tight_layout()
        plt.show()

        # Full-size result
        _, ax = plt.subplots()
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
        data_dir = os.path.join(config.work_dir, "data")
        dataset = my.cifar100_dataset(data_dir)

        self.dataloader = DataLoader(
            dataset, batch_size=config.batch_size, shuffle=True
        )
        self.visualizer = Visualizer()

        self.num_epochs = num_epochs
        self.checkpoint_path = config.checkpoint_path

    def train_step(self, real, epoch):
        real = real.to(my.DEVICE)

        # Extract grayscale target and generate 8x8 low-res color hint
        gray = T.functional.rgb_to_grayscale(real, num_output_channels=1)
        hint = F.interpolate(real, size=(8, 8), mode="bilinear", align_corners=False)

        # Predict the full RGB reconstruction using our hint-guided model
        pred_rgb = self.model(gray, hint)
        # Calculate reconstruction mean squared error
        loss = ((pred_rgb - real) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.step % 100 == 0:
            self.save_checkpoint()
            current_lr = self.optimizer.param_groups[0]["lr"]
            logger.info(
                f"Epoch: {epoch + 1}/{self.num_epochs}\n"
                f"Step: {self.step:,}\n"
                f"Loss: {loss.item():.6f}\n"
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
        self.model = DiffusionTransformer().to(my.DEVICE)
        if os.path.exists(self.checkpoint_path):
            logger.info("Load model checkpoint")
            checkpoint = torch.load(self.checkpoint_path, map_location=my.DEVICE)
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            checkpoint = None

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.00001, weight_decay=0.01
        )

        if checkpoint:
            self.step = checkpoint["step"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
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