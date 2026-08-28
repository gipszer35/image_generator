import torch
import numpy as np
import random
from collections import deque
import torchvision.transforms as T
import matplotlib.pyplot as plt
import os
from torch.utils.data import DataLoader
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

        # Create and return the DataLoader
        return DataLoader(self.dataset, batch_size=self.config.batch_size, shuffle=True)


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


class ColorizerTrainerBase:

    def __init__(self, model, lr):
        self.model = model
        self.lr = lr

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
        if os.path.exists(self.checkpoint_path):
            logger.info("Load model checkpoint")
            checkpoint = torch.load(self.checkpoint_path, map_location=my.DEVICE)
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            checkpoint = None

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=0.01
        )

        if checkpoint:
            self.step = checkpoint["step"]
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.lr
            logger.info(f"Checkpoint loaded: {self.checkpoint_path}")
        else:
            logger.info("No checkpoint found — initialized new model and optimizer.")
            self.step = 0

        my.print_parameter_summary(self.model)
