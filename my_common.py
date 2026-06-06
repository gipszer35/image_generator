import os
import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import torch.nn as nn
from PIL import Image
import math
from torchvision import datasets, transforms
import hashlib
import torchvision.transforms.functional as TF
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
import torch.nn.functional as NNF

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

NORMALIZE_TRANSFORM = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

def create_logger():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # always reset in notebooks (Colab/IPython safe)
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    handler.setFormatter(formatter)

    logger.addHandler(handler)
    logger.propagate = False  # prevents duplicate root logs

    return logger

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Cosine schedule as proposed in https://arxiv.org
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 0.0001, 0.9999)

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    def timestep_embedding(self, t, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                        These may be fractional.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = self.frequency_embedding_size // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half, dtype=torch.float32)
            / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if self.frequency_embedding_size % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t)
        t_emb = self.mlp(t_freq)
        return t_emb


class CroppedImageDataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]

        # Convert to PIL images if needed
        if isinstance(img, np.ndarray):
            if img.dtype != np.uint8:
                img = (img * 255).clip(0, 255).astype(np.uint8)
            img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        return img, -1


class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features
        self.slice = nn.Sequential(*list(vgg[:16])).eval()

        for p in self.slice.parameters():
            p.requires_grad = False

        # ImageNet normalization (required!)
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def forward(self, sr, hr):
        sr = self.normalize(sr)
        hr = self.normalize(hr)

        # Extract VGG features
        sr_f = self.slice(sr)
        hr_f = self.slice(hr)

        # L1 feature loss
        return NNF.l1_loss(sr_f, hr_f)


class SRLoss(nn.Module):
    def __init__(self, perceptual_weight=0.1):
        super().__init__()
        self.pixel_loss = nn.L1Loss()
        self.percep_loss = VGGPerceptualLoss()
        self.percep_loss.to(DEVICE)
        self.weight = perceptual_weight

    def forward(self, sr, hr):
        pixel = self.pixel_loss(sr, hr)
        percep = self.percep_loss(sr, hr)
        return pixel + self.weight * percep


def save_checkpoint(model, optimizer, path):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


def load_checkpoint(model, optimizer, path):
    checkpoint = torch.load(path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from: {path}")

    if optimizer and "optimizer_state_dict" in checkpoint:
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("Loaded optimizer state.")
        except Exception as e:
            print(f"Optimizer state not loaded: {e}")


def load_checkpoint_if_exists(model, optimizer, path):

    if os.path.exists(path):
        print("Load checkpoint")
        load_checkpoint(model, optimizer, path)
    else:
        print("No checkpoint found — initialized new model and optimizer.")
    return model, optimizer


def test_function():
    print("test works")


def load_images_cropped(directory_path, crop_size, max_num_patches_per_image, keep_first_full_scale):
    image_paths = [
        os.path.join(directory_path, fname)
        for fname in os.listdir(directory_path)
        if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
    ]

    cropped_images = []

    for path in image_paths:
        try:
            img = Image.open(path).convert("RGB")
            w, h = img.size

            seen = set()
            attempts = 0
            max_attempts = max_num_patches_per_image * 2

            while len(seen) < max_num_patches_per_image and attempts < max_attempts:
                attempts += 1
                min_side = min(w, h)

                if min_side > crop_size:
                    # First image: keep original size (no scaling)
                    if keep_first_full_scale and not seen:
                        scale = 1
                    else:
                        # Random scale that still allows a valid crop
                        scale = torch.empty(1).uniform_(crop_size / min_side, 1.0).item()
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized = img.resize((new_w, new_h), Image.BICUBIC)
                else:
                    resized = img
                    new_w, new_h = resized.size

                if new_w < crop_size or new_h < crop_size:
                    continue

                img_tensor = TF.to_tensor(resized)

                top = torch.randint(0, new_h - crop_size + 1, (1,)).item()
                left = torch.randint(0, new_w - crop_size + 1, (1,)).item()

                # Avoid duplicate crops
                crop = img_tensor[:, top : top + crop_size, left : left + crop_size]
                crop_np = crop.permute(1, 2, 0).numpy()
                hsh = hashlib.md5(crop_np.tobytes()).hexdigest()
                if hsh in seen:
                    continue
                seen.add(hsh)

                cropped_images.append(crop_np)

        except Exception as e:
            print(f"Failed to process image {path}: {e}")

    if cropped_images:
        return np.stack(cropped_images)
    else:
        return np.empty((0, crop_size, crop_size, 3), dtype=np.float32)


def cifar100_dataset():
    cifar_dataset = datasets.CIFAR100(
        root="./data", download=True, transform=NORMALIZE_TRANSFORM
    )
    return cifar_dataset


def cifar10_dataset():
    cifar_dataset = datasets.CIFAR10(
        root="./data", download=True, transform=NORMALIZE_TRANSFORM
    )
    return cifar_dataset


def cropped_dataset(
    image_dir, crop_size, max_num_patches_per_image=100, transform=NORMALIZE_TRANSFORM, keep_first_full_scale=False
):
    cropped = load_images_cropped(image_dir, crop_size, max_num_patches_per_image, keep_first_full_scale)
    cropped_dataset = CroppedImageDataset(cropped, transform=transform)
    return cropped_dataset


def mixed_dataloader(datasets, batch_size):
    mixed_dataset = ConcatDataset(datasets)
    return DataLoader(mixed_dataset, batch_size=batch_size, shuffle=True, num_workers=4)


def show_image_eval(image_type, images, loss):
    plt.figure(figsize=(1.5, 1.5))

    loss = loss.detach()
    if image_type == "real":
        p = math.exp(-loss)
    elif image_type == "fake":
        p = 1 - math.exp(-loss)
    img = images[0]

    if img.ndim == 3 and img.shape[0] in [1, 3]:
        img = img.permute(1, 2, 0)  # -> (H, W, C)

    # Scale [-1,1] → [0,1]
    img = (img + 1) / 2

    if img.shape[-1] == 1:
        img = img.squeeze(-1)
        cmap = "gray"
    else:
        cmap = None

    img = img.detach().numpy()
    plt.imshow(img, cmap=cmap)
    plt.title(f"p={p:.2f}: loss={loss:.3f}")
    plt.axis("off")

    plt.show()


def display_images(generated_images, dpi=100):
    dim = (2, 2)
    num_images = min(len(generated_images), dim[0] * dim[1])

    plt.figure(figsize=(2, 2), dpi=dpi)

    for i in range(num_images):
        plt.subplot(dim[0], dim[1], i + 1)

        img = generated_images[i].permute(1, 2, 0).detach().cpu()
        img = (img + 1) / 2
        img = img.clamp(0, 1).numpy()

        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def print_parameter_summary(model):
    print(f"{'Parameter':40s} {'Shape':20s} {'# Params'}")
    print("-" * 70)
    total = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        print(f"{name:40s} {str(list(p.shape)):20s} {n}")
    print("-" * 70)
    print("Total parameters:", total)


class GaussianNoise(nn.Module):
    def __init__(self, sigma=0.1):
        super().__init__()
        self.sigma = sigma

    def forward(self, x):
        if self.training and self.sigma > 0:
            noise = torch.randn_like(x)
            return x * (1 - self.sigma) + noise * self.sigma
        return x


def normalize(image):
    """
    Normalizes a tensor image from range [0, 1] to [-1, 1]
    """
    return (image - 0.5) / 0.5


def denormalize(tensor):
    """
    Denormalizes a tensor image from range [-1, 1] to [0, 1]
    for proper display with matplotlib.
    """
    # Inverse operation of Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tensor = tensor * 0.5 + 0.5
    tensor = torch.clamp(tensor, 0, 1)  # Clamp values just in case
    return tensor
