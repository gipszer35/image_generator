!pip install ema-pytorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import datetime
import os, sys
import matplotlib.pyplot as plt
import torchvision.transforms as T
from dataclasses import dataclass
from collections import deque
from timm.models.vision_transformer import Attention, Mlp
from torch.optim.lr_scheduler import LambdaLR
from ema_pytorch import EMA
from diffusers import AutoencoderKL
import random
from PIL import Image


def is_colab():
    return "COLAB_GPU" in os.environ


if is_colab():
    if not os.path.ismount("/content/drive"):
        from google.colab import drive

        drive.mount("/content/drive")

    ROOT_DIR = "/content/drive/MyDrive/"
    IMAGE_GENERATOR_DIR = ROOT_DIR + "ImageGenerator/"
    BATCH_SIZE = 96
else:
    ROOT_DIR = "./"
    IMAGE_GENERATOR_DIR = ROOT_DIR
    BATCH_SIZE = 2

VAE_MODEL_NAME = "stabilityai/sd-vae-ft-ema"
IMAGES_DIR = ROOT_DIR + "images"
LATENT_IMAGE_DIR = ROOT_DIR + "temp_dir_for_latents"
LATENT_SCALE = 0.18215

# DiT settings
IMAGE_SIZE = 256
LATENT_IMAGE_SIZE = 32
NUM_HEADS = 8
DIM = 512
DIT_DEPTH = 10
PATCH_SIZE = 2

DIT_CHECKPOINT_PATH = ROOT_DIR + "/DiT.vae.pt"

sys.path.append(ROOT_DIR)
import my_common as my


@dataclass(frozen=True)
class ShapeConfig:
    # Channel, Height, Width
    image: tuple = (3, IMAGE_SIZE, IMAGE_SIZE)
    latent: tuple = (4, LATENT_IMAGE_SIZE, LATENT_IMAGE_SIZE)


CONFIG_TYPE = "latent"


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_input_shape():
    shapes = {
        "image": ShapeConfig.image,
        "latent": ShapeConfig.latent,
    }
    if CONFIG_TYPE not in shapes:
        raise ValueError(f"Unknown CONFIG_TYPE: {CONFIG_TYPE}")
    return shapes[CONFIG_TYPE]


def debug_diffusion(real, x_t, x0_pred, ema_cleared, from_pure_noise, small_image):
    def normalize(x):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + 1e-8)

    _, axes = plt.subplots(1, 4, figsize=(12, 3))

    class Visualizer:
        def __init__(self, image_type="latent"):
            if image_type == "image":
                self.vae = ImageLatentManager.VAEManager(VAE_MODEL_NAME).create_vae()
            else:
                self.vae = None

        def show(self, ax, img, title):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu()

            if self.vae:
                pil_img = ImageLatentManager.latent_to_image(self.vae, img)
                ax.imshow(pil_img)
            else:
                if img.ndim == 3 and img.shape[0] == 4:
                    img = img[:3]
                img = normalize(img)

                if img.ndim == 3 and img.shape[0] == 1:
                    ax.imshow(img.squeeze(0), cmap="gray")
                else:
                    ax.imshow(img.permute(1, 2, 0))

            ax.set_title(title)
            ax.axis("off")

    visualizer = Visualizer("image")

    visualizer.show(axes[0], real, "Real")
    visualizer.show(axes[1], x_t, "Noised")
    visualizer.show(axes[2], x0_pred, "Cleaned")
    visualizer.show(axes[3], ema_cleared, "Ema Cleaned")

    plt.tight_layout()
    plt.show()

    visualizer = Visualizer("image")

    _, ax = plt.subplots()
    visualizer.show(ax, small_image, "Original small image")
    plt.show()

    visualizer = Visualizer("image")

    _, ax = plt.subplots(figsize=(8, 8))
    visualizer.show(ax, from_pure_noise, "From pure noise")
    plt.show()


def test_image_latent_manager():
    # Initialize the manager
    manager = ImageLatentManager()

    print("Step 1: Creating latents from dataset...")
    manager.save_latent_from_image_dataset()  # creates and caches latents

    print("Step 2: Creating dataloader...")
    dataloader = manager.get_dataloader()

    print("Step 3: Loading one batch of latents...")
    for _, latent_batch in enumerate(dataloader):
        # latent_batch shape: (B, 4, H, W)
        print(f"Loaded latent batch shape: {latent_batch.shape}")

        # Take the first latent in the batch
        first_latent = latent_batch[0:1]

        print("Step 4: Converting latent back to image...")
        images = manager.latents_to_images(first_latent)

        # Display the image using matplotlib
        plt.imshow(images[0])
        plt.axis("off")
        plt.title("Reconstructed Image from Latent")
        plt.show()
        break  # only process first batch for the test


class ImageLatentManager:
    """
    This class is responsible for handling image ↔ latent operations.

    Responsibilities:
    * Load images, apply basic preprocessing/transformations, and convert them
    into latent representations.
    * Save the generated latents to disk and remove the VAE from GPU memory.
    * Load saved latents later for training a DiT model.
    * Convert selected latents back into images when visualization is needed.

    This approach saves GPU memory because the VAE does not need to remain
    in GPU memory during the training process.
    """

    class VAEManager:
        def __init__(self, model_name):
            self.model_name = model_name
            self.vae = None

        def create_vae(self):
            """Create a VAE and keep it in GPU memory until program ends."""
            if self.vae is None:
                self.vae = AutoencoderKL.from_pretrained(
                    self.model_name, torch_dtype=torch.float16
                ).to(my.DEVICE)
                self.vae.eval()
            return self.vae

        def __enter__(self):
            """Context manager entry — create VAE if not already created."""
            return self.create_vae()

        def __exit__(self, exc_type, exc_value, traceback):
            """Context manager exit — delete VAE to free GPU."""
            if self.vae is not None:
                del self.vae
                self.vae = None
                torch.cuda.empty_cache()

    def __init__(self):
        self.latent_dir = LATENT_IMAGE_DIR
        self.transform = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                # Mild color change (5%)
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.003
                ),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    def cache_latents(self, dataset):
        if os.path.exists(self.latent_dir):
            print(
                f"{self.latent_dir} directory already exists. Skip to generate latent files"
            )
        else:
            os.makedirs(self.latent_dir)

            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)

            counter = 0
            with ImageLatentManager.VAEManager(VAE_MODEL_NAME) as vae:
                for images, _ in dataloader:
                    images = images.to(my.DEVICE, dtype=torch.float16)
                    with torch.inference_mode():
                        latents = vae.encode(images).latent_dist.mode() * LATENT_SCALE
                    # Save each latent separately
                    for latent in latents:
                        torch.save(
                            latent.cpu(), f"{self.latent_dir}/latent_{counter}.pt"
                        )
                        counter += 1

            print(f"Saved latents to {self.latent_dir}")

    def save_latent_from_image_dataset(self):
        dataset = my.cropped_dataset(
            IMAGES_DIR,
            crop_size=512,
            max_num_patches_per_image=2,
            transform=self.transform,
            keep_first_full_scale=True,
        )
        self.cache_latents(dataset)

    class LatentPatchDataset(Dataset):
        def __init__(self, latent_dir, crop_size=None):
            self.files = [
                os.path.join(latent_dir, f)
                for f in os.listdir(latent_dir)
                if f.endswith(".pt")
            ]
            self.crop_size = crop_size

        def __len__(self):
            return len(self.files)

        def __getitem__(self, idx):
            latent = torch.load(self.files[idx])

            if self.crop_size:
                C, H, W = latent.shape
                top = random.randint(0, H - self.crop_size)
                left = random.randint(0, W - self.crop_size)
                latent = latent[
                    :, top : top + self.crop_size, left : left + self.crop_size
                ]

            return latent

    def get_dataloader(self):
        latent_dataset = ImageLatentManager.LatentPatchDataset(
            latent_dir=self.latent_dir, crop_size=LATENT_IMAGE_SIZE
        )
        return DataLoader(latent_dataset, batch_size=BATCH_SIZE, shuffle=True)

    def latents_to_images(self, latents):
        """
        Convert a latent tensors of [4,H,W] back to images.
        Returns a PIL images.
        """
        pil_images = []
        with ImageLatentManager.VAEManager(VAE_MODEL_NAME) as vae:
            for latent in latents:
                if latent.dim() == 3:
                    latent = latent.unsqueeze(0)

                latent = latent.to(my.DEVICE, dtype=torch.float16)

                with torch.no_grad():
                    image = self.latent_to_image(vae, latent)

                pil_images.append(Image.fromarray(image))

        return pil_images

    @staticmethod
    def latent_to_image(vae, latent):
        """
        Convert a single latent tensor [4,H,W] to a PIL image.
        """
        latent = latent.unsqueeze(0)
        latent = latent.to(vae.device, dtype=torch.float16)

        with torch.no_grad():
            img = vae.decode(latent / LATENT_SCALE).sample
            img = (img / 2 + 0.5).clamp(0, 1)  # [1,C,H,W]

        img = img[0].cpu()

        # Convert to HWC and uint8 for PIL
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype("uint8")

        return Image.fromarray(img)


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(self, hidden_size, num_heads=NUM_HEADS, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(
            hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs
        )
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(
            in_features=hidden_size,
            hidden_features=mlp_hidden_dim,
            act_layer=approx_gelu,
            drop=0,
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.adaLN_modulation(c).chunk(6, dim=1)
        )
        x = x + gate_msa.unsqueeze(1) * self.attn(
            modulate(self.norm1(x), shift_msa, scale_msa)
        )
        x = x + gate_mlp.unsqueeze(1) * self.mlp(
            modulate(self.norm2(x), shift_mlp, scale_mlp)
        )
        return x


class DiffusionTransformer(nn.Module):
    def __init__(self, dim=DIM, depth=DIT_DEPTH, n_classes=10, patch=PATCH_SIZE):
        super().__init__()
        channels, H, W = get_input_shape()
        self.channels = channels
        self.dim = dim
        self.patch = patch
        self.time_embedder = my.TimestepEmbedder(dim)
        self.pos_emb = nn.Parameter(torch.randn(1, H * W // (patch * patch), dim))
        self.patch_embed = nn.Conv2d(channels, dim, patch, patch)

        # Class embedding
        self.class_emb = nn.Embedding(n_classes, dim)

        self.blocks = nn.ModuleList([DiTBlock(dim) for _ in range(depth)])
        self.unpatch = nn.ConvTranspose2d(dim, channels, patch, patch)
        self.initialize_weights()

    def initialize_weights(self):
        nn.init.xavier_uniform_(self.patch_embed.weight)
        if self.patch_embed.bias is not None:
            nn.init.zeros_(self.patch_embed.bias)

        if hasattr(self, "class_emb"):
            nn.init.normal_(self.class_emb.weight, std=0.02)

        nn.init.normal_(self.pos_emb, std=0.02)

        for block in self.blocks:
            # Attention projections
            nn.init.xavier_uniform_(block.attn.qkv.weight)
            nn.init.xavier_uniform_(block.attn.proj.weight)
            nn.init.zeros_(block.attn.proj.bias)

            # MLP
            nn.init.xavier_uniform_(block.mlp.fc1.weight)
            nn.init.xavier_uniform_(block.mlp.fc2.weight)
            nn.init.zeros_(block.mlp.fc1.bias)
            nn.init.zeros_(block.mlp.fc2.bias)

            # Very important for adaLN-Zero:
            # Initialize modulation layer to zero
            nn.init.zeros_(block.adaLN_modulation[1].weight)
            nn.init.zeros_(block.adaLN_modulation[1].bias)

        nn.init.xavier_uniform_(self.unpatch.weight)
        if self.unpatch.bias is not None:
            nn.init.zeros_(self.unpatch.bias)

    def forward(self, x_t, t, image_class=None):
        B = x_t.size(0)
        x = self.patch_embed(x_t).flatten(2).transpose(1, 2)
        x = x + self.pos_emb

        time_emb = self.time_embedder(t)
        # Add class embedding
        if image_class is not None:
            class_emb = self.class_emb(image_class)
            c = time_emb + class_emb
        else:
            c = time_emb

        for blk in self.blocks:
            x = blk(x, c)
        B, N, channels = x.shape  # N = num_tokens
        H_patch = W_patch = int(N**0.5)  # assume square layout of patches
        x = x.transpose(1, 2).reshape(B, channels, H_patch, W_patch)
        x = self.unpatch(x)
        return x


def print_basic_info(num_epochs, epoch):
    now = datetime.datetime.now()
    print("Current date and time::", now.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"Epoch [{epoch+1}/{num_epochs}]")


class DiffusionTrainer:
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
        dataloader,
        num_timesteps,
        num_epochs,
    ):

        self.dataloader = dataloader
        self.mode = CONFIG_TYPE
        self.num_epochs = num_epochs
        self.checkpoint_path = DIT_CHECKPOINT_PATH

        self.betas = my.cosine_schedule(num_timesteps).to(my.DEVICE)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar = torch.sqrt(1 - self.alpha_bar)
        self.num_timesteps = num_timesteps
        self.multi_loss_tracker = DiffusionTrainer.MultiLossTracker()
        self.load_or_init()
        self.C, self.H, self.W = get_input_shape()

    def predict_x0(self, x_t, pred_noise, t):
        a_bar = self.get_alpha_bar(t)
        pred_noise = pred_noise[0].detach().cpu()
        a_bar = a_bar[0].detach().cpu()
        # predicted clean image x̂₀
        x0_pred = (x_t - torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a_bar)
        return x0_pred

    @staticmethod
    def lr_lambda(current_step: int):
        warmup_steps = -1
        if current_step < warmup_steps:
            return 0.01 + 0.99 * (current_step / warmup_steps)  # linear warmup
        return 1.0

    def print_info(self, epoch, loss, ema_loss):
        avg_loss = self.multi_loss_tracker.calculate_loss("loss", loss)
        avg_ema_loss = self.multi_loss_tracker.calculate_loss("ema_loss", ema_loss)

        print_basic_info(self.num_epochs, epoch)
        print(f"Avg student loss: {avg_loss:.4f}")
        print(f"Avg master (Ema) loss: {avg_ema_loss:.4f}")

    def show_training_state(
        self, loss, ema_loss, real, x_t, pred_noise, ema_pred_noise, epoch, t
    ):

        self.print_info(epoch, loss, ema_loss)
        print("t:", t[0].item(), "step:", self.step)
        # if self.step % 3 == 0:
        #     self.save_checkpoint()
        real = real[0].detach().cpu()
        x_t = x_t[0].detach().cpu()

        x0_pred = self.predict_x0(x_t, pred_noise, t)
        ema_x0_pred = self.predict_x0(x_t, ema_pred_noise, t)
        print("start generating from pure noise", datetime.datetime.now())
        from_pure_noise, small_image = self.generate_big_image_from_pure_noise()
        print("finished generating from pure noise", datetime.datetime.now())

        debug_diffusion(real, x_t, x0_pred, ema_x0_pred, from_pure_noise, small_image)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t].view(-1, 1, 1, 1)

    def denoise_step(self, B, x_t, t):
        t_tensor = torch.full((B,), t, device=my.DEVICE, dtype=torch.long)

        # predict noise
        pred_noise = self.ema(x_t, t_tensor, image_class=None)

        alpha_t = self.alphas[t]
        alpha_bar_t = self.get_alpha_bar(t)
        beta_t = self.betas[t]

        # DDPM posterior mean
        mean = (
            1
            / torch.sqrt(alpha_t)
            * (x_t - (beta_t / torch.sqrt(1 - alpha_bar_t)) * pred_noise)
        )

        if t > 0:
            noise = torch.randn_like(x_t)
            sigma = torch.sqrt(beta_t)
            x_t = mean + sigma * noise
        else:
            x_t = mean
        return x_t

    def generate_small_image_from_pure_noise(self, image_class=None):

        was_training = self.ema.training
        self.ema.eval()

        B = 1
        C, H, W = get_input_shape()
        x_t = torch.randn(B, C, H, W, device=my.DEVICE)

        with torch.inference_mode():
            for t in reversed(range(self.num_timesteps)):
                x_t = self.denoise_step(B, x_t, t)

        if was_training:
            self.ema.train()

        return x_t.squeeze().cpu()

    def process_big_image(self, x_t, B, t):
        _, _, Ht, Wt = x_t.shape
        patch_size = self.H
        stride = self.H
        h_rand_start = 0
        w_rand_start = 0

        for i in range(h_rand_start, Ht - patch_size + 1, stride):
            for j in range(w_rand_start, Wt - patch_size + 1, stride):
                hs = slice(i, i + patch_size)
                ws = slice(j, j + patch_size)
                tile = x_t[:, :, hs, ws]
                x_t[:, :, hs, ws] = self.denoise_step(B, tile, t)

        return x_t

    def process_big_image2(self, x_t, B, t):
        _, _, Ht, Wt = x_t.shape
        patch_size = self.H
        stride = self.H
        h_rand_start = 0
        w_rand_start = 0

        for i in range(stride // 2, Ht - patch_size + 1, stride):
            for j in range(w_rand_start, Wt - patch_size + 1, stride):
                hs = slice(i, i + patch_size)
                ws = slice(j, j + patch_size)
                tile = x_t[:, :, hs, ws]
                x_t[:, :, hs, ws] = self.denoise_step(B, tile, t)

        for i in range(h_rand_start, Ht - patch_size + 1, stride):
            for j in range(stride // 2 , Wt - patch_size + 1, stride):
                hs = slice(i, i + patch_size)
                ws = slice(j, j + patch_size)
                tile = x_t[:, :, hs, ws]
                x_t[:, :, hs, ws] = self.denoise_step(B, tile, t)

        return x_t

    def change_every_fourth_tile(self, x_t, x_t_small, alpha):
        patch_size = 1

        patches_y = self.H // patch_size
        patches_x = self.W // patch_size

        for i in range(patches_y):
            for j in range(patches_x):

                # extract patch from small image
                h0 = i * patch_size
                w0 = j * patch_size

                patch = x_t_small[:, :, h0 : h0 + patch_size, w0 : w0 + patch_size]

                # staggered horizontal shift per row
                big_h0 = i * patch_size
                big_w0 = j * 2 * patch_size

                # bounds check
                if big_w0 + patch_size > x_t.shape[3]:
                    continue
                h_slice = slice(big_h0, big_h0 + patch_size)
                w_slice = slice(big_w0, big_w0 + patch_size)

                region = x_t[:, :, h_slice, w_slice]

                x_t[:, :, h_slice, w_slice] = (1 - alpha) * region + alpha * patch

        return x_t

    def inject_global_upscaled(self, x_t, x_t_small, alpha):
        _, _, H_big, W_big = x_t.shape

        # upscale small image to big size
        x_up = torch.nn.functional.interpolate(
            x_t_small, size=(H_big, W_big), mode="bilinear", align_corners=False
        )

        # blend into big latent
        x_t = (1 - alpha) * x_t + alpha * x_up

        return x_t

    def interpolate(self, x_t, x_t_small):
        _, _, H, W = x_t.shape

        # Upscale small → full size
        x_up = torch.nn.functional.interpolate(
            x_t_small, size=(H, W), mode="nearest"
        )

        return x_up


    def inject_side_panels(self, x_t, x_t_small, alpha):
        """
        Places:
        - left half of x_t_small -> top-left of x_t
        - right half of x_t_small -> top-right of x_t
        - middle region stays unchanged
        """

        B, C, H, W = x_t.shape
        _, _, h, w = x_t_small.shape

        # x_t is 2x resolution
        assert H == 2 * h and W == 2 * w

        mid_w = W // 2
        small_mid_w = w // 2

        x_new = x_t.clone()

        # --- split small image ---
        left = x_t_small[:, :, :, :small_mid_w]
        right = x_t_small[:, :, :, small_mid_w:]

        # --- inject left into top-left ---
        x_new[:, :, :h, :small_mid_w] = (
            x_t[:, :, :h, :small_mid_w] * (1 - alpha) + left * alpha
        )

        # --- inject right into top-right ---
        x_new[:, :, :h, mid_w + small_mid_w :] = (
            x_t[:, :, :h, mid_w + small_mid_w :] * (1 - alpha) + right * alpha
        )

        return x_new


    def inject_corner_panels(self, x_t, x_t_small, alpha):
        """
        Places x_t_small quadrants into x_t corners:
        middle remains unchanged.
        """
        B, C, H, W = x_t.shape
        _, _, h, w = x_t_small.shape

        assert H == 2 * h and W == 2 * w

        mid_h, mid_w = H // 2, W // 2
        small_mid_h, small_mid_w = h // 2, w // 2

        x_new = x_t.clone()

        # --- split into quadrants ---
        tl = x_t_small[:, :, :small_mid_h, :small_mid_w]
        tr = x_t_small[:, :, :small_mid_h, small_mid_w:]
        bl = x_t_small[:, :, small_mid_h:, :small_mid_w]
        br = x_t_small[:, :, small_mid_h:, small_mid_w:]

        def blend_and_inject(dst_slice, src):
            x_new[:, :, dst_slice[0], dst_slice[1]] = (
                x_t[:, :, dst_slice[0], dst_slice[1]] * (1 - alpha) + src * alpha
            )

        # --- apply 4 corners ---
        blend_and_inject((slice(small_mid_h, h), slice(small_mid_w, w)), tl)
        blend_and_inject((slice(small_mid_h, h), slice(w, w + small_mid_w )), tr)
        blend_and_inject((slice(h, h + small_mid_h), slice(small_mid_w, w)), bl)
        blend_and_inject((slice(h, h + small_mid_h), slice(w, w + small_mid_w)), br)

        return x_new

    def generate_big_image_from_pure_noise(self):
        was_training = self.ema.training
        self.ema.eval()

        B = 1
        img_size = 2

        x_t = torch.randn(
            B, self.C, self.H * img_size, self.W * img_size, device=my.DEVICE
        )
        x_t_small = torch.randn(B, self.C, self.H, self.W, device=my.DEVICE)

        with torch.inference_mode():
            for t in reversed(range(self.num_timesteps)):
                x_t_small = self.denoise_step(B, x_t_small, t)

        x_t = self.interpolate(x_t, x_t_small)

        # Improve image details by renoise and denosie steps
        for i in [ 3, 8, 10, 15, 21, 30, 40, 50]:
            timesteps = self.num_timesteps//i
            a_bar = self.get_alpha_bar(timesteps)

            noise = torch.randn_like(x_t)
            x_t = torch.sqrt(a_bar) * x_t + torch.sqrt(1 - a_bar) * noise

            for t in reversed(range(timesteps)):
                x_t = self.process_big_image2(x_t, B, t)

            for t in reversed(range(timesteps)):
                x_t = self.process_big_image(x_t, B, t)


        if was_training:
            self.ema.train()

        return x_t.squeeze().cpu(), x_t_small.squeeze().cpu()

    def train_step(self, real, epoch):
        x0 = real.to(my.DEVICE)

        t = torch.randint(0, self.num_timesteps, (x0.size(0),), device=my.DEVICE)

        noise = torch.randn_like(x0)
        a_bar = self.get_alpha_bar(t)

        x_t = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * noise
        pred_noise = self.model(x_t, t)
        loss = ((pred_noise - noise) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.ema.update()

        if self.step % 3 == 0:
            with torch.no_grad():  # Saves memory!
                ema_pred_noise = self.ema.ema_model(x_t, t)
                ema_loss = ((ema_pred_noise - noise) ** 2).mean()
                self.show_training_state(
                    loss,
                    ema_loss,
                    real,
                    x_t,
                    pred_noise,
                    ema_pred_noise,
                    epoch,
                    t,
                )

        self.step += 1
        return

    def train(self):
        for epoch in range(self.num_epochs):
            for real in self.dataloader:
                self.train_step(real, epoch)

    def save_checkpoint(self):
        checkpoint = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "ema_state": self.ema.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        print(f"Checkpoint saved: {self.checkpoint_path}")

    def load_or_init(self):
        print(f"\n::: DIT model:::\n")
        self.model = DiffusionTransformer().to(my.DEVICE)
        if os.path.exists(self.checkpoint_path):
            print("Load model checkpoint")
            checkpoint = torch.load(self.checkpoint_path, map_location=my.DEVICE)
            # model loaded first since the loaded model should be added to ema
            self.model.load_state_dict(checkpoint["model_state"])
        else:
            checkpoint = None

        self.ema = EMA(
            self.model,
            beta=0.9999,
            update_after_step=100,
            update_every=1,
        )

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=0.00001, weight_decay=0.01
        )
        self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)

        if checkpoint:
            self.step = checkpoint["step"]
            self.ema.load_state_dict(checkpoint["ema_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            print(f"Checkpoint loaded: {self.checkpoint_path}")
        else:
            print("No checkpoint found — initialized new model and optimizer.")
            self.step = 0

        my.print_parameter_summary(self.model)


def train():
    dataloader = ImageLatentManager().get_dataloader()

    trainer = DiffusionTrainer(
        dataloader=dataloader,
        num_timesteps=600,
        num_epochs=100000,
    )
    trainer.train()


if __name__ == "__main__":
    print("Batch size:", BATCH_SIZE)
    torch.cuda.empty_cache()

    # test_image_latent_manager()

    train()
