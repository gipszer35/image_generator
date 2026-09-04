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
from timm.models.vision_transformer import Attention, Mlp
from torch.optim.lr_scheduler import LambdaLR
from ema_pytorch import EMA
from diffusers import AutoencoderKL
import random
from PIL import Image


def is_colab():
    return "COLAB_GPU" in os.environ


@dataclass
class Config:
    root_dir: str
    work_dir: str
    batch_size: int
    images_dir: str
    content_drive: str

    data_representation: str = "latent"
    vae_model_name: str = "stabilityai/sd-vae-ft-ema"
    image_size: int = 256
    latent_image_size: int = 32
    crop_size: int = 512
    latent_scale: float = 0.18215
    num_heads: int = 8
    dim: int = 512
    dit_depth: int = 10
    lr: float = 1e-5

    @property
    def latent_image_dir(self):
        return os.path.join(self.work_dir, "temp_dir_for_latents")

    @property
    def dit_checkpoint_path(self):
        return os.path.join(self.work_dir, "DiT.vae.pt")


def create_config() -> Config:
    content_drive = "/content/drive"
    if is_colab():
        root_dir = os.path.join(content_drive, "MyDrive")
        work_dir = os.path.join(root_dir, "ImageGenerator", "dit")
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


@dataclass(frozen=True)
class ShapeConfig:
    # Channel, Height, Width
    image: tuple = (3, config.image_size, config.image_size)
    latent: tuple = (4, config.latent_image_size, config.latent_image_size)


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_input_shape():
    shapes = {
        "image": ShapeConfig.image,
        "latent": ShapeConfig.latent,
    }
    if config.data_representation not in shapes:
        raise ValueError(
            f"Unknown data representation type: {config.data_representation}"
        )
    return shapes[config.data_representation]


def debug_diffusion(real, x_t, x0_pred, ema_cleared, from_pure_noise):
    def normalize(x):
        x_min = x.min()
        x_max = x.max()
        return (x - x_min) / (x_max - x_min + 1e-8)

    _, axes = plt.subplots(1, 4, figsize=(24, 6), dpi=150)

    class Visualizer:
        def __init__(self):
            self.vae = ImageLatentManager.VAEManager(config.vae_model_name).create_vae()

        def show(self, ax, img, title):
            if isinstance(img, torch.Tensor):
                img = img.detach().cpu()
            pil_img = ImageLatentManager.latent_to_image(self.vae, img)
            ax.imshow(pil_img)

            ax.set_title(title)
            ax.axis("off")

    visualizer = Visualizer()

    visualizer.show(axes[0], real, "Real")
    visualizer.show(axes[1], x_t, "Noised")
    visualizer.show(axes[2], x0_pred, "Cleaned")
    visualizer.show(axes[3], ema_cleared, "Ema Cleaned")

    plt.tight_layout()
    plt.show()

    # Full-size result
    _, ax = plt.subplots(figsize=(8, 8))
    visualizer.show(ax, from_pure_noise, "From pure noise")
    plt.show()


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
        self.latent_dir = config.latent_image_dir
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

    def cache_latents_from_image_dataset(self):
        if os.path.exists(self.latent_dir):
            logger.info(
                f"{self.latent_dir} directory already exists. Skip to generate latent files"
            )
        else:
            logger.info("Creating latents from dataset...")
            dataset = my.cropped_dataset(
                config.images_dir,
                crop_size=config.crop_size,
                max_num_patches_per_image=1,
                transform=self.transform,
            )
            os.makedirs(self.latent_dir)
            dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
            counter = 0
            with ImageLatentManager.VAEManager(config.vae_model_name) as vae:
                for images, _ in dataloader:
                    images = images.to(my.DEVICE, dtype=torch.float16)
                    with torch.inference_mode():
                        latents = (
                            vae.encode(images).latent_dist.mode() * config.latent_scale
                        )
                    # Save each latent separately
                    for latent in latents:
                        torch.save(
                            latent.cpu(), f"{self.latent_dir}/latent_{counter}.pt"
                        )
                        counter += 1

            logger.info(f"Saved latents to {self.latent_dir}")

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
        self.cache_latents_from_image_dataset()
        latent_dataset = ImageLatentManager.LatentPatchDataset(
            latent_dir=self.latent_dir, crop_size=config.latent_image_size
        )
        return DataLoader(latent_dataset, batch_size=config.batch_size, shuffle=True)

    def latents_to_images(self, latents):
        """
        Convert a latent tensors of [4,H,W] back to images.
        Returns a PIL images.
        """
        pil_images = []
        with ImageLatentManager.VAEManager(config.vae_model_name) as vae:
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
            img = vae.decode(latent / config.latent_scale).sample
            img = (img / 2 + 0.5).clamp(0, 1)  # [1,C,H,W]

        img = img[0].cpu()

        # Convert to HWC and uint8 for PIL
        img = img.permute(1, 2, 0).numpy()
        img = (img * 255).astype("uint8")

        return Image.fromarray(img)


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


class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """

    def __init__(
        self, hidden_size, num_heads=config.num_heads, mlp_ratio=4.0, **block_kwargs
    ):
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
    def __init__(self, dim=config.dim, depth=config.dit_depth, n_classes=10, patch=2):
        super().__init__()
        channels, H, W = get_input_shape()
        self.channels = channels
        self.dim = dim
        self.patch = patch
        self.time_embedder = TimestepEmbedder(dim)
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


class DiffusionTrainer:

    def __init__(
        self,
        dataloader,
        num_timesteps,
        num_epochs,
    ):

        self.dataloader = dataloader
        self.num_epochs = num_epochs
        self.checkpoint_path = config.dit_checkpoint_path

        self.betas = self.cosine_schedule(num_timesteps).to(my.DEVICE)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.num_timesteps = num_timesteps
        self.multi_loss_tracker = my.MultiLossTracker()
        self.load_or_init()

    def predict_x0(self, x_t, pred_noise, t):
        a_bar = self.get_alpha_bar(t)
        pred_noise = pred_noise[0].detach().cpu()
        a_bar = a_bar[0].detach().cpu()
        # predicted clean image x̂₀
        x0_pred = (x_t - torch.sqrt(1 - a_bar) * pred_noise) / torch.sqrt(a_bar)
        return x0_pred

    def cosine_schedule(self, num_timesteps=1000, s=0.008):
        def f(t):
            return torch.cos((t / num_timesteps + s) / (1 + s) * 0.5 * math.pi) ** 2

        x = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alphas_cumprod = f(x) / f(torch.tensor(0.0))
        betas = 1 - alphas_cumprod[1:] / alphas_cumprod[:-1]
        return torch.clamp(betas, 1e-5, 0.999)

    @staticmethod
    def lr_lambda(current_step: int):
        warmup_steps = -1
        if current_step < warmup_steps:
            return 0.01 + 0.99 * (current_step / warmup_steps)  # linear warmup
        return 1.0

    def log_basic_info(self, epoch):
        now = datetime.datetime.now()
        logger.info(f"Current date and time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Epoch [{epoch+1}/{self.num_epochs}]")

    def log_info(self, epoch, loss, ema_loss):
        avg_loss = self.multi_loss_tracker.calculate_loss("loss", loss)
        avg_ema_loss = self.multi_loss_tracker.calculate_loss("ema_loss", ema_loss)

        self.log_basic_info(epoch)
        logger.info(f"Avg student loss: {avg_loss:.4f}")
        logger.info(f"Avg master (Ema) loss: {avg_ema_loss:.4f}")

    def show_training_state(
        self, loss, ema_loss, real, x_t, pred_noise, ema_pred_noise, epoch, t
    ):

        self.log_info(epoch, loss, ema_loss)
        logger.info(f"t: {t[0].item()} step: {self.step}")
        real = real[0].detach().cpu()
        x_t = x_t[0].detach().cpu()

        x0_pred = self.predict_x0(x_t, pred_noise, t)
        ema_x0_pred = self.predict_x0(x_t, ema_pred_noise, t)
        logger.info(f"Start generating from pure noise. {datetime.datetime.now()}")
        from_pure_noise = self.generate_image_from_pure_noise()
        logger.info(f"Finished generating from pure noise. {datetime.datetime.now()}")

        debug_diffusion(real, x_t, x0_pred, ema_x0_pred, from_pure_noise)

    def get_alpha_bar(self, t):
        return self.alpha_bar[t].view(-1, 1, 1, 1)

    def generate_image_from_pure_noise(self, image_class=None):

        was_training = self.ema.training
        self.ema.eval()

        B = 1
        C, H, W = get_input_shape()
        x_t = torch.randn(B, C, H, W, device=my.DEVICE)

        with torch.inference_mode():
            for t in reversed(range(self.num_timesteps)):

                t_tensor = torch.full((B,), t, device=my.DEVICE, dtype=torch.long)

                # predict noise
                pred_noise = self.ema(x_t, t_tensor, image_class=image_class)

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
        if was_training:
            self.ema.train()

        return x_t.squeeze().cpu()

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

        if self.step % 20 == 0:
            with torch.no_grad():  # Saves memory!
                self.save_checkpoint()
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
        logger.info(f"Save checkpoint: {self.checkpoint_path}")

        checkpoint = {
            "step": self.step,
            "model_state": self.model.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "ema_state": self.ema.state_dict(),
        }
        torch.save(checkpoint, self.checkpoint_path)
        logger.info(f"Checkpoint saved: {self.checkpoint_path}")

    def load_or_init(self):
        logger.info(f"\n::: DIT model:::\n")
        self.model = DiffusionTransformer().to(my.DEVICE)
        if os.path.exists(self.checkpoint_path):
            logger.info("Load model checkpoint")
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
            self.model.parameters(), lr=config.lr, weight_decay=0.01
        )
        self.scheduler = LambdaLR(self.optimizer, self.lr_lambda)

        if checkpoint:
            self.step = checkpoint["step"]
            self.ema.load_state_dict(checkpoint["ema_state"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state"])
            logger.info(f"Checkpoint loaded: {self.checkpoint_path}")
        else:
            logger.info("No checkpoint found — initialized new model and optimizer.")
            self.step = 0

        my.print_parameter_summary(self.model)


class LatentDiffusionTrainer(DiffusionTrainer):
    pass


class PixelDiffusionTrainer(DiffusionTrainer):
    pass


def train():
    dataloader = ImageLatentManager().get_dataloader()

    trainer = LatentDiffusionTrainer(
        dataloader=dataloader,
        num_timesteps=600,
        num_epochs=100000,
    )
    trainer.train()


if __name__ == "__main__":
    logger.info(f"Batch size: {config.batch_size}")
    logger.info(f"LR: {config.lr}")

    torch.cuda.empty_cache()

    train()
