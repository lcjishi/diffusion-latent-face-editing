from dataclasses import dataclass
import random

import numpy as np
import torch


@dataclass
class Config:
    """
    Central configuration for the latent face editing project.
    """

    # ---------------------------------------------------------
    # General
    # ---------------------------------------------------------
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------
    # Torchvision expects CelebA files under:
    # data/celeba/
    data_root: str = "./data"

    image_size: int = 128

    # Number of CelebA examples whose VAE latents are stored.
    max_train_latents: int = 2000
    max_val_latents: int = 200

    # ---------------------------------------------------------
    # Latent precomputation
    # ---------------------------------------------------------
    batch_size_precompute: int = 8
    num_workers: int = 0

    # ---------------------------------------------------------
    # Stable Diffusion VAE
    # ---------------------------------------------------------
    sd_vae_model_id: str = "stabilityai/sd-vae-ft-mse"

    # Stable Diffusion latent scaling constant.
    latent_scale: float = 0.18215

    # ---------------------------------------------------------
    # Linear probe training
    # ---------------------------------------------------------
    batch_size_probe: int = 64
    learning_rate: float = 1e-3
    num_epochs: int = 10
    weight_decay: float = 1e-4

    # ---------------------------------------------------------
    # Editing / visualization
    # ---------------------------------------------------------
    edit_step_scale: float = 4.0
    num_visualization_samples: int = 8

    # ---------------------------------------------------------
    # Files
    # ---------------------------------------------------------
    latents_file: str = "./data/processed/celeba_sdvae_latents.pt"
    checkpoint_dir: str = "./checkpoints"


def set_seed(seed: int) -> None:
    """
    Seed Python, NumPy, and PyTorch random number generators.

    This makes experiments more reproducible.
    """

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)