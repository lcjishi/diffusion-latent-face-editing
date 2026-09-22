from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchvision

from diffusers import AutoencoderKL

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from tqdm.auto import tqdm

from .config import Config


def get_transform(cfg: Config):
    """
    Create the preprocessing pipeline used for CelebA images.

    Images are:
        1. Resized to cfg.image_size x cfg.image_size
        2. Center cropped
        3. Converted to tensors
        4. Normalized from [0, 1] to [-1, 1]

    The [-1, 1] range matches the expected input range of
    the Stable Diffusion VAE.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (cfg.image_size, cfg.image_size)
            ),
            transforms.CenterCrop(
                cfg.image_size
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.5],
                [0.5],
            ),
        ]
    )


def load_celeba(
    cfg: Config,
    download: bool = False,
):
    """
    Load the CelebA training and validation datasets.

    Parameters
    ----------
    cfg:
        Project configuration.

    download:
        Whether torchvision should attempt to download CelebA.
        If you already have the dataset under data/celeba/,
        leave this False.

    Returns
    -------
    train_dataset
    val_dataset
    attr_names
    attr_to_idx
    """

    transform = get_transform(cfg)

    train_dataset = torchvision.datasets.CelebA(
        root=cfg.data_root,
        split="train",
        target_type="attr",
        transform=transform,
        download=download,
    )

    val_dataset = torchvision.datasets.CelebA(
        root=cfg.data_root,
        split="valid",
        target_type="attr",
        transform=transform,
        download=download,
    )

    attr_names = list(train_dataset.attr_names)

    attr_to_idx = {
        name: i
        for i, name in enumerate(attr_names)
    }

    return (
        train_dataset,
        val_dataset,
        attr_names,
        attr_to_idx,
    )


@torch.no_grad()
def encode_to_latent(
    x: torch.Tensor,
    vae: AutoencoderKL,
    latent_scale: float,
) -> torch.Tensor:
    """
    Encode an image batch into Stable Diffusion VAE latent space.

    The original project samples from the VAE posterior:

        z ~ q(z | x)

    and then applies Stable Diffusion's standard scaling factor.

    Parameters
    ----------
    x:
        Image tensor of shape (B, C, H, W).

    vae:
        Frozen Stable Diffusion AutoencoderKL.

    latent_scale:
        Stable Diffusion scaling factor, normally 0.18215.

    Returns
    -------
    Tensor
        Scaled latent tensors of shape (B, 4, H_latent, W_latent).
    """

    posterior = vae.encode(x).latent_dist

    z = posterior.sample()

    return z * latent_scale


def precompute_and_save_latents(
    train_dataset,
    val_dataset,
    vae: AutoencoderKL,
    attr_names: List[str],
    cfg: Config,
) -> None:
    """
    Encode a subset of CelebA through the frozen VAE and save
    the resulting latent tensors and attribute labels.

    This allows linear probes to be trained without repeatedly
    running the VAE encoder.

    Saved dictionary:

        z_train
        y_train
        z_val
        y_val
        attr_names
    """

    print("Precomputing CelebA VAE latents...")

    pin_memory = cfg.device.startswith("cuda")

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size_precompute,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size_precompute,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    z_train_list = []
    y_train_list = []

    z_val_list = []
    y_val_list = []

    # =========================================================
    # Training latents
    # =========================================================

    n_seen = 0

    for images, attrs in tqdm(
        train_loader,
        desc="Train latents",
    ):
        images = images.to(cfg.device)

        z = encode_to_latent(
            images,
            vae,
            cfg.latent_scale,
        )

        z_train_list.append(
            z.cpu()
        )

        y_train_list.append(
            attrs.cpu()
        )

        n_seen += images.size(0)

        if n_seen >= cfg.max_train_latents:
            break

    z_train = torch.cat(
        z_train_list,
        dim=0,
    )[: cfg.max_train_latents]

    y_train = torch.cat(
        y_train_list,
        dim=0,
    )[: cfg.max_train_latents]

    # =========================================================
    # Validation latents
    # =========================================================

    n_seen = 0

    for images, attrs in tqdm(
        val_loader,
        desc="Validation latents",
    ):
        images = images.to(cfg.device)

        z = encode_to_latent(
            images,
            vae,
            cfg.latent_scale,
        )

        z_val_list.append(
            z.cpu()
        )

        y_val_list.append(
            attrs.cpu()
        )

        n_seen += images.size(0)

        if n_seen >= cfg.max_val_latents:
            break

    z_val = torch.cat(
        z_val_list,
        dim=0,
    )[: cfg.max_val_latents]

    y_val = torch.cat(
        y_val_list,
        dim=0,
    )[: cfg.max_val_latents]

    # =========================================================
    # Save
    # =========================================================

    output_path = Path(
        cfg.latents_file
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "z_train": z_train,
            "y_train": y_train,
            "z_val": z_val,
            "y_val": y_val,
            "attr_names": attr_names,
        },
        output_path,
    )

    print(
        f"Saved latents to: {output_path}"
    )

    print(
        "z_train:",
        z_train.shape,
    )

    print(
        "z_val:",
        z_val.shape,
    )


def load_precomputed_latents(
    path: str,
):
    """
    Load the latent dataset previously generated by
    precompute_and_save_latents().

    Returns
    -------
    z_train
    y_train
    z_val
    y_val
    attr_names
    attr_to_idx
    """

    data = torch.load(
        path,
        map_location="cpu",
    )

    z_train = data["z_train"]
    y_train = data["y_train"]

    z_val = data["z_val"]
    y_val = data["y_val"]

    attr_names = list(
        data["attr_names"]
    )

    attr_to_idx = {
        name: i
        for i, name in enumerate(attr_names)
    }

    return (
        z_train,
        y_train,
        z_val,
        y_val,
        attr_names,
        attr_to_idx,
    )


class LatentAttributeDataset(Dataset):
    """
    Dataset containing precomputed VAE latents with a single
    CelebA binary attribute as the target.

    CelebA stores attribute labels as -1 or +1.
    This class converts them to 0 or 1 for binary classification.
    """

    def __init__(
        self,
        z_tensor: torch.Tensor,
        attr_tensor: torch.Tensor,
        attr_idx: int,
    ):
        """
        Parameters
        ----------
        z_tensor:
            Tensor of shape (N, C, H, W).

        attr_tensor:
            Tensor of shape (N, 40), with entries -1 or +1.

        attr_idx:
            Index of the CelebA attribute to use as the label.
        """

        self.z = z_tensor
        self.attrs = attr_tensor
        self.attr_idx = attr_idx

    def __len__(self):
        return self.z.shape[0]

    def __getitem__(self, idx):
        z = self.z[idx]

        attr_vector = self.attrs[idx]

        label = (
            1
            if attr_vector[self.attr_idx].item() == 1
            else 0
        )

        return z, label


def make_latent_loaders(
    attr_name: str,
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    attr_to_idx: Dict[str, int],
    cfg: Config,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation DataLoaders for one CelebA
    facial attribute.
    """

    if attr_name not in attr_to_idx:
        raise ValueError(
            f"Unknown CelebA attribute: {attr_name}"
        )

    attr_idx = attr_to_idx[
        attr_name
    ]

    print(
        f"\nPreparing loaders for "
        f"{attr_name} "
        f"(attribute index {attr_idx})"
    )

    train_dataset = LatentAttributeDataset(
        z_train,
        y_train,
        attr_idx,
    )

    val_dataset = LatentAttributeDataset(
        z_val,
        y_val,
        attr_idx,
    )

    pin_memory = cfg.device.startswith(
        "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size_probe,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size_probe,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    print(
        f"Train latents: {len(train_dataset)}"
    )

    print(
        f"Validation latents: {len(val_dataset)}"
    )

    return train_loader, val_loader