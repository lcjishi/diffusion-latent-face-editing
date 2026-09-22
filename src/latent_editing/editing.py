from typing import Dict, Optional

import matplotlib.pyplot as plt

import torch
import torchvision

from diffusers import AutoencoderKL

from .config import Config


def _get_model_device(
    model: torch.nn.Module,
) -> torch.device:
    """
    Return the device containing a model's parameters.
    """

    return next(
        model.parameters()
    ).device


@torch.no_grad()
def decode_from_latent(
    z: torch.Tensor,
    vae: AutoencoderKL,
    latent_scale: float,
) -> torch.Tensor:
    """
    Decode scaled Stable Diffusion VAE latents back into images.

    Parameters
    ----------
    z:
        Scaled latent tensors.

    vae:
        Frozen Stable Diffusion VAE.

    latent_scale:
        Stable Diffusion scaling factor used during encoding.

    Returns
    -------
    Tensor
        Reconstructed images in the range [-1, 1].
    """

    device = _get_model_device(
        vae
    )

    # Undo Stable Diffusion scaling before VAE decoding.
    z_unscaled = (
        z /
        latent_scale
    )

    reconstruction = vae.decode(
        z_unscaled.to(device)
    ).sample

    reconstruction = reconstruction.clamp(
        -1,
        1,
    )

    return reconstruction.cpu()


def edit_latent(
    z: torch.Tensor,
    direction: torch.Tensor,
    alpha: float,
) -> torch.Tensor:
    """
    Shift latent representations along one semantic direction.

    Implements:

        z' = z + alpha * d

    where:
        z       = original latent
        d       = normalized semantic direction
        alpha   = edit strength

    Positive alpha strengthens the learned attribute.
    Negative alpha suppresses it.
    """

    if z.ndim != 4:
        raise ValueError(
            "Expected z with shape "
            "(B, C, H, W)."
        )

    batch_size, channels, height, width = (
        z.shape
    )

    z_flat = z.reshape(
        batch_size,
        -1,
    )

    direction = direction.to(
        device=z_flat.device,
        dtype=z_flat.dtype,
    )

    if direction.numel() != z_flat.shape[1]:
        raise ValueError(
            "Semantic direction dimension "
            f"({direction.numel()}) does not match "
            "flattened latent dimension "
            f"({z_flat.shape[1]})."
        )

    edited_flat = (
        z_flat
        + alpha
        * direction.unsqueeze(0)
    )

    edited = edited_flat.reshape(
        batch_size,
        channels,
        height,
        width,
    )

    return edited


def combine_directions(
    directions: Dict[str, torch.Tensor],
    weights: Dict[str, float],
) -> torch.Tensor:
    """
    Combine several semantic directions using a weighted
    linear combination.

    Example
    -------
    weights = {
        "Smiling": 1.0,
        "Eyeglasses": 0.7,
        "Young": 0.8,
    }

    The resulting direction is normalized before being
    returned.
    """

    if not weights:
        raise ValueError(
            "At least one semantic direction is required."
        )

    combined = None

    for (
        attribute,
        weight,
    ) in weights.items():

        if attribute not in directions:
            raise KeyError(
                f"No learned direction found "
                f"for '{attribute}'."
            )

        direction = directions[
            attribute
        ]

        contribution = (
            float(weight)
            * direction
        )

        if combined is None:
            combined = (
                contribution.clone()
            )
        else:
            combined = (
                combined
                + contribution
            )

    norm = combined.norm()

    if norm.item() == 0:
        raise ValueError(
            "Combined semantic direction has "
            "zero norm."
        )

    return (
        combined /
        (norm + 1e-8)
    )


def show_image_grid(
    tensor: torch.Tensor,
    nrow: int = 4,
    title: Optional[str] = None,
    figsize=(8, 8),
) -> None:
    """
    Display a batch of images as a grid.

    Input images are assumed to be in the range [-1, 1].
    """

    images = (
        tensor.detach().cpu()
        * 0.5
    ) + 0.5

    images = images.clamp(
        0,
        1,
    )

    grid = torchvision.utils.make_grid(
        images,
        nrow=nrow,
    )

    plt.figure(
        figsize=figsize
    )

    plt.imshow(
        grid.permute(
            1,
            2,
            0,
        ).numpy()
    )

    plt.axis("off")

    if title is not None:
        plt.title(
            title
        )

    plt.tight_layout()

    plt.show()


def demo_edit(
    z_val: torch.Tensor,
    direction: torch.Tensor,
    vae: AutoencoderKL,
    cfg: Config,
    attribute_name: str,
    alpha: Optional[float] = None,
) -> None:
    """
    Visualize original VAE reconstructions, positive edits,
    and negative edits for one semantic direction.

    Unlike the original notebook function, this function does
    not rely on global z_val, vae, or cfg variables.
    """

    if alpha is None:
        alpha = (
            cfg.edit_step_scale
        )

    num_samples = min(
        cfg.num_visualization_samples,
        z_val.shape[0],
    )

    z_original = (
        z_val[:num_samples]
        .clone()
    )

    # ---------------------------------------------------------
    # Original VAE reconstructions
    # ---------------------------------------------------------

    x_original = decode_from_latent(
        z_original,
        vae,
        cfg.latent_scale,
    )

    # ---------------------------------------------------------
    # Positive edit
    # ---------------------------------------------------------

    z_positive = edit_latent(
        z_original,
        direction,
        alpha,
    )

    x_positive = decode_from_latent(
        z_positive,
        vae,
        cfg.latent_scale,
    )

    # ---------------------------------------------------------
    # Negative edit
    # ---------------------------------------------------------

    z_negative = edit_latent(
        z_original,
        direction,
        -alpha,
    )

    x_negative = decode_from_latent(
        z_negative,
        vae,
        cfg.latent_scale,
    )

    # ---------------------------------------------------------
    # Display
    # ---------------------------------------------------------

    show_image_grid(
        x_original,
        title=(
            f"[{attribute_name}] "
            "Original VAE reconstruction"
        ),
    )

    show_image_grid(
        x_positive,
        title=(
            f"[{attribute_name}] "
            f"+{alpha:g} → more "
            f"{attribute_name}"
        ),
    )

    show_image_grid(
        x_negative,
        title=(
            f"[{attribute_name}] "
            f"-{alpha:g} → less "
            f"{attribute_name}"
        ),
    )