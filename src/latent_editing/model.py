from typing import Dict, Tuple

import torch
import torch.nn as nn

from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from .config import Config
from .data import make_latent_loaders


def load_vae(
    cfg: Config,
) -> AutoencoderKL:
    """
    Load the pretrained Stable Diffusion VAE and freeze all
    of its parameters.

    The VAE is used only as an encoder/decoder. It is not
    fine-tuned during this project.
    """

    print(
        f"Loading VAE: "
        f"{cfg.sd_vae_model_id}"
    )

    vae = AutoencoderKL.from_pretrained(
        cfg.sd_vae_model_id
    )

    vae = vae.to(
        cfg.device
    )

    vae.eval()

    for parameter in vae.parameters():
        parameter.requires_grad = False

    print(
        f"Loaded VAE on: {cfg.device}"
    )

    print(
        "Latent channels:",
        vae.config.latent_channels,
    )

    return vae


class LatentLinearProbe(nn.Module):
    """
    Linear binary classifier operating on flattened
    Stable Diffusion VAE latents.

    The learned weight vector later serves as the semantic
    attribute direction.
    """

    def __init__(
        self,
        latent_dim: int,
    ):
        super().__init__()

        self.fc = nn.Linear(
            latent_dim,
            1,
        )

    def forward(
        self,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        z:
            Latent tensor with shape (B, C, H, W).

        Returns
        -------
        Tensor
            Binary classification logits with shape (B,).
        """

        batch_size = z.shape[0]

        z_flat = z.reshape(
            batch_size,
            -1,
        )

        logits = self.fc(
            z_flat
        )

        return logits.squeeze(-1)


@torch.no_grad()
def evaluate_probe(
    model: nn.Module,
    dataloader: DataLoader,
    device: str,
) -> float:
    """
    Compute binary classification accuracy for a trained
    linear probe.
    """

    model.eval()

    correct = 0
    total = 0

    for z_batch, y_batch in dataloader:
        z_batch = z_batch.to(
            device
        )

        y_batch = y_batch.to(
            device
        ).float()

        logits = model(
            z_batch
        )

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities > 0.5
        ).long()

        correct += (
            predictions
            == y_batch.long()
        ).sum().item()

        total += y_batch.numel()

    if total == 0:
        return 0.0

    return correct / total


@torch.no_grad()
def extract_direction(
    probe: LatentLinearProbe,
) -> torch.Tensor:
    """
    Extract and normalize the linear probe's weight vector.

    For a binary linear probe:

        f(z) = sigmoid(w^T z + b)

    w points in the direction of increasing classifier
    confidence and is therefore treated as the semantic
    editing direction.
    """

    weights = (
        probe.fc.weight
        .detach()
        .clone()
        .reshape(-1)
    )

    norm = weights.norm()

    if norm.item() == 0:
        raise ValueError(
            "Cannot create semantic direction "
            "because probe weight norm is zero."
        )

    direction = (
        weights /
        (norm + 1e-8)
    )

    return direction


def train_probe(
    attr_name: str,
    z_train: torch.Tensor,
    y_train: torch.Tensor,
    z_val: torch.Tensor,
    y_val: torch.Tensor,
    attr_to_idx: Dict[str, int],
    cfg: Config,
) -> Tuple[LatentLinearProbe, torch.Tensor]:
    """
    Train one linear probe for a CelebA facial attribute.

    Parameters
    ----------
    attr_name:
        CelebA attribute name such as:
        "Smiling", "Eyeglasses", "Young".

    z_train, y_train:
        Precomputed training latents and labels.

    z_val, y_val:
        Precomputed validation latents and labels.

    attr_to_idx:
        Mapping from CelebA attribute names to column indices.

    cfg:
        Project configuration.

    Returns
    -------
    probe:
        Trained LatentLinearProbe.

    direction:
        Normalized semantic direction derived from the
        probe's learned weight vector.
    """

    train_loader, val_loader = make_latent_loaders(
        attr_name=attr_name,
        z_train=z_train,
        y_train=y_train,
        z_val=z_val,
        y_val=y_val,
        attr_to_idx=attr_to_idx,
        cfg=cfg,
    )

    # Infer latent dimensionality from the data.
    z_example, _ = next(
        iter(train_loader)
    )

    latent_dim = z_example[
        0
    ].numel()

    print(
        f"[{attr_name}] "
        f"flattened latent dimension: "
        f"{latent_dim}"
    )

    probe = LatentLinearProbe(
        latent_dim=latent_dim
    ).to(
        cfg.device
    )

    optimizer = torch.optim.AdamW(
        probe.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(
        cfg.num_epochs
    ):
        probe.train()

        running_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=(
                f"[{attr_name}] "
                f"Epoch {epoch + 1}/"
                f"{cfg.num_epochs}"
            ),
        )

        for (
            z_batch,
            y_batch,
        ) in progress_bar:

            z_batch = z_batch.to(
                cfg.device
            )

            y_batch = y_batch.to(
                cfg.device
            ).float()

            logits = probe(
                z_batch
            )

            loss = criterion(
                logits,
                y_batch,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * z_batch.size(0)
            )

            progress_bar.set_postfix(
                loss=f"{loss.item():.4f}"
            )

        train_loss = (
            running_loss
            / len(train_loader.dataset)
        )

        val_accuracy = evaluate_probe(
            probe,
            val_loader,
            cfg.device,
        )

        print(
            f"[{attr_name}] "
            f"Epoch {epoch + 1}: "
            f"loss={train_loss:.4f} | "
            f"val_acc={val_accuracy:.4f}"
        )

    direction = extract_direction(
        probe
    )

    return probe, direction