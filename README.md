# Neural Face Attribute Editing via Stable Diffusion VAE Latent Directions

**Author:** Chenjishi Lin  
**Course:** Computer Vision — Final Project  
**Institution:** Columbia University  

---

## Overview

This project implements a lightweight and interpretable facial attribute editing by learning semantic directions directly in the latent space of a pretrained Stable Diffusion VAE.

Instead of performing diffusion inversion or iterative denoising, this project encodes faces once, learns linear attribute directions from CelebA, and edits images using simple latent-space arithmetic followed by a single VAE decoder pass.

A frozen `stabilityai/sd-vae-ft-mse` encoder is used to obtain latent
representations of CelebA images. Lightweight linear probes are then trained
to distinguish facial attributes such as:

- Smiling
- Eyeglasses
- Young
- Receding Hairline

The normalized weight vector of each trained probe is interpreted as a semantic direction in latent space.

For a latent representation `z`, editing is performed using:
$$z' = z + \alpha d$$
where:
- `d` is the learned semantic direction
- `α` controls the strength and sign of the edit

Because the learned directions occupy the same latent space, multiple attributes can also be combined through weighted linear combinations.

## Method

The pipeline consists of four main stages:

```text
CelebA image
     │
     ▼
Frozen Stable Diffusion VAE Encoder
     │
     ▼
VAE latent representation
     │
     ▼
Linear attribute probe
     │
     ▼
Normalized probe weights
     │
     ▼
Semantic direction
     │
     ▼
Latent manipulation
     │
     ▼
Frozen VAE Decoder
     │
     ▼
Edited image
```

---

## Installation

Clone the repository, then use venv environment or conda to create a Python environment.

```bash
conda create -n latentedit python=3.12
conda activate latentedit
```

Install dependencies:
```bash
pip install -r requirements.txt
```

The project uses the CelebA dataset, the repo does not include the dataset or precomputed latent files. If the precomputed latent file does not exist, the notebook can generate it from the CelebA images using the frozen Stable Diffusion VAE.

Run the experiments using `demo.ipynb`

---
## Results
The learned linear probes shows several facial attributes are strongly
separable in the frozen Stable Diffusion VAE latent space.

| Attribute | Validation Accuracy |
| --- | ---: |
| Smiling | 0.81 |
| Eyeglasses | 0.95 |
| Young | 0.81 |
| Receding Hairline | 0.92 |

---
## Acknowledgments

The project builds on
- CelebA
- Stable Diffusion / Latent Diffusion
- Hugging Face Diffusers

The pretrained VAE used in the experiments is:
stabilityai/sd-vae-ft-mse