# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS 552 Final Project: "Beyond the Illusion of Intelligence: Exploring the Knowledge Compression Spectrum."

This project demonstrates that knowledge representations (case-based reasoning, prototypes, abstract rules) are not fundamentally different paradigms, but stages along a continuous **compression spectrum**. By imposing strict information bottlenecks on an encoder-decoder model trained on synthetic physics data, we show how the system evolves from instance-level memorization to compressed, rule-like representations.

The core mechanism: varying `latent_dim` (the bottleneck size) to control compression level. Large dim → memorization (exemplar). Small dim → forces discovery of abstract physical rules (prototype).

## Running Code

No build system. Run Python scripts directly. Dependencies: PyTorch, torchvision, h5py, einops, pandas, Pillow, tqdm.

```bash
python models.py                                          # Smoke-test ViT MAE forward pass
python models_conv.py                                     # Smoke-test ConvNet forward pass
python train.py --model conv --latent_dim 128 --epochs 50 # Train ConvNet baseline
python train.py --model vit --latent_dim 10 --epochs 200  # Train ViT MAE
python HDF5PendulumDataset.py --h5-path pendulum_data.h5  # Validate HDF5 dataset
python prepare_hdf5.py --data-dir /path/to/GeneratedData --output pendulum_data.h5  # Convert raw → HDF5
```

All Python modules have `if __name__ == "__main__"` test blocks.

## Architecture

### Data Pipeline
Raw dataset from a Unity 3D physics engine: `.png` frames (64x64 RGB) + `ground_truth.csv` with episode IDs, camera positions, pendulum angle, and damping factor (encoded as bob color). `PendulumDataset` computes valid same-episode sequential transitions; `prepare_hdf5.py` pre-computes these into HDF5 for fast I/O. The action `A_t = [delta_azimuth, delta_elevation]` in spherical coordinates.

### Two Model Architectures (same interface)

Both models share the same forward signature: `forward(s_t, action) → (pred_s_next, z_t)` where `z_t` is the bottleneck representation for probing/analysis.

**ConvNet (`models_conv.py`) — primary, low-risk baseline:**
- Conv encoder (4 layers, stride=2, 64x64→4x4) → flatten → Linear → `Z_t`
- Dynamics MLP: `[Z_t, A_t]` → `Z_{t+1}`
- ConvTranspose decoder (mirror of encoder) → pixel reconstruction

**ViT MAE (`models.py`) — comparison/upgrade:**
- Patch embeddings → ViT encoder (4 layers, embed_dim=128) → CLS token → `Z_t`
- Same dynamics MLP
- ViT decoder (2 layers, embed_dim=64) → pixel reconstruction

### The Compression Spectrum Experiment

The `latent_dim` parameter is the primary experimental variable:
- **Large (64-128):** Model memorizes each sample independently — exemplar/lookup table
- **Medium (16-32):** Similar physical states share latent codes — CBR emergence
- **Small (2-8):** Latent dims forced to encode abstract physical variables — rule emergence

Evidence is gathered via: reconstruction loss curves, pairwise latent distance analysis, linear probes from `z_t` to ground truth (damping, angle), and t-SNE/UMAP visualization.

### Key Files
| File | Purpose |
|------|---------|
| `models_conv.py` | ConvNet encoder-decoder (primary model) |
| `models.py` | ViT MAE encoder-decoder (comparison model) |
| `train.py` | Training script: `--model {conv,vit}`, `--latent_dim`, logging, checkpointing |
| `HDF5PendulumDataset.py` | Fast HDF5 dataset loader |
| `PendulumDataset.py` | Raw PNG+CSV dataset loader |
| `prepare_hdf5.py` | One-time raw → HDF5 conversion |
