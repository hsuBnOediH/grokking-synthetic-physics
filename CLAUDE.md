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

---

## Experiment Progress (updated 2026-04-08)

### Server
- SSH: `ssh -i ~/.ssh/lab_server fl21@kangaroo.luddy.indiana.edu`
- Project dir: `/data/fl21/CS552_SP26_FinalProject/grokking-synthetic-physics/`

### Data
**Raw (local):** `Pendulum_Grokking_Env/GeneratedData/`
- 300 episodes × 100 frames = 30,000 PNGs (`ep{N}_frame{M}.png`) + `ground_truth.csv`
- 1 unique damping per episode (range 0.011–0.495) — the key experimental variable
- Angle: 0–360°; Camera: X/Y/Z Cartesian (converted to spherical delta for action in prepare_hdf5.py)
- ~29,400 valid transitions (consecutive frames within same episode)

**Preprocessed (server):**
- `pendulum_data.h5` — 7.8MB (245 samples, old local pipeline test only, ignore)
- `pendulum_data_v2.h5` — 699MB (**use this**; preprocessed from all 300 episodes above, ~29,400 transitions)

### Known Issue: Train/Val Split is Transition-Level (not Episode-Level)
Current `train.py` uses `random_split` on transitions (80/20). Since consecutive frames share
images (frame N+1 of episode X appears as both `S_t_next` in one sample and `S_t` in the next),
**val frames may have been seen during training**. This underestimates true generalization error.

**TODO**: Implement episode-level split — hold out 20% of episodes (e.g. 60 episodes) entirely,
so no frames from val episodes appear in training at all.

### Completed
| Phase | Run | Epochs | train_loss | val_loss | Notes |
|-------|-----|--------|------------|----------|-------|
| Phase 0 | pipeline_test (local) | 50 | 0.001202 | 0.001197 | Pipeline validated ✅ |
| Phase 1 | dim_128 (server) | 300 | 0.000194 | 0.001045 | Memorization baseline ✅ train << val confirms memorization |

### In Progress: Phase 2 — Compression Sweep (launched 2026-04-08)
6 runs in parallel, each on a dedicated GPU (CUDA_VISIBLE_DEVICES=0–5), 300 epochs each:
```bash
PYTHON=~/miniconda3/envs/grokking/bin/python
PROJ=/data/fl21/CS552_SP26_FinalProject/grokking-synthetic-physics
cd $PROJ
gpu=0
for dim in 64 32 16 8 4 2; do
  CUDA_VISIBLE_DEVICES=$gpu nohup $PYTHON train.py \
    --model conv --latent_dim $dim \
    --h5_path pendulum_data_v2.h5 \
    --save_dir runs/dim_${dim} \
    --epochs 300 \
    > runs/dim_${dim}_train.log 2>&1 &
  gpu=$((gpu+1))
done
```

**Progress snapshot (epoch ~21-24):**
| dim | epoch | train_loss | val_loss | gap | Observation |
|-----|-------|-----------|---------|-----|-------------|
| 64  | 23 | 0.000597 | 0.001151 | +0.000554 | Large gap → memorizing |
| 32  | 23 | 0.000629 | 0.001008 | +0.000379 | |
| 16  | 23 | 0.000726 | 0.001314 | +0.000588 | |
| 8   | 21 | 0.000814 | 0.001235 | +0.000421 | |
| 4   | 24 | 0.000860 | 0.001467 | +0.000607 | |
| 2   | 21 | 0.001101 | 0.001180 | +0.000079 | Tiny gap → forced generalization ✅ |

After all runs complete → Phase 3: `analyze.py` (linear probe, pairwise distance, t-SNE)

---

### Key Files
| File | Purpose |
|------|---------|
| `models_conv.py` | ConvNet encoder-decoder (primary model) |
| `models.py` | ViT MAE encoder-decoder (comparison model) |
| `train.py` | Training script: `--model {conv,vit}`, `--latent_dim`, logging, checkpointing |
| `HDF5PendulumDataset.py` | Fast HDF5 dataset loader |
| `PendulumDataset.py` | Raw PNG+CSV dataset loader |
| `prepare_hdf5.py` | One-time raw → HDF5 conversion |
