# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CS 552 Final Project: "Beyond the Illusion of Intelligence: Exploring the Knowledge Compression Spectrum."

This project demonstrates that knowledge representations (case-based reasoning, prototypes, abstract rules) are not fundamentally different paradigms, but stages along a continuous **compression spectrum**. By imposing strict information bottlenecks on an encoder-decoder model trained on synthetic physics data, we show how the system evolves from instance-level memorization to compressed, rule-like representations.

The core mechanism: varying `latent_dim` (the bottleneck size) to control compression level. Large dim → memorization (exemplar). Small dim → forces discovery of abstract physical rules (prototype).

## Running Code

No build system. Run Python scripts directly. Dependencies: PyTorch, torchvision, h5py, einops, pandas, Pillow, tqdm.

**Conda environment**: `grokking-synthetic-physics` (`/opt/miniconda3/envs/grokking-synthetic-physics`)
Always activate before running any script:
```bash
conda activate grokking-synthetic-physics
# or: conda run -n grokking-synthetic-physics python <script>
```

```bash
python design_episodes.py --output episode_design.csv     # Generate Unity episode plan (run once)
python models.py                                          # Smoke-test ViT MAE forward pass
python models_conv.py                                     # Smoke-test ConvNet forward pass
python train.py --model conv --latent_dim 128 --epochs 50 # Train ConvNet baseline
python train.py --model vit --latent_dim 10 --epochs 200  # Train ViT MAE
python HDF5PendulumDataset.py --h5-path pendulum_data_v3.h5 --design-csv episode_design.csv  # Validate splits
python prepare_hdf5.py --data-dir GeneratedDataV3 --output pendulum_data_v3.h5               # Convert raw → HDF5
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

## Server
- SSH: `ssh -i ~/.ssh/lab_server fl21@kangaroo.luddy.indiana.edu`
- Project dir: `/data/fl21/CS552_SP26_FinalProject/grokking-synthetic-physics/`

---

## Experiment Design v3 (updated 2026-04-10) — CURRENT

v1 (single damping, 300 episodes): pipeline validation only.
v2 (5D, 3000 episodes, uniform sampling + post-hoc split): **abandoned** — Latin-hypercube holdout with 5 dims yields only (2/3)^5 ≈ 13% IID episodes (320 train), Far-OOD nearly empty (8 episodes). Split imbalance makes evaluation unreliable.

v3 fix: **pre-design episodes by band combination**. `design_episodes.py` enumerates all 3^5=243 band combos, assigns episode counts per OOD level, outputs `episode_design.csv`. Unity reads this CSV and samples each episode's params within the assigned sub-range. IID/Near/Far counts are exact by construction.

### 5 Physical Dimensions

| Dim | Parameter | Unity Variable | Range | Single-frame observable? | Encoding |
|-----|-----------|---------------|-------|-------------------------|----------|
| D1 | **Pendulum length** | `rod.localScale.y` + bob position | [0.5, 2.0] | YES — geometry | Natural visual |
| D2 | **Initial angle** | `pendulumSystem.rotation.x` | [15, 85]° | YES — geometry | Natural visual |
| D3 | **Gravity** | `Physics.gravity.y` | [4.0, 14.0] m/s² | NO — dynamics only | Bob color Hue |
| D4 | **Damping** | `pendulumRb.angularDamping` | [0.01, 0.5] | NO — dynamics only | Bob color Saturation |
| D5 | **Initial angular velocity** | `pendulumRb.angularVelocity.x` | [-3.0, 3.0] rad/s | NO — invisible from single frame | **Not encoded — purely latent** |

Three tiers of information accessibility (itself an interesting finding):
- **Tier A (geometric)**: length, angle — directly readable from pixel geometry
- **Tier B (color-coded)**: gravity, damping — model must learn color → physics mapping
- **Tier C (purely latent)**: initial angular velocity — only inferable from temporal dynamics

Color encoding: `HSV(hue=gravity_normalized, saturation=lerp(0.3,1.0,damping_normalized), value=1.0)`

Note: mass was excluded because Unity's `angularDamping` acts directly on angular velocity (not torque), so mass may cancel out entirely in the equation of motion. Initial angular velocity is used instead — it is unambiguously dynamically relevant and single-frame-invisible.

### Train/Test Split — Stratified Band Partition

Each dimension normalized to [0, 1], divided into 3 bands: Low [0, 0.33], Mid [0.33, 0.67], High [0.67, 1.0].
Latin-hypercube rotation of holdout bands:

| Dimension | Train bands | Holdout band |
|-----------|------------|-------------|
| D1 Length | Low + High | **Mid** |
| D2 Angle | Low + Mid | **High** |
| D3 Gravity | Mid + High | **Low** |
| D4 Damping | Low + High | **Mid** |
| D5 Ang. Velocity | Low + Mid | **High** |

**OOD score** = Σ d_i(p_i), where d_i = 0 if in training band, else min-distance to nearest training band boundary.
Test sets: **IID** (score=0), **Near-OOD** (score ∈ (0, 0.5]), **Far-OOD** (score > 0.5).
Second axis: **k = number of OOD dimensions** (0-5).

Prediction:
- Memorization models: fail at k≥1
- CBR/similarity models: OK at small k + small distance, fail otherwise
- Rule models: OK everywhere

### Data — V3 Structured Episode Design

`design_episodes.py` output (`episode_design.csv`): one row per episode, columns = `episode_id, combo_id, n_ood_dims, split, {param}_lo, {param}_hi` for all 5 params.

| k (n_ood_dims) | Split label | Band combos | eps/combo | Total episodes |
|---|---|---|---|---|
| 0 | iid | 32 | 100 | 3200 |
| 1 | near_ood | 80 | 10 | 800 |
| 2 | near_ood | 80 | 10 | 800 |
| 3 | far_ood | 40 | 15 | 600 |
| 4 | far_ood | 10 | 30 | 300 |
| 5 | far_ood | 1 | 300 | 300 |
| **Total** | | 243 | | **6000** |

- Train/IID-val: 80/20 random split within the 3200 IID episodes → ~2560 train / ~640 val
- Near-OOD evaluation: 1600 episodes (k=1,2)
- Far-OOD evaluation: 1200 episodes (k=3,4,5)
- Total frames: 600K → HDF5 ~14 GB
- Unity generation time: ~5-6 hours

### Bottleneck Sweep

True information dimensions: current angle (1) + angular velocity (1) + gravity (1) + damping (1) + length (1) + camera pose (2) = **7 continuous quantities** → critical transition at latent_dim ≈ 8.

Sweep: `latent_dim = [2, 4, 8, 16, 32, 64, 128]`

| dim | info ratio (dim/7) | Expected behavior |
|-----|--------------------|-------------------|
| 2 | 0.29 | Severe bottleneck, large error but strong generalization |
| 4 | 0.57 | Must prioritize which physics to encode |
| **8** | **1.14** | **Critical point** — sufficient if model discovers correct factorization |
| 16+ | 2.3+ | Increasing memorization |

### Models

| Model | Type | Dynamics? | Priority |
|-------|------|----------|----------|
| **ConvNet AE** | Learned, spatial bias (conv) | Yes (dynamics MLP) | P0 |
| **ViT MAE** | Learned, global attention | Yes (dynamics MLP) | P1 |
| **PCA baseline** | Linear learned | Yes (dynamics MLP on top) | P1 |
| **JPEG baseline** | Hand-crafted DCT | Yes (dynamics MLP on top) | P2 stretch |

JPEG/PCA: compress S_t → z_compressed, train same dynamics MLP architecture, decode back. Isolates compression mechanism effect.

### Evaluation Metrics & Key Plots

**Core metrics:**
- M1: Reconstruction MSE (per IID / Near-OOD / Far-OOD)
- M2: Generalization Gap Ratio = (MSE_OOD - MSE_IID) / MSE_IID — **the key metric**
- M3: Linear Probe R² (z_t → each ground-truth dimension)
- M4: Latent disentanglement (correlation matrix z dims vs GT dims)
- M5: OOD performance by number of OOD dimensions k

**6 key plots:**
1. Money Plot: latent_dim vs Generalization Gap Ratio (ConvNet/ViT/PCA lines)
2. IID vs OOD Error: latent_dim vs MSE, solid=IID dashed=OOD
3. Linear Probe Heatmap: rows=GT dims, cols=latent_dim, color=R²
4. OOD by k: x=k, y=MSE, one curve per latent_dim
5. t-SNE/UMAP of latent space at dim ∈ {2,8,32,128}, colored by GT dims
6. Reconstruction montage: rows=latent_dim, cols=IID/Near-OOD/Far-OOD examples

### Known Confounds
1. **Model capacity vs bottleneck**: changing latent_dim also changes total params → report param counts
2. **Color encoding leaks physics**: gravity/damping readable from single frame → control experiment with fixed white bob
3. **Camera pose uses latent capacity**: may mask physics compression → include camera in linear probes

### Fallback
If 5D too sparse: fall back to 3D (length, gravity, damping). If time runs out: keep 1D damping setup + proper episode-level OOD split.

---

### v1 Reference Results (single-dimension damping, 2026-04-08)

These results validated the pipeline and gave initial signal for the compression spectrum. Not used in final report.

| dim | train_loss | val_loss | val/train | z_std |
|-----|-----------|---------|-----------|-------|
| 128 | 0.000194 | 0.001045 | 5.4x | 1.21 |
| 64 | 0.000143 | 0.000546 | 3.8x | 1.62 |
| 32 | 0.000190 | 0.000564 | 3.0x | 2.15 |
| 16 | 0.000237 | 0.000763 | 3.2x | 3.25 |
| 8 | 0.000331 | 0.000738 | 2.2x | 3.77 |
| 4 | 0.000468 | 0.001070 | 2.3x | 5.34 |
| 2 | 0.000636 | 0.000777 | 1.2x | 6.32 |

---

### Key Files
| File | Purpose |
|------|---------|
| `models_conv.py` | ConvNet encoder-decoder (primary model) |
| `models.py` | ViT MAE encoder-decoder (comparison model) |
| `train.py` | Training script: `--model {conv,vit}`, `--latent_dim`, logging, checkpointing |
| `design_episodes.py` | **V3** — enumerate 243 band combos, output `episode_design.csv` for Unity |
| `HDF5PendulumDataset.py` | Fast HDF5 loader; `make_splits()` builds 4 DataLoaders from design CSV |
| `prepare_hdf5.py` | One-time raw → HDF5 conversion (all 5D fields + episode ID) |
| `DataGenerator.cs` | Unity V3 — reads `episode_design.csv`, minimal Inspector, samples within sub-ranges |
| `PendulumDataset.py` | Raw PNG+CSV loader (legacy, not used in main pipeline) |
| `split_dataset.py` | V2 leftover — superseded by `design_episodes.py`, keep for reference |

---

## Implementation Status (as of 2026-04-10)

### DONE ✅
- `DataGenerator.cs` — V3 rewrite: reads `episode_design.csv`, samples within sub-ranges, minimal Inspector (5 scene refs + file path + framesPerEpisode)
- `design_episodes.py` — enumerates 243 band combos, assigns episodes per k-level, outputs `episode_design.csv`
- `prepare_hdf5.py` — supports all 5D fields (gravity, length, init_angular_velocity, episode)
- `HDF5PendulumDataset.py` — V3: filters by episode_design.csv split labels; `make_splits()` builds 4 DataLoaders

### TODO (next session) 📋

**Step 1 — Generate episode_design.csv and trigger Unity**:
```bash
# Generate the design file (run once)
python design_episodes.py --output episode_design.csv
# → 6000 episodes: 3200 IID / 1600 near_ood / 1200 far_ood

# Copy to Unity project root (where DataGenerator reads it)
cp episode_design.csv /path/to/UnityProject/
```
In Unity Inspector: set `episodeDesignFile = "episode_design.csv"`, `saveDirectory = "GeneratedDataV3"`. Hit Play.

**Step 2 — Post-generation validation** (after Unity finishes):
```bash
# Validate raw CSV
python3 -c "
import pandas as pd
df = pd.read_csv('GeneratedDataV3/ground_truth.csv')
print('Episodes:', df.Episode.nunique(), '  Frames:', len(df))
print(df.groupby('Episode').first()[['Damping','Gravity','Length','InitAngularVelocity','Angle']].describe())
"

# Convert to HDF5
python prepare_hdf5.py --data-dir GeneratedDataV3 --output pendulum_data_v3.h5

# Validate splits (should match episode_design.csv counts)
python HDF5PendulumDataset.py --h5-path pendulum_data_v3.h5 --design-csv episode_design.csv
```

**Step 3 — Update train.py** to use `make_splits()`:
```python
from HDF5PendulumDataset import make_splits
loaders = make_splits("pendulum_data_v3.h5", "episode_design.csv")
# loaders: {"train", "iid_val", "near_ood", "far_ood"}
```
Log all 4 split losses per epoch.

**Step 4 — Transfer to server and run sweep**:
```bash
scp pendulum_data_v3.h5 fl21@kangaroo.luddy.indiana.edu:/data/fl21/CS552_SP26_FinalProject/
scp episode_design.csv  fl21@kangaroo.luddy.indiana.edu:/data/fl21/CS552_SP26_FinalProject/
```
- 7 latent_dims × ConvNet = 7 runs (parallel on 6 GPUs)
- Then ViT 7 runs
- Then PCA baseline
