# Compression Spectrum on Synthetic Physics — Code README

Repository: <https://github.com/hsuBnOediH/grokking-synthetic-physics>

Dataset and full experimental outputs (HDF5 frames, training logs,
checkpoints, probe results) are mirrored on Google Drive:

> <https://drive.google.com/file/d/1N4rixKHdS5GHfcp3i2OJQLXT5G6GgZrd/view?usp=sharing>

This submission contains the code, data-generation pipeline, training
scripts, and evaluation tools for the project *Beyond the "Illusion of
Intelligence": Exploring the Knowledge Compression Spectrum*. The paper
has been submitted separately. To inspect the dataset and full training
artifacts, use the Drive link above.

---

## 1. Repository Layout

```
.
├── README.md                       # this file
├── grokking-synthetic-physics/     # Python pipeline (training / probes)
│   ├── design_episodes.py          # Generate episode plan -> episode_design.csv
│   ├── episode_design.csv          # 6,000-row plan; consumed by Unity & loader
│   ├── prepare_hdf5.py             # Raw PNG + ground_truth.csv -> HDF5
│   ├── HDF5PendulumDataset.py      # Fast HDF5 dataloader, IID/Near/Far splits
│   ├── PendulumDataset.py          # Legacy raw-PNG loader (only imported by
│   │                               # prepare_hdf5.py for default paths)
│   ├── models_conv.py              # ConvNet encoder/decoder + dynamics MLP
│   ├── models.py                   # ViT MAE encoder/decoder + dynamics MLP
│   ├── models_dct.py               # Fixed-DCT encoder/decoder, MLP only trainable
│   ├── train.py                    # Training loop with z_std early stopping
│   ├── probe.py                    # Linear probe (Ridge, R^2) on frozen latents
│   └── launch_vit_after_conv.sh    # Watcher script (waits for ConvNet PIDs,
│                                   # then launches ViT sweep on 7 GPUs)
│
└── Pendulum_Grokking_Env/          # Unity 2022.3 URP project (stock template)
    └── Assets/
        └── DataGenerator.cs        # the only authored Unity asset; the rest
                                    # of the URP project is unmodified stock
                                    # template and is not included here
```

All Python and C# in this submission was written by the authors. External
dependencies are listed in Sections 2.1 and 7.

---

## 2. Environment Setup

### 2.1 Python (training and evaluation)

```bash
conda create -n grokking-synthetic-physics python=3.10
conda activate grokking-synthetic-physics
pip install torch torchvision h5py einops numpy pandas Pillow tqdm \
            scikit-learn matplotlib
```

Any recent CUDA-enabled PyTorch works; we did not pin specific versions. CPU
is sufficient for the forward-pass smoke tests (Section 4.1); the full
training sweep requires a GPU.

### 2.2 Unity (only needed to regenerate raw frames)

The submitted file `Pendulum_Grokking_Env/Assets/DataGenerator.cs` is the
only authored Unity asset; the rest of the URP project is unmodified stock
template and is not included in the submission.

To regenerate the raw frame dataset, create a fresh Unity 2022.3 LTS project
from the Universal Render Pipeline template, drop `DataGenerator.cs` into
`Assets/`, and place `episode_design.csv` at the project root. On a Mac M1
Pro the full 6,000-episode render takes roughly 10 hours.

To reproduce the paper's results, this step is not necessary — the prepared
HDF5 dataset is mirrored on Google Drive (see Section 3.1).

---

## 3. Data Pipeline

### 3.1 Pre-rendered dataset (recommended)

The packed HDF5 dataset (`pendulum_data_v3.h5`, ~14 GB, ~600,000 transitions
across 6,000 episodes) is hosted on Google Drive:

> <https://drive.google.com/file/d/1N4rixKHdS5GHfcp3i2OJQLXT5G6GgZrd/view?usp=sharing>

Place the HDF5 in `grokking-synthetic-physics/` together with
`episode_design.csv` and skip directly to Section 4. Sections 3.2–3.4 below
describe the full pipeline for completeness — none of them are required when
using the pre-rendered dataset.

### 3.2 Generating raw frames (Unity)

1. Run `python design_episodes.py --output episode_design.csv` to produce
   the 6,000-row episode plan. The script enumerates all `3^5 = 243` band
   combinations over the five physical parameters, classifies each combo by
   `n_ood_dims` (the number of dimensions in the holdout band), and assigns
   episodes per `(combo, n_ood_dims)` cell to hit the target counts (default
   3,200 IID + 1,600 Near-OOD + 1,200 Far-OOD).
2. Place `episode_design.csv` in `Pendulum_Grokking_Env/`.
3. Open the Unity project and press Play. `DataGenerator.cs` reads
   `episode_design.csv` and renders each episode (100 frames at 64x64 RGB)
   within the prescribed parameter sub-range, writing PNG frames and a
   `ground_truth.csv` to a `GeneratedDataV3/` directory.
4. Total render time: ~10 hours on a Mac M1 Pro; total raw size: ~2 GB of
   PNGs.

### 3.3 Raw -> HDF5 conversion

```bash
cd grokking-synthetic-physics
python prepare_hdf5.py --data-dir ../Pendulum_Grokking_Env/GeneratedDataV3 \
                       --output pendulum_data_v3.h5
```

`prepare_hdf5.py` walks the Unity output, packs frames + ground-truth metadata
into a single HDF5, and pre-computes the table of valid in-episode
`(t, t+1)` transitions so the dataloader is `O(1)` per item. The HDF5 stores:
`S_t`, `S_t_next`, `action` (delta_azimuth, delta_elevation in spherical
coords), `cam_pos_t`, `cam_pos_t_next`, the five physical fields
(`length`, `angle`, `gravity`, `damping`, `init_angular_velocity`), the
realised `angular_velocity`, and the `episode` id.

The packed HDF5 is ~14 GB because it stores uncompressed `(64, 64, 3)` arrays
for both `S_t` and `S_t_next` per transition.

### 3.4 Splits

Splits are defined entirely by `episode_design.csv` (column `split`):

| Split    | n_ood_dims | Episodes (default) |
|----------|------------|--------------------|
| IID      | 0          | 3,200              |
| Near-OOD | 1, 2       | 1,600              |
| Far-OOD  | 3, 4, 5    | 1,200              |

`HDF5PendulumDataset.make_splits()` returns four DataLoaders: train / IID-val
/ near-OOD / far-OOD, with an episode-level (not frame-level) train/val
split inside the IID episodes (default 80/20, seed 42). Each DataLoader
preloads its samples into RAM (`preload=True`, `num_workers=0`); we tried
multi-worker HDF5 access and it deadlocked intermittently with no throughput
win once the data was preloaded. Loader batch size is 64 (set inside
`make_splits` and overrides any `--batch_size` argument passed to `train.py`).

---

## 4. Training

### 4.1 Smoke tests (no data needed)

```bash
python models_conv.py     # ConvNet forward pass
python models.py          # ViT MAE forward pass
python models_dct.py      # DCT baseline forward pass
```

Each module's `__main__` block instantiates the model, runs a dummy batch,
and prints shapes — useful to confirm a CUDA / MPS / CPU install works
before downloading 14 GB of data.

### 4.2 Single training run

```bash
python train.py --model conv --latent_dim 32 --epochs 2000 \
                --save_dir runs/conv_dim32

python train.py --model vit  --latent_dim 128 --epochs 2000 \
                --save_dir runs/vit_dim128

python train.py --model dct  --latent_dim 64  --epochs 200 \
                --no_early_stop --save_dir runs/dct_dim64
```

Training loop (per epoch):

- Forward pass over the train loader, **Adam** optimizer (`lr=1e-4` default)
  with a cosine-annealing schedule over `--epochs`.
- Pixel-space MSE between predicted and true `S_{t+1}`.
- Evaluation on `iid_val`, `near_ood`, and `far_ood` loaders **every epoch**.
- All four MSEs + latent mean / std + lr written to `log.csv`.

Each run writes:

- `runs/<save_dir>/log.csv` — per-epoch train/iid_val/near_ood/far_ood loss,
  z_mean, z_std, lr.
- `runs/<save_dir>/checkpoint_epoch####.pt` — rolling periodic checkpoints
  (kept: last `--keep_checkpoints`, default 3).
- `runs/<save_dir>/images/recon_epoch####.png` — reconstruction grid
  (target vs predicted) at each saved checkpoint.
- `runs/<save_dir>/model_final.pt` — final weights.

Key training arguments:

```
--epochs 2000          maximum epochs (z_std early stop usually ends sooner)
--lr 1e-4              learning rate (Adam)
--batch_size 32        argparse default; ignored — make_splits uses 64
--resume <ckpt.pt>     resume from checkpoint (auto-detects crash vs extension)
--keep_checkpoints 3   rolling window of last N periodic checkpoints
--min_epochs 200       earliest epoch the z_std stop can trigger
--zstd_patience 50     window size for the convergence check
--zstd_threshold 0.01  relative range across the window (1%)
--no_early_stop        disable z_std early stop, run full --epochs
--save_every 50        checkpoint interval (epochs)
```

Resume modes (auto-detected from `--epochs` vs the original run):

- **Crash recovery** (`--epochs <= original`): restores optimizer + scheduler
  state exactly.
- **Extension** (`--epochs > original`): keeps model weights but starts a
  fresh cosine schedule over the remaining epochs (the original schedule
  has decayed lr to ~0).

The DCT baseline must use `--no_early_stop`: its encoder is fixed, so
`z_std` is constant from epoch 1 and would otherwise trigger the stop
immediately, starving the dynamics MLP of training.

### 4.3 Full sweep (paper results)

Seven widths × three architectures = 21 runs.

ConvNet sweep (manual launch, 7 GPUs):

```bash
for i in 0 1 2 3 4 5 6; do
  dims=(2 4 8 16 32 64 128)
  CUDA_VISIBLE_DEVICES=$i python train.py \
      --model conv --latent_dim ${dims[$i]} --epochs 2000 \
      --save_dir runs/conv_dim${dims[$i]} &
done
```

ViT sweep, auto-launched once the ConvNet PIDs all exit:

```bash
nohup bash launch_vit_after_conv.sh > logs/vit_launcher.log 2>&1 &
```

`launch_vit_after_conv.sh` polls a hard-coded list of ConvNet PIDs, waits
until they all exit, then launches the ViT sweep on the same 7 GPUs. Edit
the PID list and the conda-init path at the top of the script before reuse.

DCT sweep (no early stop, fewer epochs since the encoder is fixed):

```bash
for i in 0 1 2 3 4 5 6; do
  dims=(2 4 8 16 32 64 128)
  CUDA_VISIBLE_DEVICES=$i python train.py \
      --model dct --latent_dim ${dims[$i]} --epochs 200 --no_early_stop \
      --save_dir runs/dct_dim${dims[$i]} &
done
```

---

## 5. Evaluation

### 5.1 Reconstruction loss (already in log.csv)

Each `runs/<save_dir>/log.csv` already contains per-epoch IID-val, Near-OOD,
and Far-OOD MSE — written by `train.py` every epoch, no separate eval step
needed.

The headline metric reported in the paper is the Generalization Gap Ratio:

```
GGR = (MSE_FarOOD - MSE_IID) / MSE_IID
```

### 5.2 Linear probes

After training, freeze each encoder and probe what the latent encodes:

```bash
# ConvNet sweep only
python probe.py --model conv --sweep

# Both ConvNet + ViT
python probe.py --both --sweep

# All three (conv + vit + dct)
python probe.py --all --sweep

# Single model (debugging)
python probe.py --model conv --latent_dim 8 \
       --checkpoint runs/conv_dim8/model_final.pt
```

`probe.py` looks for checkpoints under `runs/<model>_dim<N>_v2/` first, then
`runs/<model>_dim<N>/`, then falls back to the latest `checkpoint_epoch*.pt`
in the matching directory. For each `(model, dim)`:

1. Run the encoder over all IID frames -> Z [N, latent_dim].
2. 80/20 split (seeded), fit Ridge regression `z_t -> v` per ground-truth
   variable `v` with `alpha=1.0` and `StandardScaler` normalisation.
3. Report R^2 on the held-out 20%, clipped to [0, 1].

Outputs:

- `probe_results/probe_results.csv` — wide-form table (one row per
  `(model, latent_dim)`, columns = R^2 per variable).
- `probe_results/probe_heatmap_<model>.svg` — annotated heatmap (rows =
  variables grouped by tier, cols = `latent_dim`).

Variables probed (eight total, three tiers):

- **Tier A** (geometric, single-frame visible): `length`, `angle`
- **Tier B** (color-coded): `gravity` (HSV hue), `damping` (HSV saturation)
- **Tier C** (latent or camera-related): `angular_velocity`, `init_ang_vel`,
  `cam_azimuth`, `cam_elevation` (the last two derived from `cam_pos_t`
  via `cartesian_to_spherical`)

We use linear (rather than MLP) probes deliberately: the question is
whether the encoder *itself* made the variable available, not whether some
more powerful classifier could in principle find it.

---

## 6. Project Evolution and Design Decisions

A handful of training-side choices were set on day one and never
revisited: Adam (`lr = 1e-4`), cosine annealing over `--epochs`,
pixel-space MSE between predicted and ground-truth `S_{t+1}`, and no KL
term — the bottleneck width is the regulariser, and adding a KL would
have confounded capacity with information rate. All decoders end in a
sigmoid so reconstructions live in `[0, 1]` and pixel MSE is comparable
across architectures with structurally different decoder bodies.

The rest of this section walks through the design areas that *changed*
during the project. Each subsection follows the same arc: what we tried
first, why, what went wrong, what replaced it.

### 6.1 Splits — defining IID / Near / Far OOD

The first split implementation (since removed; lived in
`split_dataset.py` and produced the v1/v2 datasets) sampled OOD bands
via a Latin-hypercube design. We started here because Latin-hypercube
is the standard recipe for spreading samples across a high-dimensional
band space without clumping.

It did not work. With five physical parameters and three bands per
parameter, only `(2/3)^5 ≈ 13%` of randomly drawn `(low, mid, high)^5`
combinations land entirely outside the IID band, and the Far-OOD region
(four or five dimensions out-of-band) sees fewer than 1% of episodes.
After stratification the Far-OOD bucket was nearly empty, and any
Far-OOD generalisation number measured on it would have been dominated
by sampling noise rather than by what the model had learned.

We replaced it with a deterministic band-stratified design
(`design_episodes.py`). Instead of sampling, we enumerate all
`3^5 = 243` `(low, mid, high)^5` combinations explicitly, classify each
combination by `n_ood_dims` (0 through 5 — the number of dimensions
falling in the holdout band), and assign episode counts per
`(combo, n_ood_dims)` cell so the IID / Near-OOD / Far-OOD totals
(3,200 / 1,600 / 1,200) are exact by construction. This required
re-rendering the entire dataset (`pendulum_data.h5` → `_v2.h5` →
`_v3.h5`), but the resulting splits are reproducible and statistically
balanced rather than emergent from sampling.

### 6.2 Architecture — from one encoder to three

The first version of the project carried only a ViT bottleneck encoder.
We started with the ViT because its inductive bias is the weakest of
the standard image encoders — there is no spatial equivariance baked
in — which made it the cleanest test of the hypothesis that compression
alone can drive generalisation. If a representation emerges, attribute
it to the bottleneck pressure rather than to a structural prior the
architecture brought for free.

The ConvNet was added as a baseline because we needed to know whether
the results we were seeing were ViT-specific. ConvNets bring strong
spatial inductive bias (translation equivariance, locality), so any
shared phenomenon between the two architectures could not be blamed on
attention specifically. The ConvNet's encoder produces a 4×4×256
spatial feature map then linearly projected into the bottleneck; the
ViT's encoder produces a single CLS token that plays the same role.

The DCT baseline was added last. With two learned encoders we could
compare *compression levels* but not separate the effect of *learned
compression* from *compression as a generic mechanism*. A fixed image
transform isolates that. The DCT model trains nothing in the encoder
or decoder — they are registered as buffers — so the only adapted
parameters are in the dynamics MLP. We chose DCT over PCA because PCA
components would have been fitted on the training data and would have
leaked distributional information into the Far-OOD test, and over JPEG
because JPEG quantises after the DCT, which would have made
`K`-coefficient counts incomparable across architectures at the same
nominal `latent_dim`. The implementation uses an orthonormal 2D DCT-II
per channel, takes the top `K` zigzag-ordered coefficients
(low-frequency first), and splits `K` across the three colour channels
via `divmod(K, 3)`.

Adding the third architecture forced one further decision: all three
models had to share a single forward signature
(`forward(s_t, action) → (pred_s_next, z_t)`) and a single dynamics MLP
(`[K + A] → 2K → K` with GELU). Without this, "ConvNet at K=32" and
"DCT at K=32" would not have referred to the same compression budget —
any difference between them could have been attributed to the dynamics
network rather than to the encoder.

A quiet bug belongs in this section. The ConvNet decoder ended in a
sigmoid from the start; the ViT decoder did not, until a fix part-way
through the project. For roughly two weeks, ViT pixel-MSE numbers were
not strictly comparable to ConvNet numbers because the two models'
output ranges differed. All numbers reported in the paper come from
runs after the fix — but the episode is a reminder that
cross-architecture comparisons silently break unless every output is
range-aligned.

### 6.3 Data loading — from streaming to preload

The initial dataloader followed standard PyTorch practice: open the
HDF5 file inside the `Dataset`, read each `(S_t, S_{t+1})` pair on
demand, spawn several DataLoader workers for I/O parallelism. We
expected this to be the right starting point because it works for most
disk-backed datasets.

It did not survive a 21-run sweep. With multiple workers reading the
same HDF5 file concurrently, we hit intermittent file-handle deadlocks
that froze training mid-epoch. We then noticed that even single-worker
HDF5 reads were slower than expected, because the `Dataset` was using
h5py's "fancy indexing" (passing a list of indices into the dataset
slice) which is internally O(N · indices) rather than a sequential
scan.

The fix was structural rather than incremental. The full dataset
(~14 GB, mostly uncompressed `64×64×3` arrays) fits in the memory of
the lab nodes, so we preload everything once at start-up and serve all
subsequent reads from RAM. The preload itself does a single sequential
pass over the HDF5 followed by a NumPy boolean mask — roughly an order
of magnitude faster than the index-based version. With everything in
RAM, multi-worker access loses its purpose, so we set `num_workers=0`
permanently. The cost is a longer warm-up at the start of each run;
the gain is no deadlocks and consistent epoch times. A side-effect of
this design is that `make_splits` hard-codes `batch_size=64`, so
`train.py --batch_size` is silently ignored.

### 6.4 Stopping criterion — from val-loss plateau to z_std

The original training loop stopped when validation loss stopped
improving — the textbook approach.

It was unreliable at both ends of the bottleneck sweep. At very small
`latent_dim` (2, 4) the loss decreases too slowly across many epochs,
so any patience setting that is short enough to be useful at large dim
will fire prematurely at small dim. At very large `latent_dim` (64,
128) the validation loss flattens early but the encoder continues to
refine the latent space — the visible probe results keep changing for
tens of epochs after the loss plateau. A single patience value cannot
cover both regimes.

We replaced the criterion with a check on `z_std` (the standard
deviation of `z_t` over the training set) rather than validation loss.
`z_std` turned out to be monotone in encoder "settledness" — it shrinks
while the encoder is still rearranging directions, and stabilises once
the representation has converged. The early-stop condition
(`zstd_converged` in `train.py`) asks whether the relative range of
`z_std` over the last `--zstd_patience` epochs has fallen below
`--zstd_threshold`. We added an `--no_early_stop` switch specifically
for the DCT baseline: its encoder is fixed, so `z_std` is constant
from epoch 1 and would otherwise trigger the stop on the very first
window.

Two adjacent pieces of training infrastructure fall under the same
"make a long sweep survivable" header. Rolling checkpoints keep the
last `N` periodic saves and delete older ones, so a multi-day sweep
does not fill the disk. Resume modes auto-detect crash recovery vs
extension: crash recovery (`--epochs ≤ original`) restores optimiser
and scheduler state exactly, while extension (`--epochs > original`)
keeps the weights but starts a fresh cosine over the remaining epochs.
Without the latter, naively resuming with a larger `--epochs` would
continue at the original cosine's terminal `lr ≈ 0` and accomplish
nothing.

### 6.5 What changed about the question itself

The thesis we set out to test was straightforward: smaller bottleneck →
smaller OOD generalisation gap → compression drives the formation of
generalisable representations. The early gross-MSE numbers seemed to
support this. We then added linear probes (`probe.py`) as a secondary
check, expecting them to confirm the story by showing physics
variables emerging in the latent at small bottleneck widths.

We chose **linear** probes deliberately. The question is whether the
encoder *itself* exposed each variable as a linear direction — whether
the representation contains the variable at all — not whether some
more powerful downstream model could in principle reconstruct it.
Ridge regression with `α = 1.0` on top of `StandardScaler`-normalised
latents, R² on a held-out 20% clipped to `[0, 1]`. An MLP probe would
be answering a different question.

The probe results inverted the framing. At small `latent_dim`, neither
the ConvNet nor the ViT encodes gravity. They encode `cam_elevation` —
camera pose — with R² ≈ 1 across the entire sweep. The "compression
drives generalisation" reading was wrong because the compressed latent
was not encoding the physical task at all; it was encoding a visual
shortcut from the action stream that happened to be predictive of the
next frame.

This is also where an earlier rendering choice came back to matter.
The unobservable variables — gravity and damping — are surfaced through
HSV colour on the bob (`hue = gravity_norm`,
`sat = lerp(0.3, 1.0, damping_norm)`, `value = 1.0`). The 0.3 floor on
saturation was added so the bob remains visually distinguishable
against the white background even at zero damping. That decision is
also what makes `cam_elevation` findable as a shortcut: gravity is
visible, but only as low-frequency colour content, while
`cam_elevation` directly determines where the bob *sits* in the image.
The encoder does not have to discover the physics; it just has to read
the action stream's effect on geometry.

The DCT baseline became central rather than a sanity check at this
point. A fixed encoder cannot pursue shortcuts — its representation
is whatever the DCT basis exposes. When we ran the same probes on
DCT, gravity appeared at `latent_dim = 8` (R² = 0.62) and stayed
strong throughout the sweep, while the learned encoders did not find
gravity until much larger widths (`dim = 32` for ConvNet, `dim = 128`
for ViT). Gravity is encoded as hue, which is literally low-frequency
pixel content; the DCT exposes it for free. The learned encoders have
to "re-discover" what the fixed transform already gives directly, and
they only get there once the bottleneck is wide enough to fit both
the cam_elevation shortcut and the physics target.

The paper's central claim shifted accordingly. Compression pressure
remains real and measurable, but compression on its own does not
point toward physics; the model takes whichever shortcut is cheapest
under its architecture. Producing rule-like representations requires
both enough capacity to encode the rule and an architecture whose
inductive bias makes the rule cheaper to encode than any available
shortcut. The DCT result is what made that argument force-able rather
than speculative.

---

## 7. Code Attribution

All Python and C# in this submission was written by the authors with the
following exceptions, all used unmodified:

- **Masked Autoencoder structural template** (in `models.py`): the
  encoder/decoder structure follows He et al., *Masked Autoencoders Are
  Scalable Vision Learners* (CVPR 2022), reimplemented from scratch in
  PyTorch (no patches are masked; the only bottleneck is the CLS-token
  projection).
- **Standard Python libraries** used unmodified: PyTorch, torchvision,
  scikit-learn (Ridge regression in `probe.py`), einops, h5py, NumPy,
  pandas, Pillow, tqdm, matplotlib. See Section 2.1 for installation.
- **Unity URP**: the submission contains only
  `Pendulum_Grokking_Env/Assets/DataGenerator.cs`, the single Unity
  asset authored by us. The rest of the URP project (scenes, settings,
  the pendulum prefab) is unmodified stock Unity 2022.3 LTS Universal
  Render Pipeline template and is not included in this submission;
  see Section 2.2 for what would be needed to recreate a runnable
  Unity project around `DataGenerator.cs`.
- **Claude Code** (Anthropic): used during development for boilerplate,
  refactoring, and review across the Python pipeline; all design and
  modeling decisions were made by the authors and every line was
  reviewed before commit. The full statement on AI use is in the paper.
