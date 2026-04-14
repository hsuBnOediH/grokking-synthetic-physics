"""
probe.py — Linear probe analysis for the compression spectrum experiment.

For each trained model: freeze encoder, extract z_t over all IID samples,
fit Ridge regression z_t → each GT variable, report R².

Key question: is dim=2's low GGR "true rule extraction" or "uniform failure"?
- High R² at small dim → encoder found physics rules
- Low R²  at small dim → encoder is just confused (uniform failure)

Usage
-----
# Sweep all available ConvNet dims
python probe.py --model conv --sweep

# Sweep both conv and vit
python probe.py --sweep

# Single model
python probe.py --model conv --latent_dim 8 \
    --checkpoint runs/conv_dim8_v2/model_final.pt
"""

import argparse
import os
import math
import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from HDF5PendulumDataset import HDF5PendulumDataset
from models_conv import ConvBottleneckAE
from models import ContinuousBottleneckMAE

try:
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import r2_score as sk_r2
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[warn] sklearn not found — falling back to numpy OLS (no regularization)")


# ── Ground-truth variable registry ────────────────────────────────────
# (probe_name, hdf5_field_or_derived, tier_label)
# Tier A = geometric / single-frame visible
# Tier B = color-coded (not direct geometry)
# Tier C = purely latent (invisible from single frame)
GT_VARS = [
    ("gravity",          "gravity",              "B"),
    ("damping",          "damping",              "B"),
    ("length",           "length",               "A"),
    ("angle",            "angle",                "A"),
    ("angular_velocity", "angular_velocity",     "C"),
    ("init_ang_vel",     "init_angular_velocity","C"),
    ("cam_azimuth",      "_derived_",            "C"),
    ("cam_elevation",    "_derived_",            "C"),
]
GT_NAMES = [n for n, _, _ in GT_VARS]


# ── Encoder helpers ────────────────────────────────────────────────────

@torch.no_grad()
def encode(model, model_type, s_t):
    """Run encoder only — skips dynamics and decoder for speed."""
    if model_type == "conv":
        feat = model.encoder(s_t)                      # [B, C, 4, 4]
        return model.to_latent(feat.view(s_t.shape[0], -1))  # [B, dim]
    else:  # vit
        B = s_t.shape[0]
        x = model.patch_embed(s_t)
        cls = model.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1) + model.pos_embed
        x = model.encoder(x)
        return model.to_latent(x[:, 0, :])            # CLS token → z_t


def cam_to_spherical(cam_xyz):
    """cam_pos_t [N,3] → (azimuth [N], elevation [N])."""
    x, y, z = cam_xyz[:, 0], cam_xyz[:, 1], cam_xyz[:, 2]
    r = np.sqrt(x**2 + y**2 + z**2) + 1e-8
    az = np.arctan2(x, z)
    el = np.arccos(np.clip(y / r, -1.0, 1.0))
    return az, el


# ── Data collection ────────────────────────────────────────────────────

def collect_z_and_gt(model, model_type, loader, device):
    """
    Run encoder over all batches. Returns:
        Z  : np.ndarray [N, latent_dim]
        gt : dict {probe_name: np.ndarray [N]}
    """
    model.eval()
    z_list, gt_lists = [], {n: [] for n in GT_NAMES}
    cam_list = []

    for batch in loader:
        s_t = batch["S_t"].to(device)
        z = encode(model, model_type, s_t)
        z_list.append(z.cpu().numpy())

        for name, field, _ in GT_VARS:
            if field == "_derived_":
                continue
            gt_lists[name].append(batch[field].numpy().squeeze())
        cam_list.append(batch["cam_pos_t"].numpy())

    Z = np.concatenate(z_list, axis=0)

    cam = np.concatenate(cam_list, axis=0)
    az, el = cam_to_spherical(cam)
    gt_lists["cam_azimuth"]   = [az]
    gt_lists["cam_elevation"] = [el]

    gt = {n: np.concatenate(gt_lists[n], axis=0) for n in GT_NAMES}
    return Z, gt


# ── Linear probe ──────────────────────────────────────────────────────

def _standardize(arr, mean=None, std=None):
    if mean is None:
        mean, std = arr.mean(0), arr.std(0) + 1e-8
    return (arr - mean) / std, mean, std


def ridge_r2(Z_tr, y_tr, Z_te, y_te, alpha=1.0):
    """Ridge regression, return R² on test set (clipped to [0, 1])."""
    if HAS_SKLEARN:
        sz = StandardScaler().fit(Z_tr)
        sy = StandardScaler().fit(y_tr.reshape(-1, 1))
        Ztr = sz.transform(Z_tr)
        ytr = sy.transform(y_tr.reshape(-1, 1)).ravel()
        Zte = sz.transform(Z_te)
        yte = sy.transform(y_te.reshape(-1, 1)).ravel()
        pred = Ridge(alpha=alpha).fit(Ztr, ytr).predict(Zte)
        r2 = float(sk_r2(yte, pred))
    else:
        Ztr, zm, zs = _standardize(Z_tr)
        ytr, ym, ys = _standardize(y_tr)
        Zte, _, _   = _standardize(Z_te, zm, zs)
        yte, _, _   = _standardize(y_te, ym, ys)
        w, _, _, _  = np.linalg.lstsq(Ztr, ytr, rcond=None)
        pred = Zte @ w
        ss_res = ((yte - pred)**2).sum()
        ss_tot = ((yte - yte.mean())**2).sum() + 1e-10
        r2 = float(1.0 - ss_res / ss_tot)

    return max(0.0, min(1.0, r2))


def probe_model(model, model_type, latent_dim, loader, device, test_frac=0.2, seed=0):
    """Fit probes for all GT vars. Returns dict {var_name: R²}."""
    print(f"  Extracting z_t ({len(loader.dataset)} samples)...", flush=True)
    Z, gt = collect_z_and_gt(model, model_type, loader, device)
    print(f"  Z: {Z.shape}", flush=True)

    rng  = np.random.default_rng(seed)
    idx  = rng.permutation(len(Z))
    n_te = max(1, int(len(Z) * test_frac))
    tr, te = idx[n_te:], idx[:n_te]

    Z_tr, Z_te = Z[tr], Z[te]
    results = {}
    for name, _, tier in GT_VARS:
        y = gt[name]
        r2 = ridge_r2(Z_tr, y[tr], Z_te, y[te])
        results[name] = r2
        print(f"    {name:22s} [tier {tier}]  R² = {r2:.4f}", flush=True)

    return results


# ── Checkpoint discovery ───────────────────────────────────────────────

def find_checkpoints(runs_dir, model_type, dims):
    """Return list of (dim, path) for available model_final.pt files."""
    found = []
    for dim in dims:
        # Try suffixed dirs first (_v2), then plain
        for suffix in ("_v2", ""):
            run = os.path.join(runs_dir, f"{model_type}_dim{dim}{suffix}")
            final = os.path.join(run, "model_final.pt")
            if os.path.exists(final):
                found.append((dim, final))
                break
            # Fallback: latest checkpoint
            ckpts = sorted([
                f for f in os.listdir(run)
                if f.startswith("checkpoint_epoch") and f.endswith(".pt")
            ]) if os.path.isdir(run) else []
            if ckpts:
                found.append((dim, os.path.join(run, ckpts[-1])))
                print(f"  [info] dim={dim}: using latest ckpt ({ckpts[-1]})")
                break
        else:
            print(f"  [warn] No checkpoint for {model_type} dim={dim} in {runs_dir}")
    return found


def load_model(model_type, latent_dim, ckpt_path, device):
    if model_type == "conv":
        model = ConvBottleneckAE(latent_dim=latent_dim, action_dim=2)
    else:
        model = ContinuousBottleneckMAE(latent_dim=latent_dim, action_dim=2)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    return model.to(device).eval()


# ── Heatmap plot ──────────────────────────────────────────────────────

def plot_heatmap(df_model, model_type, output_path):
    dims = sorted(df_model["latent_dim"].tolist())
    tiers = {n: t for n, _, t in GT_VARS}

    mat = np.full((len(GT_NAMES), len(dims)), np.nan)
    for j, dim in enumerate(dims):
        row = df_model[df_model["latent_dim"] == dim].iloc[0]
        for i, name in enumerate(GT_NAMES):
            mat[i, j] = row.get(name, np.nan)

    fig, ax = plt.subplots(figsize=(max(7, len(dims) * 1.1 + 1), 4.5))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1,
                   cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="R²", fraction=0.03, pad=0.02)

    ax.set_xticks(range(len(dims)))
    ax.set_xticklabels([str(d) for d in dims], fontsize=10)
    ax.set_yticks(range(len(GT_NAMES)))
    ax.set_yticklabels(
        [f"{n}  [tier {tiers[n]}]" for n in GT_NAMES], fontsize=9)
    ax.set_xlabel("latent_dim", fontsize=11)
    ax.set_title(f"Linear Probe R²  —  {model_type.upper()}", fontsize=12)

    # Draw tier separator lines
    tier_boundaries = []
    prev = GT_VARS[0][2]
    for i, (_, _, t) in enumerate(GT_VARS):
        if t != prev:
            tier_boundaries.append(i - 0.5)
            prev = t
    for yb in tier_boundaries:
        ax.axhline(yb, color="white", linewidth=1.5, linestyle="--")

    # Annotate cells
    for i in range(len(GT_NAMES)):
        for j in range(len(dims)):
            v = mat[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8.5,
                        color="white" if v > 0.65 else "black")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Heatmap saved: {output_path}", flush=True)


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Linear probe for compression spectrum")
    parser.add_argument("--model",  default="conv", choices=["conv", "vit"])
    parser.add_argument("--latent_dim", type=int, default=None,
                        help="Required for single-model mode")
    parser.add_argument("--checkpoint", default=None,
                        help="Path to model_final.pt (single-model mode)")
    parser.add_argument("--sweep", action="store_true",
                        help="Sweep all available dims for --model")
    parser.add_argument("--both",  action="store_true",
                        help="Sweep both conv and vit")
    parser.add_argument("--dims",  nargs="+", type=int,
                        default=[2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--runs_dir",   default="runs")
    parser.add_argument("--h5_path",    default="pendulum_data_v3.h5")
    parser.add_argument("--design_csv", default="episode_design.csv")
    parser.add_argument("--output_dir", default="probe_results")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--seed",       type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else
                          "mps"  if torch.backends.mps.is_available() else "cpu")
    print(f"Device : {device}")
    print(f"sklearn: {HAS_SKLEARN}")

    # ── Build probe dataset (all IID episodes) ─────────────────────────
    print("\nLoading IID probe dataset...", flush=True)
    df_design = pd.read_csv(args.design_csv)
    iid_eps = set(df_design.loc[df_design["split"] == "iid", "episode_id"].tolist())
    ds = HDF5PendulumDataset(args.h5_path, episode_ids=iid_eps, preload=True)
    loader = DataLoader(ds, batch_size=args.batch_size,
                        shuffle=False, num_workers=0, pin_memory=False)
    print(f"IID transitions: {len(ds)}", flush=True)

    # ── Determine what to run ──────────────────────────────────────────
    if args.both:
        model_types = ["conv", "vit"]
        sweep_dims  = {mt: find_checkpoints(args.runs_dir, mt, args.dims)
                       for mt in model_types}
    elif args.sweep:
        model_types = [args.model]
        sweep_dims  = {args.model: find_checkpoints(args.runs_dir, args.model, args.dims)}
    else:
        assert args.checkpoint and args.latent_dim, \
            "Single-model mode requires --checkpoint and --latent_dim"
        model_types = [args.model]
        sweep_dims  = {args.model: [(args.latent_dim, args.checkpoint)]}

    # ── Run probes ─────────────────────────────────────────────────────
    all_records = []
    for mt in model_types:
        for dim, ckpt in sweep_dims[mt]:
            print(f"\n{'='*60}")
            print(f"Model: {mt}  latent_dim={dim}  |  {ckpt}")
            model  = load_model(mt, dim, ckpt, device)
            r2s    = probe_model(model, mt, dim, loader, device, seed=args.seed)
            record = {"model": mt, "latent_dim": dim, **r2s}
            all_records.append(record)
            del model
            torch.cuda.empty_cache() if device.type == "cuda" else None

    # ── Save CSV ───────────────────────────────────────────────────────
    results = pd.DataFrame(all_records)
    csv_path = os.path.join(args.output_dir, "probe_results.csv")
    results.to_csv(csv_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved: {csv_path}")
    print(results.to_string(index=False))

    # ── Heatmap per model type ─────────────────────────────────────────
    for mt in results["model"].unique():
        sub = results[results["model"] == mt].sort_values("latent_dim")
        if len(sub) >= 2:
            hm_path = os.path.join(args.output_dir, f"probe_heatmap_{mt}.png")
            plot_heatmap(sub, mt, hm_path)


if __name__ == "__main__":
    main()
