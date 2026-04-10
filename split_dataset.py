"""
split_dataset.py — Stratified band partition for v2 5D physics experiment.

Reads ground_truth.csv (per-frame rows), computes per-episode physics params,
applies the Latin-hypercube band holdout, computes OOD scores, and outputs
episode_split.csv with columns: Episode, split, ood_score, n_ood_dims

Band partition (normalized to [0, 1]):
  D1 Length:       train = Low + High   holdout = Mid
  D2 Angle:        train = Low + Mid    holdout = High
  D3 Gravity:      train = Mid + High   holdout = Low
  D4 Damping:      train = Low + High   holdout = Mid
  D5 AngVelocity:  train = Low + Mid    holdout = High

OOD score = Σ d_i, where d_i = 0 if in train band, else distance to nearest train band edge.
Test sets:
  IID      : ood_score == 0
  Near-OOD : 0 < ood_score <= 0.5
  Far-OOD  : ood_score > 0.5
"""

import argparse
import numpy as np
import pandas as pd


# --- Physics parameter ranges (must match DataGenerator.cs) ---
PARAM_RANGES = {
    "Length":              (0.5, 2.0),
    "Angle":               (15.0, 85.0),
    "Gravity":             (4.0, 14.0),
    "Damping":             (0.01, 0.5),
    "InitAngularVelocity": (-3.0, 3.0),
}

# Band boundaries in normalized [0, 1] space
LOW_HI   = 0.33
MID_HI   = 0.67

# Latin-hypercube holdout band per dimension
# "low"  = [0, 0.33]   "mid" = [0.33, 0.67]   "high" = [0.67, 1.0]
HOLDOUT_BAND = {
    "Length":              "mid",
    "Angle":               "high",
    "Gravity":             "low",
    "Damping":             "mid",
    "InitAngularVelocity": "high",
}

BAND_EDGES = {
    "low":  (0.0,    LOW_HI),
    "mid":  (LOW_HI, MID_HI),
    "high": (MID_HI, 1.0),
}


def normalize(value, lo, hi):
    return (value - lo) / (hi - lo)


def ood_distance_for_dim(norm_val, holdout_band):
    """Return d_i for one dimension: 0 if in train band, else min-distance to train bands."""
    lo, hi = BAND_EDGES[holdout_band]
    if lo <= norm_val <= hi:
        # In holdout band — distance to nearest train-band edge
        return min(norm_val - lo, hi - norm_val)
    else:
        return 0.0  # In a train band


def compute_episode_params(df):
    """Aggregate per-episode physics params (constant within an episode, take first frame)."""
    ep_groups = df.groupby("Episode").first().reset_index()
    return ep_groups[["Episode"] + list(PARAM_RANGES.keys())]


def assign_split(ep_df, train_frac=2000/3000, iid_val_frac=500/3000,
                 near_ood_frac=250/3000, seed=42):
    """
    1. Compute normalized params and OOD scores.
    2. Classify each episode as IID or OOD.
    3. Sample split assignments:
         - train: from IID episodes
         - iid_val: from IID episodes  (not used for training)
         - near_ood: from OOD episodes with ood_score <= 0.5
         - far_ood: from OOD episodes with ood_score > 0.5
    """
    rng = np.random.default_rng(seed)
    n = len(ep_df)

    # Normalize each parameter
    norm_vals = {}
    for col, (lo, hi) in PARAM_RANGES.items():
        norm_vals[col] = ep_df[col].apply(lambda v: normalize(v, lo, hi)).values

    # Per-dimension OOD distance and flag
    ood_distances = np.zeros((n, len(PARAM_RANGES)))
    for i, col in enumerate(PARAM_RANGES):
        hband = HOLDOUT_BAND[col]
        ood_distances[:, i] = [ood_distance_for_dim(v, hband) for v in norm_vals[col]]

    ood_score = ood_distances.sum(axis=1)
    n_ood_dims = (ood_distances > 0).sum(axis=1)

    ep_df = ep_df.copy()
    ep_df["ood_score"] = ood_score
    ep_df["n_ood_dims"] = n_ood_dims

    # Separate IID vs OOD
    iid_mask = ood_score == 0.0
    near_mask = (ood_score > 0.0) & (ood_score <= 0.5)
    far_mask  = ood_score > 0.5

    iid_eps   = ep_df.loc[iid_mask, "Episode"].values
    near_eps  = ep_df.loc[near_mask, "Episode"].values
    far_eps   = ep_df.loc[far_mask, "Episode"].values

    print(f"IID episodes:      {len(iid_eps)}")
    print(f"Near-OOD episodes: {len(near_eps)}")
    print(f"Far-OOD episodes:  {len(far_eps)}")

    # Shuffle for random sampling
    rng.shuffle(iid_eps)
    rng.shuffle(near_eps)
    rng.shuffle(far_eps)

    # IID -> train + iid_val
    n_iid_val = min(500, len(iid_eps) // 5)  # up to 500, at most 20% of IID
    iid_val_set = set(iid_eps[:n_iid_val])
    train_set   = set(iid_eps[n_iid_val:])

    # OOD -> near + far
    near_ood_set = set(near_eps)
    far_ood_set  = set(far_eps)

    def get_split(ep_id):
        if ep_id in train_set:    return "train"
        if ep_id in iid_val_set:  return "iid_val"
        if ep_id in near_ood_set: return "near_ood"
        if ep_id in far_ood_set:  return "far_ood"
        return "unknown"

    ep_df["split"] = ep_df["Episode"].apply(get_split)
    return ep_df[["Episode", "split", "ood_score", "n_ood_dims"]]


def main():
    parser = argparse.ArgumentParser(description="Stratified band split for v2 5D experiment")
    parser.add_argument("--csv", default="ground_truth.csv",
                        help="Path to ground_truth.csv from Unity data generation")
    parser.add_argument("--output", default="episode_split.csv",
                        help="Output CSV with per-episode split assignments")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Reading {args.csv} ...")
    df = pd.read_csv(args.csv)

    # Validate columns
    required = ["Episode", "Frame"] + list(PARAM_RANGES.keys())
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}\nFound: {list(df.columns)}")

    ep_df = compute_episode_params(df)
    print(f"Total episodes: {len(ep_df)}")

    split_df = assign_split(ep_df, seed=args.seed)

    # Summary
    counts = split_df["split"].value_counts()
    print("\n=== Split Summary ===")
    for s in ["train", "iid_val", "near_ood", "far_ood"]:
        print(f"  {s:12s}: {counts.get(s, 0)} episodes")

    split_df.to_csv(args.output, index=False)
    print(f"\nSaved to {args.output}")

    # OOD score distribution
    print("\n=== OOD Score Stats ===")
    print(split_df.groupby("split")["ood_score"].describe())


if __name__ == "__main__":
    main()
