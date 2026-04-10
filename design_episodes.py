"""
design_episodes.py — Enumerate all 3^5 band combinations and assign episodes.

Instead of sampling uniformly from the full parameter space and post-hoc classifying,
this script pre-computes exactly which (IID / Near-OOD / Far-OOD) region each episode
belongs to, then Unity samples within the assigned sub-range.

Band partition (normalized [0,1]):
  Low  = [0.00, 0.33]
  Mid  = [0.33, 0.67]
  High = [0.67, 1.00]

Holdout band per dimension (episodes with this band are OOD for that dimension):
  D1 Length:   Mid
  D2 Angle:    High
  D3 Gravity:  Low
  D4 Damping:  Mid
  D5 AngVel:   High

OOD classification by n_ood_dims (number of dimensions in their holdout band):
  IID      : n_ood_dims == 0
  Near-OOD : n_ood_dims in {1, 2}
  Far-OOD  : n_ood_dims in {3, 4, 5}

Target episode counts (configurable):
  IID total      : ~2000  (32 combos  → ~63 eps/combo)
  Near-OOD total : ~500   (160 combos →  ~3 eps/combo)
  Far-OOD total  : ~600   (51 combos  → ~12 eps/combo)

Output: episode_design.csv
  episode_id, combo_id, n_ood_dims, split,
  length_lo, length_hi, angle_lo, angle_hi,
  gravity_lo, gravity_hi, damping_lo, damping_hi, angvel_lo, angvel_hi
"""

import itertools
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Physics parameter ranges — must match DataGenerator.cs Inspector values
# ---------------------------------------------------------------------------
PARAM_RANGES = {
    "Length":  (0.5,  2.0),
    "Angle":   (15.0, 85.0),
    "Gravity": (4.0,  14.0),
    "Damping": (0.01, 0.5),
    "AngVel":  (-3.0, 3.0),
}
PARAMS = list(PARAM_RANGES.keys())

# Band boundaries in normalized [0,1] space
BAND_EDGES = {
    "low":  (0.00, 0.33),
    "mid":  (0.33, 0.67),
    "high": (0.67, 1.00),
}

# Which band is the OOD holdout for each dimension
HOLDOUT = {
    "Length":  "mid",
    "Angle":   "high",
    "Gravity": "low",
    "Damping": "mid",
    "AngVel":  "high",
}

# Target total episodes per OOD level
TARGETS = {
    0: 3200,  # IID      — 32  combos → 100 eps/combo
    1:  800,  # k=1 OOD  — 80  combos →  10 eps/combo
    2:  800,  # k=2 OOD  — 80  combos →  10 eps/combo
    3:  600,  # k=3 OOD  — 40  combos →  15 eps/combo
    4:  300,  # k=4 OOD  — 10  combos →  30 eps/combo
    5:  300,  # k=5 OOD  —  1  combo  → 300 eps/combo
}

# ---------------------------------------------------------------------------

def denorm(norm_lo, norm_hi, param_lo, param_hi):
    lo = param_lo + norm_lo * (param_hi - param_lo)
    hi = param_lo + norm_hi * (param_hi - param_lo)
    return lo, hi


def enumerate_combos():
    """Return list of (combo_id, {param: band_name}, n_ood_dims)."""
    combos = []
    for combo_id, bands in enumerate(itertools.product(["low", "mid", "high"], repeat=5)):
        band_dict = dict(zip(PARAMS, bands))
        n_ood = sum(1 for p in PARAMS if band_dict[p] == HOLDOUT[p])
        combos.append((combo_id, band_dict, n_ood))
    return combos


def eps_per_combo(n_ood, n_combos_at_level):
    """Round-divide target episodes evenly across combos at this OOD level."""
    return max(1, round(TARGETS[n_ood] / n_combos_at_level))


def main(seed=42, output="episode_design.csv"):
    rng = np.random.default_rng(seed)

    combos = enumerate_combos()
    assert len(combos) == 3**5  # sanity

    # Count combos per OOD level
    from collections import Counter
    combo_counts = Counter(n for _, _, n in combos)

    print("=== Band combination counts ===")
    for k in sorted(combo_counts):
        print(f"  k={k}: {combo_counts[k]:3d} combos  →  "
              f"{eps_per_combo(k, combo_counts[k])} eps/combo  →  "
              f"~{eps_per_combo(k, combo_counts[k]) * combo_counts[k]} episodes")

    rows = []
    for combo_id, band_dict, n_ood in combos:
        n_eps = eps_per_combo(n_ood, combo_counts[n_ood])
        split = ("iid" if n_ood == 0
                 else "near_ood" if n_ood <= 2
                 else "far_ood")

        for _ in range(n_eps):
            row = {
                "combo_id":   combo_id,
                "n_ood_dims": n_ood,
                "split":      split,
            }
            for p in PARAMS:
                p_lo, p_hi = PARAM_RANGES[p]
                n_lo, n_hi = BAND_EDGES[band_dict[p]]
                actual_lo, actual_hi = denorm(n_lo, n_hi, p_lo, p_hi)
                key = p.lower()
                row[f"{key}_lo"] = round(actual_lo, 6)
                row[f"{key}_hi"] = round(actual_hi, 6)
            rows.append(row)

    df = pd.DataFrame(rows)

    # Shuffle so Unity sees random order (no OOD pattern by episode index)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    df.insert(0, "episode_id", range(len(df)))

    df.to_csv(output, index=False)

    print(f"\n=== Episode design summary ===")
    print(f"Total episodes : {len(df)}")
    print(f"Total frames   : {len(df) * 100:,}  (100 frames/episode)")
    summary = df.groupby(["n_ood_dims", "split"]).size().reset_index(name="episodes")
    print(summary.to_string(index=False))
    print(f"\nSaved to: {output}")

    # Verify parameter bounds look right
    print("\n=== Sample bounds check (first 3 IID episodes) ===")
    iid_sample = df[df["split"] == "iid"].head(3)
    print(iid_sample[["episode_id", "n_ood_dims",
                        "length_lo", "length_hi",
                        "gravity_lo", "gravity_hi"]].to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed",   type=int, default=42)
    parser.add_argument("--output", default="episode_design.csv")
    args = parser.parse_args()
    main(seed=args.seed, output=args.output)
