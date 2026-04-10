import h5py
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class HDF5PendulumDataset(Dataset):
    """
    Fast HDF5 dataset for the V3 pendulum compression experiment.

    Filtering by train/eval split is done via episode_design.csv
    (produced by design_episodes.py). The HDF5 stores an 'episode' field
    per transition; we build a filtered index array at init time.

    Args:
        h5_path        : Path to the HDF5 file (produced by prepare_hdf5.py)
        design_csv     : Path to episode_design.csv. If None, all transitions used.
        splits         : List of split labels to include, e.g. ["iid"].
                         Valid labels: "iid", "near_ood", "far_ood".
                         For train vs. iid_val, pass episode_ids directly.
        episode_ids    : Explicit set/list of episode IDs to include.
                         Takes precedence over splits if both provided.
        transform      : torchvision transform (default: ToTensor)
    """

    def __init__(self, h5_path, design_csv=None, splits=None,
                 episode_ids=None, transform=None):
        self.h5_path = h5_path
        self.file = None  # opened lazily per worker

        # Build set of allowed episode IDs
        allowed_episodes = None
        if episode_ids is not None:
            allowed_episodes = set(int(e) for e in episode_ids)
        elif design_csv is not None and splits is not None:
            df = pd.read_csv(design_csv)
            mask = df["split"].isin(splits)
            allowed_episodes = set(df.loc[mask, "episode_id"].tolist())

        # Build index array
        with h5py.File(self.h5_path, 'r') as f:
            total = f['S_t'].shape[0]
            if allowed_episodes is None:
                self.indices = np.arange(total, dtype=np.int64)
            else:
                ep_arr = f['episode'][:]  # shape (N,)
                self.indices = np.array(
                    [i for i in range(total) if int(ep_arr[i]) in allowed_episodes],
                    dtype=np.int64
                )

        if transform is None:
            self.transform = transforms.Compose([transforms.ToTensor()])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')

        i = int(self.indices[idx])
        s_t      = self.transform(self.file['S_t'][i])
        s_t_next = self.transform(self.file['S_t_next'][i])

        return {
            "S_t":                  s_t,
            "S_t_next":             s_t_next,
            "action":               torch.tensor(self.file['action'][i],               dtype=torch.float32),
            "cam_pos_t":            torch.tensor(self.file['cam_pos_t'][i],            dtype=torch.float32),
            "cam_pos_t_next":       torch.tensor(self.file['cam_pos_t_next'][i],       dtype=torch.float32),
            "damping":              torch.tensor(self.file['damping'][i],              dtype=torch.float32),
            "gravity":              torch.tensor(self.file['gravity'][i],              dtype=torch.float32),
            "length":               torch.tensor(self.file['length'][i],              dtype=torch.float32),
            "init_angular_velocity":torch.tensor(self.file['init_angular_velocity'][i],dtype=torch.float32),
            "angle":                torch.tensor(self.file['angle'][i],               dtype=torch.float32),
            "angular_velocity":     torch.tensor(self.file['angular_velocity'][i],    dtype=torch.float32),
            "episode":              int(self.file['episode'][i]),
        }

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


def make_splits(h5_path, design_csv, train_frac=0.8, seed=42):
    """
    Build four DataLoaders (train, iid_val, near_ood, far_ood) from HDF5 + design CSV.

    IID episodes are randomly split train/val using train_frac.

    Returns dict: {"train": DataLoader, "iid_val": DataLoader,
                   "near_ood": DataLoader, "far_ood": DataLoader}
    """
    df = pd.read_csv(design_csv)

    rng = np.random.default_rng(seed)
    iid_eps = df.loc[df["split"] == "iid", "episode_id"].values.copy()
    rng.shuffle(iid_eps)
    n_train = int(len(iid_eps) * train_frac)
    train_eps = set(iid_eps[:n_train].tolist())
    val_eps   = set(iid_eps[n_train:].tolist())

    near_eps = set(df.loc[df["split"] == "near_ood", "episode_id"].tolist())
    far_eps  = set(df.loc[df["split"] == "far_ood",  "episode_id"].tolist())

    def loader(ep_ids, shuffle):
        ds = HDF5PendulumDataset(h5_path, episode_ids=ep_ids)
        return DataLoader(ds, batch_size=64, shuffle=shuffle, num_workers=4, pin_memory=True)

    return {
        "train":    loader(train_eps,  shuffle=True),
        "iid_val":  loader(val_eps,    shuffle=False),
        "near_ood": loader(near_eps,   shuffle=False),
        "far_ood":  loader(far_eps,    shuffle=False),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path",    default="pendulum_data_v3.h5")
    parser.add_argument("--design-csv", default="episode_design.csv")
    args = parser.parse_args()

    print("=== Testing full split pipeline ===")
    loaders = make_splits(args.h5_path, args.design_csv)
    for name, dl in loaders.items():
        batch = next(iter(dl))
        print(f"{name:10s}: {len(dl.dataset):6d} transitions | "
              f"S_t {tuple(batch['S_t'].shape)} | "
              f"gravity range [{batch['gravity'].min():.2f}, {batch['gravity'].max():.2f}]")
