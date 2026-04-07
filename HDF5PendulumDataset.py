import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class HDF5PendulumDataset(Dataset):
    def __init__(self, h5_path="pendulum_data.h5", transform=None):
        """
        Loads the pendulum dataset directly from a compiled HDF5 file.
        This provides massive I/O speedups suitable for Phase 2 training.
        """
        self.h5_path = h5_path
        
        # We don't keep the file open continuously to avoid multi-processing DataLoader issues.
        # Instead, we open it in the first __getitem__ call (per worker).
        self.file = None
        
        # To get the length, we open it once briefly
        with h5py.File(self.h5_path, 'r') as f:
            self.length = f['S_t'].shape[0]

        if transform is None:
            # We assume images are uint8 numpy arrays [H, W, C]
            self.transform = transforms.Compose([
                transforms.ToTensor() # Converts to [C, H, W] and normalizes [0, 1]
            ])
        else:
            self.transform = transform

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if self.file is None:
            self.file = h5py.File(self.h5_path, 'r')
            
        # 1. Read Raw Tensors
        # HDF5 returns numpy arrays here
        img_t_np = self.file['S_t'][idx]
        img_t_next_np = self.file['S_t_next'][idx]
        
        # 2. Applies transformation
        # HDF5 read returns shape [64, 64, 3], ToTensor expects this numpy shape
        s_t = self.transform(img_t_np)
        s_t_next = self.transform(img_t_next_np)

        # 3. Read metadata
        action_t = torch.tensor(self.file['action'][idx], dtype=torch.float32)
        cam_pos_t = torch.tensor(self.file['cam_pos_t'][idx], dtype=torch.float32)
        cam_pos_t_next = torch.tensor(self.file['cam_pos_t_next'][idx], dtype=torch.float32)
        damping = torch.tensor(self.file['damping'][idx], dtype=torch.float32)
        angle = torch.tensor(self.file['angle'][idx], dtype=torch.float32)
        angular_velocity = torch.tensor(self.file['angular_velocity'][idx], dtype=torch.float32)

        return {
            "S_t": s_t,
            "S_t_next": s_t_next,
            "action": action_t,
            "cam_pos_t": cam_pos_t,
            "cam_pos_t_next": cam_pos_t_next,
            "damping": damping,
            "angle": angle,
            "angular_velocity": angular_velocity,
        }

    def close(self):
        if self.file is not None:
            self.file.close()
            self.file = None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5-path", default="pendulum_data.h5", help="Path to HDF5 dataset")
    args = parser.parse_args()

    dataset = HDF5PendulumDataset(h5_path=args.h5_path)
    # Testing num_workers>0 to ensure our JIT h5py opening works across processes
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2)
    
    print(f"HDF5 dataset initialized with length: {len(dataset)}")
    
    batch = next(iter(dataloader))
    
    print("\n--- Fast HDF5 Dataloader Test ---")
    print(f"S_t shape: {batch['S_t'].shape} (min={batch['S_t'].min():.2f}, max={batch['S_t'].max():.2f})")
    print(f"Action shape: {batch['action'].shape}")
    print(f"First action sample (d_theta, d_phi): {batch['action'][0].tolist()}")
