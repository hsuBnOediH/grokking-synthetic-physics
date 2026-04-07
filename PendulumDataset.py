import os
import argparse
import math
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class PendulumDataset(Dataset):
    DEFAULT_DATA_DIR = "/Users/fengli/Documents/BAM/CompressionProject/Pendulum_Grokking_Env/GeneratedData"

    def __init__(self, data_dir=None, transform=None):
        """
        Args:
            data_dir (string, optional): Directory with all images and ground_truth.csv.
                If not provided, uses env var PENDULUM_DATA_DIR, then DEFAULT_DATA_DIR.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        resolved_data_dir = data_dir or os.getenv("PENDULUM_DATA_DIR") or self.DEFAULT_DATA_DIR
        self.data_dir = os.path.expanduser(resolved_data_dir)
        self.csv_path = os.path.join(self.data_dir, "ground_truth.csv")

        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        if not os.path.isfile(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        
        # Load CSV data
        self.df = pd.read_csv(self.csv_path)
        
        # Default transforms: Convert PIL Image to PyTorch Tensor and normalize to [0, 1]
        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor()
            ])
        else:
            self.transform = transform

        # Build valid transition pairs: (S_t, S_t+1)
        self.valid_transitions = self._build_transition_index()

    def _build_transition_index(self):
        """
        Filters the dataframe to only include valid t -> t+1 transitions.
        Ensures we don't cross episode boundaries.
        """
        valid_indices = []
        for i in range(len(self.df) - 1):
            current_row = self.df.iloc[i]
            next_row = self.df.iloc[i + 1]
            
            # Check if both frames belong to the SAME episode
            if current_row['Episode'] == next_row['Episode']:
                # Ensure they are sequential frames
                if next_row['Frame'] == current_row['Frame'] + 1:
                    valid_indices.append(i)
                    
        print(f"Loaded {len(valid_indices)} valid (S_t, S_t+1) transition pairs.")
        return valid_indices

    def _cartesian_to_spherical(self, pos):
        """
        Converts 3D Cartesian coordinates to Spherical coordinates (azimuth, elevation).
        Assuming Unity Left-Handed Coordinate System: Y is Up, X is Right, Z is Forward.
        Returns: [theta (azimuth in XZ), phi (elevation from Y)]
        """
        x, y, z = pos[0], pos[1], pos[2]
        r = torch.norm(pos) + 1e-8 # Add small epsilon to avoid division by zero
        
        # Azimuth angle in XZ plane (-pi to pi)
        theta = torch.atan2(x, z) 
        # Elevation angle from Y axis (0 to pi)
        phi = torch.acos(torch.clamp(y / r, -1.0, 1.0))
        
        return torch.stack([theta, phi])

    def __len__(self):
        return len(self.valid_transitions)

    def __getitem__(self, idx):
        # Get the row index for S_t
        row_idx = self.valid_transitions[idx]
        row_t = self.df.iloc[row_idx]
        row_t_next = self.df.iloc[row_idx + 1]

        # 1. Load Images (S_t and S_t+1)
        img_name_t = f"ep{int(row_t['Episode'])}_frame{int(row_t['Frame'])}.png"
        img_path_t = os.path.join(self.data_dir, img_name_t)
        image_t = Image.open(img_path_t).convert("RGB")

        img_name_t_next = f"ep{int(row_t_next['Episode'])}_frame{int(row_t_next['Frame'])}.png"
        img_path_t_next = os.path.join(self.data_dir, img_name_t_next)
        image_t_next = Image.open(img_path_t_next).convert("RGB")

        if self.transform:
            image_t = self.transform(image_t)
            image_t_next = self.transform(image_t_next)

        # 2. Extract Action (Camera Ego-motion)
        # Convert absolute camera position to delta spherical angles
        cam_pos_t = torch.tensor([row_t['Camera_X'], row_t['Camera_Y'], row_t['Camera_Z']], dtype=torch.float32)
        cam_pos_t_next = torch.tensor([row_t_next['Camera_X'], row_t_next['Camera_Y'], row_t_next['Camera_Z']], dtype=torch.float32)
        
        sphere_t = self._cartesian_to_spherical(cam_pos_t)
        sphere_t_next = self._cartesian_to_spherical(cam_pos_t_next)
        
        # Action is the delta: [delta_azimuth, delta_elevation]
        # Wrap delta_azimuth to [-pi, pi] to handle crossover
        action_t = sphere_t_next - sphere_t
        action_t[0] = (action_t[0] + torch.pi) % (2 * torch.pi) - torch.pi

        damping = torch.tensor([row_t['Damping']], dtype=torch.float32)
        angle = torch.tensor([row_t['Angle']], dtype=torch.float32)
        angular_velocity = torch.tensor([row_t['AngularVelocity']], dtype=torch.float32)

        return {
            "S_t": image_t,
            "S_t_next": image_t_next,
            "action": action_t,
            "cam_pos_t": cam_pos_t,
            "cam_pos_t_next": cam_pos_t_next,
            "damping": damping,
            "angle": angle,
            "angular_velocity": angular_velocity,
        }

# ==========================================
# Test the DataLoader if run directly
# ==========================================
if __name__ == "__main__":
    from torch.utils.data import DataLoader

    parser = argparse.ArgumentParser(description="Quick data loader check for pendulum dataset")
    parser.add_argument(
        "--data-dir",
        default=None,
        help=(
            "Path to dataset folder. If omitted, uses $PENDULUM_DATA_DIR, "
            f"then {PendulumDataset.DEFAULT_DATA_DIR}."
        ),
    )
    args = parser.parse_args()

    dataset = PendulumDataset(data_dir=args.data_dir)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Fetch one batch
    batch = next(iter(dataloader))
    
    print("\n--- Dataloader Test ---")
    print(f"S_t shape: {batch['S_t'].shape}")           # Should be [4, 3, 64, 64]
    print(f"S_t_next shape: {batch['S_t_next'].shape}") # Should be [4, 3, 64, 64]
    print(f"Action shape: {batch['action'].shape}")     # Should be [4, 2]
    print(f"First action sample (d_theta, d_phi): {batch['action'][0].tolist()}")
    print(f"Damping shape: {batch['damping'].shape}")   # Should be [4, 1]