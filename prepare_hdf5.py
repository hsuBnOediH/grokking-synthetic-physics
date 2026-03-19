import os
import argparse
import h5py
import numpy as np
from tqdm import tqdm
import pandas as pd
from PIL import Image
import torch
import math

def cartesian_to_spherical(x, y, z):
    r = math.sqrt(x**2 + y**2 + z**2) + 1e-8
    theta = math.atan2(x, z)
    phi = math.acos(max(-1.0, min(1.0, y / r)))
    return theta, phi

def convert_to_hdf5(data_dir, output_file):
    csv_path = os.path.join(data_dir, "ground_truth.csv")
    df = pd.read_csv(csv_path)
    
    # Pre-calculate valid transitions like in PendulumDataset
    valid_indices = []
    for i in range(len(df) - 1):
        if df.iloc[i]['Episode'] == df.iloc[i + 1]['Episode'] and \
           df.iloc[i + 1]['Frame'] == df.iloc[i]['Frame'] + 1:
            valid_indices.append(i)
            
    num_samples = len(valid_indices)
    print(f"Found {num_samples} valid transition pairs. Converting to HDF5...")
    
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        s_t_ds = f.create_dataset('S_t', shape=(num_samples, 64, 64, 3), dtype=np.uint8, chunks=True)
        s_t_next_ds = f.create_dataset('S_t_next', shape=(num_samples, 64, 64, 3), dtype=np.uint8, chunks=True)
        action_ds = f.create_dataset('action', shape=(num_samples, 2), dtype=np.float32)
        cam_pos_t_ds = f.create_dataset('cam_pos_t', shape=(num_samples, 3), dtype=np.float32)
        cam_pos_t_next_ds = f.create_dataset('cam_pos_t_next', shape=(num_samples, 3), dtype=np.float32)
        damping_ds = f.create_dataset('damping', shape=(num_samples, 1), dtype=np.float32)
        angle_ds = f.create_dataset('angle', shape=(num_samples, 1), dtype=np.float32)

        for out_idx, row_idx in enumerate(tqdm(valid_indices)):
            row_t = df.iloc[row_idx]
            row_t_next = df.iloc[row_idx + 1]

            # 1. Images
            img_name_t = f"ep{int(row_t['Episode'])}_frame{int(row_t['Frame'])}.png"
            img_path_t = os.path.join(data_dir, img_name_t)
            img_t = np.array(Image.open(img_path_t).convert("RGB"))

            img_name_t_next = f"ep{int(row_t_next['Episode'])}_frame{int(row_t_next['Frame'])}.png"
            img_path_t_next = os.path.join(data_dir, img_name_t_next)
            img_t_next = np.array(Image.open(img_path_t_next).convert("RGB"))

            # 2. Camera Ego-Motion
            ctx, cty, ctz = row_t['Camera_X'], row_t['Camera_Y'], row_t['Camera_Z']
            cnxtx, cnxty, cnxtz = row_t_next['Camera_X'], row_t_next['Camera_Y'], row_t_next['Camera_Z']
            
            theta_t, phi_t = cartesian_to_spherical(ctx, cty, ctz)
            theta_next, phi_next = cartesian_to_spherical(cnxtx, cnxty, cnxtz)
            
            delta_theta = theta_next - theta_t
            delta_theta = (delta_theta + math.pi) % (2 * math.pi) - math.pi
            delta_phi = phi_next - phi_t
            
            # Store in HDF5
            s_t_ds[out_idx] = img_t
            s_t_next_ds[out_idx] = img_t_next
            action_ds[out_idx] = np.array([delta_theta, delta_phi], dtype=np.float32)
            cam_pos_t_ds[out_idx] = np.array([ctx, cty, ctz], dtype=np.float32)
            cam_pos_t_next_ds[out_idx] = np.array([cnxtx, cnxty, cnxtz], dtype=np.float32)
            damping_ds[out_idx] = np.array([row_t['Damping']], dtype=np.float32)
            angle_ds[out_idx] = np.array([row_t['InitAngle']], dtype=np.float32)
            
    print(f"\nConversion complete! Saved to {output_file}")


if __name__ == "__main__":
    from PendulumDataset import PendulumDataset
    parser = argparse.ArgumentParser(description="Convert Pendulum dataset to HDF5")
    parser.add_argument("--data-dir", default=PendulumDataset.DEFAULT_DATA_DIR, help="Path to raw dataset")
    parser.add_argument("--output", default="pendulum_data.h5", help="Output HDF5 file path")
    args = parser.parse_args()

    convert_to_hdf5(args.data_dir, args.output)
