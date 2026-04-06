import torch
import torch.nn as nn


class ConvBottleneckAE(nn.Module):
    """
    Convolutional Autoencoder with tunable information bottleneck.
    Same forward interface as ContinuousBottleneckMAE:
        forward(s_t, action) -> (pred_s_next, z_t)

    Architecture:
        S_t -> Conv Encoder -> Z_t (bottleneck) -> fuse with A_t -> Conv Decoder -> S_{t+1}
    """
    def __init__(self,
                 img_size=64,
                 in_chans=3,
                 action_dim=2,
                 latent_dim=10,
                 base_channels=32):
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder = nn.Sequential(
            nn.Conv2d(in_chans, base_channels, 4, 2, 1),       # -> 32x32
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),  # -> 16x16
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),  # -> 8x8
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),  # -> 4x4
            nn.BatchNorm2d(base_channels * 8),
            nn.ReLU(inplace=True),
        )

        self.flatten_dim = base_channels * 8 * 4 * 4  # 256 * 16 = 4096

        # THE BOTTLENECK
        self.to_latent = nn.Linear(self.flatten_dim, latent_dim)

        # Dynamics: [Z_t, A_t] -> Z_{t+1}
        self.latent_dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

        # Decoder: expand latent back to spatial
        self.from_latent = nn.Linear(latent_dim, self.flatten_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 8, base_channels * 4, 4, 2, 1),  # -> 8x8
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2, 4, 2, 1),  # -> 16x16
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels * 2, base_channels, 4, 2, 1),      # -> 32x32
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_channels, in_chans, 4, 2, 1),               # -> 64x64
            nn.Sigmoid(),  # Output in [0, 1] to match ToTensor() normalization
        )

    def forward(self, s_t, action):
        """
        s_t: [B, 3, 64, 64]
        action: [B, 2]
        Returns:
            pred_s_next: [B, 3, 64, 64]
            z_t: [B, latent_dim]
        """
        B = s_t.shape[0]

        # Encode
        feat = self.encoder(s_t)                    # [B, 256, 4, 4]
        feat_flat = feat.view(B, -1)                # [B, 4096]
        z_t = self.to_latent(feat_flat)             # [B, latent_dim]

        # Dynamics
        z_action = torch.cat([z_t, action], dim=1)  # [B, latent_dim + 2]
        z_next = self.latent_dynamics(z_action)      # [B, latent_dim]

        # Decode
        dec_flat = self.from_latent(z_next)          # [B, 4096]
        dec_feat = dec_flat.view(B, -1, 4, 4)        # [B, 256, 4, 4]
        pred_s_next = self.decoder(dec_feat)         # [B, 3, 64, 64]

        return pred_s_next, z_t


if __name__ == "__main__":
    model = ConvBottleneckAE(latent_dim=10, action_dim=2)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    dummy_s_t = torch.randn(4, 3, 64, 64)
    dummy_action = torch.randn(4, 2)

    pred_s_next, z_t = model(dummy_s_t, dummy_action)
    print(f"Input:  {dummy_s_t.shape}")
    print(f"Latent: {z_t.shape}")
    print(f"Output: {pred_s_next.shape}")
    print(f"Output range: [{pred_s_next.min():.3f}, {pred_s_next.max():.3f}]")
