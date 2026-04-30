"""
models_dct.py — DCT-based world model (fixed encoder/decoder, learned dynamics only).

Architecture:
  S_t → 2D DCT per channel → top-K coefficients (zigzag order) → z_t
  z_t + A_t → dynamics MLP → z_{t+1}'
  z_{t+1}' → zero-pad → 2D inverse DCT → pred_S_{t+1}

Only the dynamics MLP is trained. Encoder and decoder are fixed linear transforms.

This isolates the effect of compression *mechanism* (learned vs hand-crafted DCT)
vs. compression *level* (latent_dim K), enabling direct comparison with ConvNet/ViT.
"""

import math
import torch
import torch.nn as nn


def _make_dct_matrix(N: int) -> torch.Tensor:
    """Return the orthonormal DCT-II matrix of size [N, N].

    D[k, n] = w_k * cos(pi * k * (2n+1) / (2N))
    where w_0 = 1/sqrt(N), w_k = sqrt(2/N) for k >= 1.
    """
    n = torch.arange(N, dtype=torch.float64)
    k = torch.arange(N, dtype=torch.float64).unsqueeze(1)   # [N, 1]
    mat = torch.cos(math.pi * k * (2.0 * n + 1.0) / (2.0 * N))  # [N, N]
    mat[0] *= 1.0 / math.sqrt(N)
    mat[1:] *= math.sqrt(2.0 / N)
    return mat.float()


def _make_zigzag(N: int) -> list:
    """Return flat indices in zigzag order for an N×N matrix (low-freq first)."""
    indices = []
    for d in range(2 * N - 1):
        if d % 2 == 0:
            r, c = min(d, N - 1), max(0, d - (N - 1))
            while r >= 0 and c < N:
                indices.append(r * N + c)
                r -= 1
                c += 1
        else:
            r, c = max(0, d - (N - 1)), min(d, N - 1)
            while r < N and c >= 0:
                indices.append(r * N + c)
                r += 1
                c -= 1
    return indices


class DCTBottleneckAE(nn.Module):
    """
    DCT world model with tunable bottleneck.

    Same forward interface as ConvBottleneckAE / ContinuousBottleneckMAE:
        forward(s_t, action) → (pred_s_next, z_t)

    Encoder/decoder are *fixed* (no gradients). Only dynamics MLP is trained.
    latent_dim K coefficients are split as evenly as possible across in_chans channels.
    """

    def __init__(self,
                 img_size: int = 64,
                 in_chans: int = 3,
                 action_dim: int = 2,
                 latent_dim: int = 32):
        super().__init__()
        self.img_size = img_size
        self.in_chans = in_chans
        self.latent_dim = latent_dim

        # ── Fixed DCT basis ───────────────────────────────────────────────
        dct_mat = _make_dct_matrix(img_size)           # [N, N]
        self.register_buffer('dct_mat', dct_mat)

        # ── Zigzag index map (per-channel flat indices, low-freq first) ───
        zigzag = _make_zigzag(img_size)                # list of N*N ints
        self.register_buffer('zigzag_idx',
                             torch.tensor(zigzag, dtype=torch.long))

        # How many coefficients each channel contributes
        # e.g. K=8, C=3 → [3, 3, 2]
        base, rem = divmod(latent_dim, in_chans)
        self.k_per_chan = [base + (1 if i < rem else 0) for i in range(in_chans)]

        # ── Dynamics MLP (the only learned part) ─────────────────────────
        self.latent_dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim),
        )

    # ── DCT helpers ───────────────────────────────────────────────────────

    def _dct2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D DCT-II.  x: [B, C, H, W] → [B, C, H, W]"""
        # Apply along W:  x @ D.T
        x = x @ self.dct_mat.T
        # Apply along H:  D @ x
        x = self.dct_mat @ x
        return x

    def _idct2d(self, x: torch.Tensor) -> torch.Tensor:
        """2D inverse DCT-II.  x: [B, C, H, W] → [B, C, H, W]
        D is orthonormal ⇒ D^{-1} = D.T, so:
          IDCT along H: D.T @ x
          IDCT along W: x @ D
        """
        x = self.dct_mat.T @ x
        x = x @ self.dct_mat
        return x

    # ── Encode / decode ───────────────────────────────────────────────────

    def _encode(self, s_t: torch.Tensor) -> torch.Tensor:
        """[B, C, H, W] → [B, latent_dim]"""
        B = s_t.shape[0]
        dct = self._dct2d(s_t)                              # [B, C, H, W]
        flat = dct.reshape(B, self.in_chans, -1)            # [B, C, H*W]
        flat_zz = flat[:, :, self.zigzag_idx]               # [B, C, H*W] (zigzag order)
        parts = [flat_zz[:, c, :self.k_per_chan[c]]
                 for c in range(self.in_chans)]
        return torch.cat(parts, dim=1)                      # [B, latent_dim]

    def _decode(self, z: torch.Tensor) -> torch.Tensor:
        """[B, latent_dim] → [B, C, H, W]"""
        B = z.shape[0]
        N = self.img_size
        dct_flat = torch.zeros(B, self.in_chans, N * N,
                               device=z.device, dtype=z.dtype)
        offset = 0
        for c in range(self.in_chans):
            k = self.k_per_chan[c]
            if k > 0:
                dct_flat[:, c, self.zigzag_idx[:k]] = z[:, offset:offset + k]
            offset += k
        dct = dct_flat.reshape(B, self.in_chans, N, N)
        img = self._idct2d(dct)
        return torch.sigmoid(img)                           # → [0, 1]

    # ── Forward ───────────────────────────────────────────────────────────

    def forward(self, s_t: torch.Tensor, action: torch.Tensor):
        """
        s_t    : [B, 3, 64, 64]
        action : [B, 2]
        Returns:
            pred_s_next : [B, 3, 64, 64]
            z_t         : [B, latent_dim]
        """
        z_t    = self._encode(s_t)                                   # [B, K]
        z_next = self.latent_dynamics(torch.cat([z_t, action], dim=1))  # [B, K]
        pred   = self._decode(z_next)                                # [B, 3, H, W]
        return pred, z_t


# ── Smoke test ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=== DCTBottleneckAE smoke test ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    for latent_dim in [2, 4, 8, 16, 32, 64, 128]:
        model = DCTBottleneckAE(latent_dim=latent_dim).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\nlatent_dim={latent_dim:3d}  |  k_per_chan={model.k_per_chan}"
              f"  |  trainable params: {n_params:,}")

        s_t    = torch.rand(4, 3, 64, 64, device=device)
        action = torch.randn(4, 2, device=device)
        pred, z = model(s_t, action)

        # Reconstruction sanity: encode → decode (no dynamics)
        z_enc    = model._encode(s_t)
        recon    = model._decode(z_enc)
        psnr_db  = -10 * torch.log10(((s_t - recon) ** 2).mean()).item()

        print(f"  z_t: {tuple(z.shape)}  pred: {tuple(pred.shape)}"
              f"  pred_range: [{pred.min():.3f}, {pred.max():.3f}]"
              f"  recon_psnr: {psnr_db:.1f} dB")

        assert pred.shape == (4, 3, 64, 64), "output shape mismatch"
        assert z.shape == (4, latent_dim),   "latent shape mismatch"
        assert pred.min() >= 0.0 and pred.max() <= 1.0, "output out of [0,1]"

    print("\nAll checks passed.")
