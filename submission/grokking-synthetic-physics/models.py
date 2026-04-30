import torch
import torch.nn as nn
from einops import rearrange

class PatchEmbed(nn.Module):
    """
    Convert 64x64 Image to Patches and project to embedding dimension.
    """
    def __init__(self, img_size=64, patch_size=8, in_chans=3, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # We use a non-overlapping convolutional layer to extract patches
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: [Batch, Channels, Height, Width]
        x = self.proj(x)                # [B, embed_dim, H/patch_size, W/patch_size]
        x = rearrange(x, 'b e h w -> b (h w) e') # [Batch, Num_Patches, Embed_Dim]
        return x

class ContinuousBottleneckMAE(nn.Module):
    """
    Masked Autoencoder tuned for Phase Transition in Physical Representation Learning.
    Architecture:
      St -> ViT Encoder -> [Latent Space Zt] -> fuse with At -> ViT Decoder -> S_{t+1}
    """
    def __init__(self, 
                 img_size=64, 
                 patch_size=8, 
                 in_chans=3,
                 action_dim=2,
                 latent_dim=10,         # THE STRICT BOTTLENECK 
                 embed_dim=128,         # Encoder internal dimension
                 depth=4,               # Encoder layers
                 num_heads=4, 
                 decoder_embed_dim=64,  # Decoder internal dimension
                 decoder_depth=2, 
                 decoder_num_heads=4):
        super().__init__()
        self.patch_size = patch_size

        # --------------------------------------------------------------------------
        # 1. PERCEPTION / ENCODER (Extracts physical properties)
        # --------------------------------------------------------------------------
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        
        # THE BOTTLENECK: Squash the entire CLS token into a tiny continuous vector Z_t
        self.to_latent = nn.Linear(embed_dim, latent_dim)

        # --------------------------------------------------------------------------
        # 2. DYNAMICS FUSION (Applies Physical Rules / Ego Motion)
        # --------------------------------------------------------------------------
        # We concatenate Z_t (state) and A_t (action) and predict the new latent state Z_{t+1}'
        self.latent_dynamics = nn.Sequential(
            nn.Linear(latent_dim + action_dim, latent_dim * 2),
            nn.GELU(),
            nn.Linear(latent_dim * 2, latent_dim)
        )

        # --------------------------------------------------------------------------
        # 3. RENDERING / DECODER (Generates next visual state S_{t+1})
        # --------------------------------------------------------------------------
        # Un-bottleneck: Expand the tiny Z_{t+1}' back into a full sequence of tokens for decoder
        self.to_decoder_tokens = nn.Linear(latent_dim, num_patches * decoder_embed_dim)
        
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim))
        
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim, nhead=decoder_num_heads, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)

        # Final projection to pixel space (Patch_size * Patch_size * Colors)
        self.head = nn.Linear(decoder_embed_dim, patch_size * patch_size * in_chans)

        self._initialize_weights()

    def _initialize_weights(self):
        # A simple initialization
        # Normally you would use sine-cosine pos embedding, but learned is fine for 64x64
        torch.nn.init.normal_(self.pos_embed, std=.02)
        torch.nn.init.normal_(self.decoder_pos_embed, std=.02)
        torch.nn.init.normal_(self.cls_token, std=.02)

    def forward(self, s_t, action):
        """
        s_t: [Batch, 3, 64, 64]
        action: [Batch, 2] (delta_theta, delta_phi)
        Returns:
            pred_s_next: [Batch, 3, 64, 64]
            z_t: [Batch, latent_dim] (For probing and topological analysis)
        """
        B = s_t.shape[0]

        # --- ENCODER ---
        x = self.patch_embed(s_t)               # [B, 64_Patches, Embed_Dim]
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, Embed_Dim]
        x = torch.cat((cls_tokens, x), dim=1)         # [B, 65, Embed_Dim]
        x = x + self.pos_embed                        # Add position
        
        x = self.encoder(x)                     # [B, 65, Embed_Dim]
        
        # Take the processed CLS token as the global state representation
        cls_out = x[:, 0, :]                    # [B, Embed_Dim]
        
        # ---> CONSTRICT TO BOTTLENECK Z_t <---
        z_t = self.to_latent(cls_out)           # [B, Latent_Dim (e.g. 10)]

        # --- DYNAMICS FUSION ---
        # Predict z_{t+1} strictly based on tiny physics vector and action
        z_action_concat = torch.cat([z_t, action], dim=1) # [B, Latent_Dim + Action_Dim]
        z_next_pred = self.latent_dynamics(z_action_concat) # [B, Latent_Dim]

        # --- DECODER ---
        # Expand the next latent state into decoder tokens (one for each spatial patch)
        dec_tokens = self.to_decoder_tokens(z_next_pred) # [B, Num_Patches * Decoder_Embed_dim]
        dec_tokens = rearrange(dec_tokens, 'b (n d) -> b n d', n=self.patch_embed.num_patches)
        
        dec_tokens = dec_tokens + self.decoder_pos_embed
        
        dec_out = self.decoder(dec_tokens)      # [B, 64_Patches, Decoder_Embed_dim]

        # Reconstruct image patches
        pred_patches = self.head(dec_out)       # [B, 64_Patches, patch_size * patch_size * 3]
        
        # Reshape patches back into [B, 3, 64, 64] image
        p = self.patch_size
        h = w = self.patch_embed.img_size // p
        pred_s_next = rearrange(pred_patches, 'b (h w) (p1 p2 c) -> b c (h p1) (w p2)',
                                h=h, w=w, p1=p, p2=p, c=3)
        pred_s_next = torch.sigmoid(pred_s_next)  # constrain to [0,1], consistent with ConvNet

        # Return both the prediction and the bottleneck representation
        # Returning Z_t allows you to compute metrics on its topology during training
        return pred_s_next, z_t


# ==========================================
# Test the Model Flow
# ==========================================
if __name__ == "__main__":
    model = ContinuousBottleneckMAE(
        img_size=64, 
        latent_dim=10,  # <-- The variable you will tune drastically to induce Grokking
        action_dim=2
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    # Generate dummy data mimicking a batch from PendulumDataset
    dummy_s_t = torch.randn(4, 3, 64, 64)       # Batch of 4 images
    dummy_action = torch.randn(4, 2)            # Batch of 4 camera deltas

    print(f"Input S_t shape: {dummy_s_t.shape}")
    print(f"Input Action shape: {dummy_action.shape}")

    # Forward pass
    pred_s_next, z_t = model(dummy_s_t, dummy_action)

    print(f"Latent Bottleneck Z_t shape: {z_t.shape}")
    print(f"Output predicted S_t+1 shape: {pred_s_next.shape}")
