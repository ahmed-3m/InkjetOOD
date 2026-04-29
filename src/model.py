"""
model.py
========
Conditional Diffusion Model (CDM) for Inkjet Print Quality Control.

Architecture: UNet with Multi-Head Conditioning
    - Timestep embedding  (sinusoidal)
    - Template embedding  (A / B / C  →  0 / 1 / 2)
    - Feature embedding   (8 feature types)
    - Quality embedding   (GOOD=0, BAD=1)
    - BBox embedding      (x_c, y_c, w, h encoded via MLP)

Classification is performed by comparing per-class noise-prediction error
(Algorithm 1 from the thesis):

    score(x) = E_t[ ||ε - ε_θ(x_t, t, c=GOOD)||² ]
             - E_t[ ||ε - ε_θ(x_t, t, c=BAD )||² ]

    score > 0  →  defect (OOD / BAD)
    score < 0  →  in-distribution (GOOD)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sinusoidal timestep embedding  (Vaswani et al., 2017)
# ---------------------------------------------------------------------------

class SinusoidalPosEmb(nn.Module):
    """Maps a scalar timestep to a fixed-size sinusoidal embedding."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = math.log(10000) / (half - 1)
        freqs = torch.exp(torch.arange(half, device=device) * -freqs)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Bounding-box position encoder
# ---------------------------------------------------------------------------

class BBoxEncoder(nn.Module):
    """
    Encodes normalised bounding-box coordinates (x_c, y_c, w, h) into a
    dense embedding of the same size as the timestep embedding so that the
    two signals can be added together before injection into each UNet block.
    """

    def __init__(self, embed_dim: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(4, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim),
        )

    def forward(self, bbox: torch.Tensor) -> torch.Tensor:
        """bbox: (B, 4) with values in [0, 1]."""
        return self.mlp(bbox)


# ---------------------------------------------------------------------------
# Conditioned convolutional block
# ---------------------------------------------------------------------------

class ConvBlock(nn.Module):
    """
    Residual convolutional block with full multi-head conditioning.

    Conditioning signals (timestep, template, feature, quality, bbox) are
    projected to the channel dimension and added to the feature map after
    the first normalisation layer — a standard FiLM-style injection.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        time_dim: int,
        num_templates: int = 3,
        num_features: int = 8,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        # Conditioning projections  (all map to out_ch so they can be summed)
        self.time_proj     = nn.Linear(time_dim, out_ch)
        self.template_emb  = nn.Embedding(num_templates, out_ch)
        self.feature_emb   = nn.Embedding(num_features, out_ch)
        self.quality_emb   = nn.Embedding(2, out_ch)
        self.bbox_proj     = nn.Linear(time_dim, out_ch)

        self.residual = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(
        self,
        x: torch.Tensor,
        t_emb: torch.Tensor,
        template_id: torch.Tensor,
        feature_id: torch.Tensor,
        quality: torch.Tensor,
        bbox_emb: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(self.conv1(x))

        # Aggregate all conditioning signals
        cond = (
            self.time_proj(t_emb)
            + self.template_emb(template_id)
            + self.feature_emb(feature_id)
            + self.quality_emb(quality)
            + self.bbox_proj(bbox_emb)
        )
        h = h + cond[:, :, None, None]   # broadcast over H, W
        h = F.silu(h)
        h = F.silu(self.norm2(self.conv2(h)))
        return h + self.residual(x)


# ---------------------------------------------------------------------------
# Noise predictor  (UNet backbone)
# ---------------------------------------------------------------------------

class NoisePredictorV3(nn.Module):
    """
    UNet that predicts the noise ε added at timestep t, conditioned on:
        (template, feature-type, quality-label, bounding-box, timestep).

    Parameters
    ----------
    in_channels  : number of image channels (3 for RGB)
    base_channels: width multiplier; 64 gives ~9.3 M parameters
    num_templates: number of template classes (A / B / C → 3)
    num_features : number of feature types (8 YOLO classes)
    time_dim     : dimension of the sinusoidal timestep embedding
    """

    def __init__(
        self,
        in_channels: int  = 3,
        base_channels: int = 64,
        num_templates: int = 3,
        num_features: int  = 8,
        time_dim: int      = 256,
    ):
        super().__init__()

        bc = base_channels

        # Timestep MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_dim),
            nn.Linear(time_dim, time_dim * 4),
            nn.GELU(),
            nn.Linear(time_dim * 4, time_dim),
        )

        # BBox encoder
        self.bbox_encoder = BBoxEncoder(time_dim)

        # Encoder (3 stages + pooling)
        self.enc1 = ConvBlock(in_channels, bc,     time_dim, num_templates, num_features)
        self.enc2 = ConvBlock(bc,          bc * 2, time_dim, num_templates, num_features)
        self.enc3 = ConvBlock(bc * 2,      bc * 4, time_dim, num_templates, num_features)
        self.pool = nn.MaxPool2d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(bc * 4, bc * 8, time_dim, num_templates, num_features)

        # Decoder (3 stages + upsampling)
        self.up3  = nn.ConvTranspose2d(bc * 8, bc * 4, 2, stride=2)
        self.dec3 = ConvBlock(bc * 8, bc * 4, time_dim, num_templates, num_features)
        self.up2  = nn.ConvTranspose2d(bc * 4, bc * 2, 2, stride=2)
        self.dec2 = ConvBlock(bc * 4, bc * 2, time_dim, num_templates, num_features)
        self.up1  = nn.ConvTranspose2d(bc * 2, bc,     2, stride=2)
        self.dec1 = ConvBlock(bc * 2, bc,     time_dim, num_templates, num_features)

        self.out  = nn.Conv2d(bc, in_channels, 1)

    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        template_id: torch.Tensor,
        feature_id: torch.Tensor,
        quality: torch.Tensor,
        bbox: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        x           : (B, C, H, W)  noisy image x_t
        t           : (B,)          diffusion timestep
        template_id : (B,)          template index  {0, 1, 2}
        feature_id  : (B,)          feature index   {0 … 7}
        quality     : (B,)          quality label   {0=GOOD, 1=BAD}
        bbox        : (B, 4)        normalised bbox (x_c, y_c, w, h)

        Returns
        -------
        (B, C, H, W)  predicted noise ε̂
        """
        t_emb    = self.time_mlp(t)
        bbox_emb = self.bbox_encoder(bbox)

        # Encoder
        e1 = self.enc1(x,             t_emb, template_id, feature_id, quality, bbox_emb)
        e2 = self.enc2(self.pool(e1), t_emb, template_id, feature_id, quality, bbox_emb)
        e3 = self.enc3(self.pool(e2), t_emb, template_id, feature_id, quality, bbox_emb)

        # Bottleneck
        bn = self.bottleneck(self.pool(e3), t_emb, template_id, feature_id, quality, bbox_emb)

        # Decoder (skip connections)
        d3 = self.dec3(torch.cat([self.up3(bn), e3], 1), t_emb, template_id, feature_id, quality, bbox_emb)
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1), t_emb, template_id, feature_id, quality, bbox_emb)
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1), t_emb, template_id, feature_id, quality, bbox_emb)

        return self.out(d1)

    # ------------------------------------------------------------------
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
