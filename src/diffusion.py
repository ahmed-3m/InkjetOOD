from __future__ import annotations
"""
diffusion.py
============
Gaussian diffusion schedule utilities.

Implements the forward (noising) process:

    q(x_t | x_0) = N(x_t; √ᾱ_t · x_0,  (1 − ᾱ_t) · I)

and helper functions needed at training and inference time.

Two schedules are supported:
  - 'linear'   : β_t linearly spaced between β_start and β_end  (DDPM default)
  - 'cosine'   : ᾱ_t follows a cosine curve (improved DDPM)

The cosine schedule is the default used in this thesis because it provides a
smoother noise progression and maintains signal longer at early timesteps,
which benefits classification of subtle texture defects.
"""

import math
from typing import Optional
import torch


class DiffusionSchedule:
    """
    Pre-computes all derived quantities (alphas, sigma, etc.) for the
    requested noise schedule and exposes a single `q_sample` method used
    during both training and the classification scoring loop.
    """

    def __init__(
        self,
        num_timesteps: int = 1000,
        schedule: str = 'cosine',
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        device: str = 'cuda',
    ):
        """
        Parameters
        ----------
        num_timesteps : total diffusion steps T
        schedule      : 'linear' or 'cosine'
        beta_start    : β_1  (only for linear schedule)
        beta_end      : β_T  (only for linear schedule)
        device        : torch device string
        """
        self.num_timesteps = num_timesteps
        self.device = device

        if schedule == 'cosine':
            betas = self._cosine_betas(num_timesteps)
        else:
            betas = torch.linspace(beta_start, beta_end, num_timesteps)

        betas = betas.to(device)
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)

        # Pre-compute sqrt terms used in q_sample
        self.betas                    = betas
        self.alphas                   = alphas
        self.alpha_bar                = alpha_bar
        self.sqrt_alpha_bar           = alpha_bar.sqrt()
        self.sqrt_one_minus_alpha_bar = (1.0 - alpha_bar).sqrt()

    # ------------------------------------------------------------------
    @staticmethod
    def _cosine_betas(num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        """
        Cosine beta schedule from Nichol & Dhariwal (2021).
        Ensures ᾱ_t follows a cosine curve from ~1 to ~0.
        """
        steps = num_timesteps + 1
        t = torch.linspace(0, num_timesteps, steps)
        f = torch.cos(((t / num_timesteps) + s) / (1 + s) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.clamp(0.0001, 0.9999)

    # ------------------------------------------------------------------
    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ):
        """
        Forward diffusion: x_t = √ᾱ_t · x_0 + √(1 − ᾱ_t) · ε

        Parameters
        ----------
        x_0   : (B, C, H, W)  clean image
        t     : (B,)           timestep indices
        noise : optional pre-sampled noise; sampled fresh if None

        Returns
        -------
        x_t   : (B, C, H, W)  noisy image at timestep t
        noise : (B, C, H, W)  noise that was added (needed for the loss)
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ab   = self.sqrt_alpha_bar[t].view(-1, 1, 1, 1)
        sqrt_1mab = self.sqrt_one_minus_alpha_bar[t].view(-1, 1, 1, 1)
        x_t = sqrt_ab * x_0 + sqrt_1mab * noise
        return x_t, noise
