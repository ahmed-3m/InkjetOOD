from __future__ import annotations
"""
trainer.py
==========
Training loop for the Conditional Diffusion Model.

Loss Function
-------------
Two terms are combined:

1. **Denoising MSE** (standard DDPM objective):

       L_denoise = E_{t, ε}[ ||ε − ε_θ(x_t, t, c)||² ]

2. **Class Separation Loss** (contribution of this thesis):

       L_sep = −E_{t, ε}[ ||ε_θ(x_t, t, c=GOOD) − ε_θ(x_t, t, c=BAD)||² ]

   By *maximising* the distance between the GOOD-conditioned and BAD-conditioned
   noise predictions on the **same noisy image**, the model is pushed to produce
   meaningfully different reconstructions for different quality conditions.
   This directly increases the classification score gap:

       score = E_t[L_denoise(GOOD)] − E_t[L_denoise(BAD)]

   making AUROC improve, especially for class-imbalanced features (e.g. `angle`)
   and visually subtle features (e.g. `e.rought2`).

   The separation loss weight λ is the primary ablation hyperparameter studied
   in the thesis (Chapter 5/6 ablations).  A value of λ=0.01 was found optimal
   for the inkjet dataset; λ=0.02 is optimal for CIFAR-10.

Total loss:
       L = L_denoise + λ · L_sep
"""

from pathlib import Path
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_cdm(
    model,
    schedule,
    train_loader: DataLoader,
    epochs: int = 100,
    lr: float = 2e-4,
    sep_loss_weight: float = 0.01,
    device: str = 'cuda',
    save_path: Optional[Union[str, Path]] = None,
    log_interval: int = 1,
) -> list[dict]:
    """
    Train the CDM with denoising MSE + class separation loss.

    Parameters
    ----------
    model            : NoisePredictorV3 instance
    schedule         : DiffusionSchedule instance
    train_loader     : DataLoader yielding batches from InkjetCDMDataset
    epochs           : number of training epochs
    lr               : initial learning rate (AdamW, cosine-annealed)
    sep_loss_weight  : λ weighting the class separation auxiliary loss
    device           : 'cuda' or 'cpu'
    save_path        : path for saving the best checkpoint
    log_interval     : print epoch summary every N epochs

    Returns
    -------
    history : list of per-epoch metric dicts
    """
    model.train()
    optimizer  = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    lr_sched   = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    best_loss = float('inf')
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        running_mse = 0.0
        running_sep = 0.0
        running_tot = 0.0
        n_batches   = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        for batch in pbar:
            images      = batch['image'].to(device)
            template_id = batch['template_id'].to(device)
            feature_id  = batch['feature_id'].to(device)
            quality     = batch['quality'].to(device)
            bbox        = batch['bbox'].to(device)

            # ── sample random timestep ──────────────────────────────────
            t = torch.randint(0, schedule.num_timesteps, (images.size(0),), device=device)
            x_t, noise = schedule.q_sample(images, t)

            # ── denoising loss (standard DDPM) ──────────────────────────
            noise_pred = model(x_t, t, template_id, feature_id, quality, bbox)
            mse_loss   = F.mse_loss(noise_pred, noise)

            # ── class separation loss ───────────────────────────────────
            # Only computed when λ > 0.  When λ=0 (baseline) we skip the
            # two extra forward passes — saving 2/3 VRAM and run time.
            if sep_loss_weight > 0.0:
                q_good    = torch.zeros_like(quality)
                q_bad     = torch.ones_like(quality)
                pred_good = model(x_t.detach(), t, template_id, feature_id, q_good, bbox)
                pred_bad  = model(x_t.detach(), t, template_id, feature_id, q_bad,  bbox)
                sep_loss  = -F.mse_loss(pred_good, pred_bad)
                total_loss = mse_loss + sep_loss_weight * sep_loss
            else:
                sep_loss   = torch.tensor(0.0, device=device)
                total_loss = mse_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            running_mse += mse_loss.item()
            running_sep += sep_loss.item()
            running_tot += total_loss.item()
            n_batches   += 1
            pbar.set_postfix({'total': f'{total_loss.item():.5f}',
                              'mse':   f'{mse_loss.item():.5f}',
                              'sep':   f'{sep_loss.item():.5f}'})

        lr_sched.step()

        avg_mse = running_mse / n_batches
        avg_sep = running_sep / n_batches
        avg_tot = running_tot / n_batches

        record = {
            'epoch': epoch,
            'loss_total': avg_tot,
            'loss_mse'  : avg_mse,
            'loss_sep'  : avg_sep,
        }
        history.append(record)

        if epoch % log_interval == 0:
            print(
                f"Epoch {epoch:>3}/{epochs}  |  "
                f"total={avg_tot:.6f}  mse={avg_mse:.6f}  sep={avg_sep:.6f}  "
                f"lr={lr_sched.get_last_lr()[0]:.2e}"
            )

        # ── checkpoint ────────────────────────────────────────────────
        if save_path is not None and avg_tot < best_loss:
            best_loss = avg_tot
            torch.save(
                {
                    'epoch'             : epoch,
                    'model_state_dict'  : model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss'              : avg_tot,
                    'history'           : history,
                    'config'            : {
                        'sep_loss_weight': sep_loss_weight,
                        'lr'             : lr,
                        'epochs'         : epochs,
                    },
                },
                save_path,
            )
            print(f"  ✓ Saved checkpoint  (loss={avg_tot:.6f})")

    return history
