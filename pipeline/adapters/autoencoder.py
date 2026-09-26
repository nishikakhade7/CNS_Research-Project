"""Autoencoder adapter — scaled input, train-only-fit learned latent representation.

Uses CUDA automatically when `torch.cuda.is_available()` is True on this machine (verified,
not assumed); otherwise runs on CPU with the reason logged by the caller via
`pipeline.core.environment`. Trained exclusively on the TRAIN split — val/test/AndMal are
only ever passed through `.encode()` on the already-fitted weights, never used to update them.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


class Autoencoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], latent_dim: int):
        super().__init__()
        enc_layers = []
        d = input_dim
        for h in hidden_dims:
            enc_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        enc_layers += [nn.Linear(d, latent_dim)]
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers = []
        d = latent_dim
        for h in reversed(hidden_dims):
            dec_layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        dec_layers += [nn.Linear(d, input_dim)]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z


def train_autoencoder(X_train: np.ndarray, cfg: dict, log) -> tuple[Autoencoder, torch.device]:
    ae_cfg = cfg["models"]["autoencoder"]
    device_pref = ae_cfg["device"]
    device = torch.device("cuda" if (device_pref == "auto" and torch.cuda.is_available()) or device_pref == "cuda" else "cpu")
    if device.type == "cuda":
        log.info(f"Autoencoder training on CUDA: {torch.cuda.get_device_name(0)}")
    else:
        reason = "torch.cuda.is_available() is False" if device_pref == "auto" else "device explicitly set to cpu in config"
        log.info(f"Autoencoder training on CPU ({reason}) — not pretending this is GPU execution.")

    torch.manual_seed(cfg["seed"])
    model = Autoencoder(X_train.shape[1], ae_cfg["hidden_dims"], ae_cfg["latent_dim"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=ae_cfg["lr"])
    loss_fn = nn.MSELoss()

    X_t = torch.tensor(X_train, dtype=torch.float32, device=device)
    n = X_t.shape[0]
    batch_size = ae_cfg["batch_size"]

    model.train()
    for epoch in range(ae_cfg["epochs"]):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            batch = X_t[idx]
            opt.zero_grad()
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"  autoencoder epoch {epoch+1}/{ae_cfg['epochs']}: train MSE={total_loss/n:.5f}")

    model.eval()
    return model, device


def encode(model: Autoencoder, device: torch.device, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the already-fitted encoder — never re-fits. Returns (latent, reconstruction_error)."""
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
        recon, z = model(X_t)
        err = ((recon - X_t) ** 2).mean(dim=1)
        return z.cpu().numpy(), err.cpu().numpy()
