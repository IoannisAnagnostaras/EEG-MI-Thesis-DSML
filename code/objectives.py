from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("Expected square matrix.")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)
        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)
        similarity = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        similarity = similarity.masked_fill(mask, float("-inf"))
        labels = torch.arange(batch_size, 2 * batch_size, device=z.device)
        labels = torch.cat([labels, torch.arange(batch_size, device=z.device)])
        return F.cross_entropy(similarity, labels)


class VICRegLoss(nn.Module):
    def __init__(self, sim_coeff: float, var_coeff: float, cov_coeff: float) -> None:
        super().__init__()
        self.sim_coeff = sim_coeff
        self.var_coeff = var_coeff
        self.cov_coeff = cov_coeff

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        repr_loss = F.mse_loss(z1, z2)

        z1 = z1 - z1.mean(dim=0)
        z2 = z2 - z2.mean(dim=0)

        std_z1 = torch.sqrt(z1.var(dim=0) + 1e-4)
        std_z2 = torch.sqrt(z2.var(dim=0) + 1e-4)
        std_loss = torch.mean(F.relu(1 - std_z1)) + torch.mean(F.relu(1 - std_z2))

        cov_z1 = (z1.T @ z1) / (z1.shape[0] - 1)
        cov_z2 = (z2.T @ z2) / (z2.shape[0] - 1)
        cov_loss = _off_diagonal(cov_z1).pow(2).mean() + _off_diagonal(cov_z2).pow(2).mean()

        return self.sim_coeff * repr_loss + self.var_coeff * std_loss + self.cov_coeff * cov_loss

