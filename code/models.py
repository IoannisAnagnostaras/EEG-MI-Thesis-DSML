from __future__ import annotations

import copy

import torch
from torch import nn


class SENetBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 2) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


class EEGNetEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        embedding_dim: int = 256,
        dropout: float = 0.25,
        temporal_kernel: int = 64,
        depth_multiplier: int = 2,
    ) -> None:
        super().__init__()
        f1 = max(embedding_dim // 8, 16)
        f2 = embedding_dim
        self.feature_dim = f2

        self.temporal = nn.Sequential(
            nn.Conv2d(1, f1, (1, temporal_kernel), padding=(0, temporal_kernel // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(
                f1,
                f1 * depth_multiplier,
                (n_channels, 1),
                groups=f1,
                bias=False,
            ),
            nn.BatchNorm2d(f1 * depth_multiplier),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(dropout),
        )
        self.separable = nn.Sequential(
            nn.Conv2d(
                f1 * depth_multiplier,
                f1 * depth_multiplier,
                (1, 16),
                padding=(0, 8),
                groups=f1 * depth_multiplier,
                bias=False,
            ),
            nn.Conv2d(f1 * depth_multiplier, f2, (1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.head_norm = nn.LayerNorm(f2)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.separable(x)
        x = x.squeeze(2).transpose(1, 2)
        return self.head_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        return tokens.mean(dim=1)


class ShallowConvNetEncoder(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_times: int,
        dropout: float = 0.5,
        n_filters: int = 40,
        temporal_kernel: int = 25,
        pool_kernel: int = 75,
        pool_stride: int = 15,
    ) -> None:
        super().__init__()
        self.temporal_spatial = nn.Sequential(
            nn.Conv2d(1, n_filters, kernel_size=(1, temporal_kernel), bias=False),
            nn.Conv2d(n_filters, n_filters, kernel_size=(n_channels, 1), bias=False),
            nn.BatchNorm2d(n_filters),
        )
        self.pool = nn.AvgPool2d(kernel_size=(1, pool_kernel), stride=(1, pool_stride))
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_times)
            tokens = self._forward_tokens_no_norm(dummy)
            self.token_dim = int(tokens.shape[-1])
            self.n_tokens = int(tokens.shape[1])
            self.feature_dim = int(self.token_dim * self.n_tokens)

        self.token_norm = nn.LayerNorm(self.token_dim)

    def _forward_tokens_no_norm(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal_spatial(x)
        x = torch.square(x)
        x = self.pool(x)
        x = torch.log(torch.clamp(x, min=1e-6))
        x = self.dropout(x)
        return x.squeeze(2).transpose(1, 2)

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self._forward_tokens_no_norm(x)
        return self.token_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        return torch.flatten(tokens, 1)


class LiSENetEncoder(nn.Module):
    def __init__(self, n_channels: int, n_times: int, dropout: float = 0.2, se_reduction: int = 2) -> None:
        super().__init__()
        self.temporal = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(1, 128), padding="same", bias=False),
            SENetBlock(8, reduction=se_reduction),
            nn.BatchNorm2d(8),
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=(n_channels, 1), bias=False),
            SENetBlock(16, reduction=se_reduction),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.feature_compression = nn.Sequential(
            nn.Conv2d(16, 16, kernel_size=(1, 32), padding="same", groups=16, bias=False),
            nn.Conv2d(16, 16, kernel_size=(1, 1), bias=False),
            SENetBlock(16, reduction=se_reduction),
            nn.BatchNorm2d(16),
            nn.ELU(),
            nn.AvgPool2d((1, 16)),
            nn.Dropout(dropout),
        )
        self.token_norm = nn.LayerNorm(16)

        with torch.no_grad():
            dummy = torch.zeros(1, n_channels, n_times)
            tokens = self.forward_tokens(dummy)
            self.token_dim = tokens.shape[-1]
            self.n_tokens = tokens.shape[1]
            self.feature_dim = self.token_dim * self.n_tokens

    def _encode_map(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)
        x = self.temporal(x)
        x = self.spatial(x)
        x = self.feature_compression(x)
        return x

    def forward_tokens(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encode_map(x)
        x = x.squeeze(2).transpose(1, 2)
        return self.token_norm(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.forward_tokens(x)
        return torch.flatten(tokens, 1)


class EEGClassifier(nn.Module):
    def __init__(
        self,
        encoder: nn.Module,
        n_classes: int,
        hidden_dim: int | None = None,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        if hidden_dim is None:
            self.classifier = nn.Linear(encoder.feature_dim, n_classes)
        else:
            self.classifier = nn.Sequential(
                nn.Linear(encoder.feature_dim, hidden_dim),
                nn.ELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, n_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.classifier(features)


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TokenPredictor(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MaskedLatentModel(nn.Module):
    def __init__(self, encoder: nn.Module, mask_ratio: float = 0.35, ema_decay: float = 0.99) -> None:
        super().__init__()
        self.student = encoder
        self.teacher = copy.deepcopy(encoder)
        for parameter in self.teacher.parameters():
            parameter.requires_grad = False
        self.teacher.eval()
        token_dim = int(getattr(encoder, "token_dim", encoder.feature_dim))
        self.predictor = TokenPredictor(token_dim)
        self.mask_ratio = mask_ratio
        self.ema_decay = ema_decay
        self.mask_token = nn.Parameter(torch.zeros(1, 1, token_dim))

    def train(self, mode: bool = True):
        super().train(mode)
        # Keep EMA teacher deterministic even while student/predictor train.
        self.teacher.eval()
        return self

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, float]:
        with torch.no_grad():
            teacher_tokens = self.teacher.forward_tokens(x)

        student_tokens = self.student.forward_tokens(x)
        batch_size, n_tokens, dim = student_tokens.shape
        n_mask = max(1, int(self.mask_ratio * n_tokens))
        mask = torch.zeros(batch_size, n_tokens, dtype=torch.bool, device=x.device)
        for index in range(batch_size):
            positions = torch.randperm(n_tokens, device=x.device)[:n_mask]
            mask[index, positions] = True

        masked = student_tokens.clone()
        masked[mask] = self.mask_token.expand(batch_size, n_tokens, dim)[mask]
        predicted = self.predictor(masked)
        loss = torch.mean((predicted[mask] - teacher_tokens.detach()[mask]) ** 2)
        return loss, float(mask.float().mean().item())

    @torch.no_grad()
    def update_teacher(self) -> None:
        for teacher_param, student_param in zip(self.teacher.parameters(), self.student.parameters()):
            teacher_param.data.mul_(self.ema_decay).add_(student_param.data, alpha=1.0 - self.ema_decay)
