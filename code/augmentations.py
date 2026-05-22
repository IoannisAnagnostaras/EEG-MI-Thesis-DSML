from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def random_scale(x: torch.Tensor, min_scale: float = 0.9, max_scale: float = 1.1) -> torch.Tensor:
    scales = torch.empty(x.size(0), 1, 1, device=x.device).uniform_(min_scale, max_scale)
    return x * scales


def gaussian_noise(x: torch.Tensor, std: float = 0.02) -> torch.Tensor:
    return x + torch.randn_like(x) * std


def random_time_mask(x: torch.Tensor, mask_ratio: float = 0.2) -> tuple[torch.Tensor, torch.Tensor]:
    batch, channels, samples = x.shape
    mask = torch.zeros_like(x)
    span = max(1, int(samples * mask_ratio))
    starts = torch.randint(0, max(samples - span, 1), (batch,), device=x.device)
    x_masked = x.clone()
    for i, start in enumerate(starts):
        x_masked[i, :, start : start + span] = 0.0
        mask[i, :, start : start + span] = 1.0
    return x_masked, mask


def make_two_views(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    view1 = gaussian_noise(random_scale(x))
    view2 = gaussian_noise(random_scale(x))
    return view1, view2


def time_warp(x: torch.Tensor, min_scale: float = 0.8, max_scale: float = 1.25) -> torch.Tensor:
    batch, channels, samples = x.shape
    scale = float(torch.empty(1, device=x.device).uniform_(min_scale, max_scale).item())
    warped_len = max(4, int(round(samples * scale)))
    warped = F.interpolate(x, size=warped_len, mode="linear", align_corners=False)
    return F.interpolate(warped, size=samples, mode="linear", align_corners=False)


def cutout_zero(x: torch.Tensor, min_ratio: float = 0.1, max_ratio: float = 0.3) -> torch.Tensor:
    batch, _, samples = x.shape
    ratio = float(torch.empty(1, device=x.device).uniform_(min_ratio, max_ratio).item())
    span = max(1, int(samples * ratio))
    out = x.clone()
    starts = torch.randint(0, max(samples - span + 1, 1), (batch,), device=x.device)
    for i, start in enumerate(starts.tolist()):
        out[i, :, start : start + span] = 0.0
    return out


def cutout_resize(x: torch.Tensor, min_ratio: float = 0.1, max_ratio: float = 0.25) -> torch.Tensor:
    batch, channels, samples = x.shape
    ratio = float(torch.empty(1, device=x.device).uniform_(min_ratio, max_ratio).item())
    span = max(1, int(samples * ratio))
    out = torch.empty_like(x)
    starts = torch.randint(0, max(samples - span + 1, 1), (batch,), device=x.device)
    for i, start in enumerate(starts.tolist()):
        keep = torch.cat([x[i : i + 1, :, :start], x[i : i + 1, :, start + span :]], dim=-1)
        out[i : i + 1] = F.interpolate(keep, size=samples, mode="linear", align_corners=False)
    return out


def crop_resize(x: torch.Tensor, min_ratio: float = 0.7, max_ratio: float = 0.95) -> torch.Tensor:
    batch, channels, samples = x.shape
    ratio = float(torch.empty(1, device=x.device).uniform_(min_ratio, max_ratio).item())
    crop = max(4, int(samples * ratio))
    out = torch.empty_like(x)
    starts = torch.randint(0, max(samples - crop + 1, 1), (batch,), device=x.device)
    for i, start in enumerate(starts.tolist()):
        piece = x[i : i + 1, :, start : start + crop]
        out[i : i + 1] = F.interpolate(piece, size=samples, mode="linear", align_corners=False)
    return out


def horizontal_flip(x: torch.Tensor) -> torch.Tensor:
    return torch.flip(x, dims=[-1])


def permute_segments(x: torch.Tensor, min_segments: int = 2, max_segments: int = 5) -> torch.Tensor:
    batch, channels, samples = x.shape
    segments = int(torch.randint(min_segments, max_segments + 1, (1,), device=x.device).item())
    if segments <= 1:
        return x
    boundaries = torch.linspace(0, samples, steps=segments + 1, device=x.device).long()
    out = torch.empty_like(x)
    for i in range(batch):
        chunks = [x[i : i + 1, :, boundaries[j] : boundaries[j + 1]] for j in range(segments)]
        order = list(range(segments))
        random.shuffle(order)
        permuted = torch.cat([chunks[idx] for idx in order], dim=-1)
        out[i : i + 1] = permuted
    return out


def make_li_paper_view(x: torch.Tensor) -> torch.Tensor:
    ops = [
        lambda t: gaussian_noise(t, std=0.02),   # amplitude addition
        lambda t: random_scale(t, 0.85, 1.15),   # amplitude scale
        time_warp,
        cutout_resize,
        cutout_zero,
        crop_resize,
        horizontal_flip,
        permute_segments,
    ]
    k = int(torch.randint(2, 4, (1,), device=x.device).item())
    out = x
    for op in random.sample(ops, k=k):
        out = op(out)
    return out
