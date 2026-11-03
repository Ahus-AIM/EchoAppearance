"""Small identity-initialised residual CNN. Output has the input's size and pixel coordinates."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def _norm(kind: str, channels: int) -> nn.Module:
    if kind == "none":
        return nn.Identity()
    if kind == "bn":
        return nn.BatchNorm2d(channels)
    if kind == "in":  # per-image statistics: discards absolute brightness per patch/frame, so it must earn its place
        return nn.InstanceNorm2d(channels, affine=True)
    raise ValueError(f"unknown norm {kind!r}")


class ResBlock(nn.Module):
    def __init__(self, channels: int, norm: str = "none", dilation: int = 1) -> None:
        super().__init__()
        self.n1, self.n2 = _norm(norm, channels), _norm(norm, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=dilation, dilation=dilation)
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.act(self.n2(self.conv1(self.act(self.n1(x))))))


class ResidualEnhancer(nn.Module):
    """y_raw = x + f(x). No pooling/stride/resizing; final conv is zero-initialised (identity at start).

    Axes studied separately: `channels` (width), `blocks` (depth), `dilation` (receptive field at fixed depth/width)
    and `norm` ('none' keeps absolute intensity behaviour; 'bn' and 'in' are ablations)."""

    def __init__(self, channels: int = 32, blocks: int = 6, norm: str = "none", dilation: int = 1, extra_in: int = 0) -> None:
        super().__init__()
        self.stem = nn.Conv2d(1 + extra_in, channels, 3, padding=1)  # extra_in: conditioning channels, not added back
        self.blocks = nn.Sequential(*[ResBlock(channels, norm, dilation) for _ in range(blocks)])
        self.out_norm = _norm(norm, channels)
        self.act = nn.GELU()
        self.head = nn.Conv2d(channels, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        # stem and head are 3x3 (d=1); every block adds two 3x3 convs of dilation d
        # conv-only receptive field. Instance norm makes every output depend on whole-image statistics (non-local):
        # tiled inference is then not equivalent to full-frame inference.
        self.receptive_field = 1 + 2 * (1 + 2 * blocks * dilation + 1)
        self.local = norm != "in"
        self.spec = {"channels": channels, "blocks": blocks, "norm": norm, "dilation": dilation}

    def forward(self, x: Tensor, extra: Tensor | None = None) -> Tensor:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"expected (N,1,H,W), got {tuple(x.shape)}")
        s = self.stem(x if extra is None else torch.cat([x, extra], 1))
        return x + self.head(self.act(self.out_norm(self.blocks(s))))
