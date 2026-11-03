"""Fixed paper architecture: beam CNN, bilinear sampling and Cartesian head."""
import torch.nn.functional as F
from torch import nn
from .model import ResidualEnhancer


class BeamEnhancer(nn.Module):
    def __init__(self, channels=32, blocks=6, norm='none', dilation=2,
                 head_channels=16, head_blocks=1, depth='none', domain='beam'):
        super().__init__()
        if depth != 'none' or domain != 'beam':
            raise ValueError('This release contains the fixed unconditioned beamspace architecture')
        self.domain, self.depth = domain, ()
        self.beam = ResidualEnhancer(channels, blocks, norm, dilation)
        self.cart = ResidualEnhancer(head_channels, head_blocks, 'none', 1)
        self.spec = dict(channels=channels, blocks=blocks, norm=norm, dilation=dilation,
                         head_channels=head_channels, head_blocks=head_blocks, depth=depth, domain=domain)

    def forward(self, frames, ranges, grid):
        converted = F.grid_sample(self.beam(frames), grid, mode='bilinear',
                                  padding_mode='border', align_corners=True)
        return self.cart(converted)
