"""Tensor-only appearance filtering; geometry is explicit and independent of file formats."""
from dataclasses import dataclass, replace
import math
from pathlib import Path

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .beam import BeamEnhancer
from .scan_conversion import Grid, SectorGeometry


@dataclass(frozen=True)
class Recording:
    """Native frames (B, depth, beam), acquisition geometry and original frame times."""
    frames: Tensor
    geometry: SectorGeometry
    timestamps: Tensor
    sampling: str = 'endpoints'
    output_size: tuple[int, int] = (256, 256)

    @property
    def shape(self):
        return self.frames.shape

    def to(self, device):
        return replace(self, frames=self.frames.to(device))


@dataclass(frozen=True)
class TensorPair:
    unfiltered: Tensor
    filtered: Tensor
    support: Tensor


def full_sector_grid(geometry: SectorGeometry, output_size: tuple[int, int]) -> Grid:
    """Fit the entire sector onto (height, width), preserving physical coordinates.

    Native geometry uses metres and radians. Output pixel centres span the
    sector's bounding rectangle; the support mask removes points outside it.
    Pass an explicit Grid instead to preserve an existing task's coordinates.
    """
    h, w = output_size
    if not isinstance(h, int) or not isinstance(w, int) or min(h, w) < 2:
        raise ValueError('output_size must contain integer height and width >= 2')
    start, stop = geometry.tilt_rad - geometry.width_rad / 2, geometry.tilt_rad + geometry.width_rad / 2
    angles = [start, stop] + [a for a in (-math.pi, -math.pi/2, 0., math.pi/2, math.pi) if start <= a <= stop]
    radii = (geometry.depth_start_m - geometry.depth_offset_m, geometry.depth_end_m - geometry.depth_offset_m)
    xs = [geometry.apex_x_m + r * math.sin(a) for r in radii for a in angles]
    zs = [geometry.apex_z_m + r * math.cos(a) for r in radii for a in angles]
    dx, dz = (max(xs)-min(xs))/(w-1), (max(zs)-min(zs))/(h-1)
    return Grid(h, w, min(xs)-dx/2, min(zs)-dz/2, dx, dz)


def _sampling_grid(shape, geometry, grid, output_size, sampling, device):
    if geometry is None:
        raise ValueError('Acquisition geometry is required; width and height alone are insufficient')
    if (not all(math.isfinite(v) for v in vars(geometry).values())
            or geometry.depth_end_m <= geometry.depth_start_m
            or geometry.depth_start_m < geometry.depth_offset_m
            or not 0 < geometry.width_rad <= math.pi):
        raise ValueError('Geometry must have finite coordinates and positive depth/angle extents')
    if grid is not None and output_size is not None:
        raise ValueError('Provide either grid or output_size')
    grid = grid or full_sector_grid(geometry, output_size or (256, 256))
    if (min(grid.height, grid.width) < 2 or grid.dx_m <= 0 or grid.dz_m <= 0
            or not all(math.isfinite(v) for v in vars(grid).values())):
        raise ValueError('Grid must have finite coordinates and positive size/spacing')
    sampling = sampling or 'endpoints'
    if sampling not in ('endpoints', 'centres'):
        raise ValueError('sampling must be endpoints or centres')
    nd, nb = shape
    ys = grid.z0_m + (torch.arange(grid.height, device=device, dtype=torch.float64) + .5) * grid.dz_m
    xs = grid.x0_m + (torch.arange(grid.width, device=device, dtype=torch.float64) + .5) * grid.dx_m
    zz, xx = torch.meshgrid(ys, xs, indexing='ij')
    dx, dz = xx - geometry.apex_x_m, zz - geometry.apex_z_m
    radius = torch.hypot(dx, dz) + geometry.depth_offset_m
    angle = torch.atan2(dx, dz) - geometry.tilt_rad
    fraction = (radius - geometry.depth_start_m) / (geometry.depth_end_m - geometry.depth_start_m)
    row = fraction * (nd-1) if sampling == 'endpoints' else fraction * nd - .5
    col = (angle / geometry.width_rad + .5) * (nb-1)
    support = ((radius >= geometry.depth_start_m) & (radius <= geometry.depth_end_m)
               & (angle.abs() <= geometry.width_rad / 2))
    coords = torch.stack((2*col.clamp(0,nb-1)/(nb-1)-1, 2*row.clamp(0,nd-1)/(nd-1)-1), -1)
    return coords, support


class AppearanceFilter(nn.Module):
    """Filter native tensors (B,H,W) without changing their shape or pixel coordinates.

    Native uint8 is divided by 255; float32 must already be in [0,1]. No
    per-frame normalisation occurs. Forward returns a beamspace tensor; use
    scan_convert for display. compare retains the complete paired pipeline.
    'endpoints' samples both radial endpoints, as in public EchoXFlow;
    'centres' uses the registered display-pair depth-bin convention.
    Frames are processed in bounded chunks. Device transfers are explicit.
    Floating input gradients are retained; load_filter freezes model weights.
    """
    def __init__(self, model: BeamEnhancer, chunk_size: int = 1):
        super().__init__()
        if not isinstance(chunk_size, int) or chunk_size < 1:
            raise ValueError('chunk_size must be a positive integer')
        self.model, self.chunk_size = model, chunk_size

    @property
    def display_head(self):
        """The checkpoint's Cartesian residual head, applied after scan conversion."""
        return self.model.cart

    def forward(self, frames: Recording | Tensor) -> Tensor:
        """Apply the beamspace CNN, returning a tensor with the input (B,H,W) shape.

        Outputs remain on the beam grid and are intentionally not clamped before
        scan conversion. Pass display_head to scan_convert for the full trained
        appearance model, including its Cartesian residual head.
        """
        if isinstance(frames, Recording):
            frames = frames.frames
        if not isinstance(frames, Tensor) or frames.ndim != 3 or frames.shape[0] < 1 or min(frames.shape[1:]) < 2:
            raise ValueError('Expected a tensor shaped (B,H,W), B >= 1 and H,W >= 2')
        parameter = next(self.model.parameters())
        if frames.device != parameter.device or parameter.dtype != torch.float32:
            raise ValueError('Frames and float32 model must be on the same device')
        if any(m.training for m in self.model.modules()):
            raise ValueError('Call model.eval() before appearance filtering')
        if frames.dtype == torch.uint8:
            frames = frames.float() / 255
        elif frames.dtype != torch.float32:
            raise ValueError('Use native uint8 or float32 brightness')
        if not bool(torch.isfinite(frames).all()) or bool((frames < 0).any()) or bool((frames > 1).any()):
            raise ValueError('Brightness must be finite and in [0,1]')
        return torch.cat([self.model.beam(batch[:, None])[:, 0] for batch in frames.split(self.chunk_size)])

    def compare(self, recording: Recording | Tensor, *, geometry: SectorGeometry | None = None,
                output_size: tuple[int, int] | None = None, grid: Grid | None = None,
                sampling: str | None = None) -> TensorPair:
        if isinstance(recording, Recording):
            geometry = geometry or recording.geometry
            sampling = sampling or recording.sampling
            if grid is None:
                output_size = output_size or recording.output_size
            frames = recording.frames
        else:
            frames = recording
        if not isinstance(frames, Tensor) or frames.ndim != 3 or frames.shape[0] < 1 or min(frames.shape[1:]) < 2:
            raise ValueError('Expected a tensor shaped (B,H,W), B >= 1 and H,W >= 2')
        parameter = next(self.model.parameters())
        if frames.device != parameter.device or parameter.dtype != torch.float32:
            raise ValueError('Frames and float32 model must be on the same device')
        if any(m.training for m in self.model.modules()):
            raise ValueError('Call model.eval() before appearance filtering')
        if frames.dtype == torch.uint8:
            x = frames.to(torch.float32) / 255
        elif frames.dtype == torch.float32:
            x = frames
        else:
            raise ValueError('Use native uint8 or float32 brightness')
        if not bool(torch.isfinite(x).all()) or bool((x < 0).any()) or bool((x > 1).any()):
            raise ValueError('Brightness must be finite and in [0,1]')
        coords, support = _sampling_grid(x.shape[1:], geometry, grid, output_size, sampling, x.device)
        ranges = x.new_tensor([geometry.depth_start_m, geometry.depth_end_m, x.shape[1]])
        plain, filtered = [], []
        for batch in x.split(self.chunk_size):
            coords_batch = coords.to(x.dtype)[None].expand(batch.shape[0], -1, -1, -1)
            inp = batch[:, None]
            plain.append(F.grid_sample(inp, coords_batch, mode='bilinear', padding_mode='border', align_corners=True)[:,0])
            filtered.append(self.model(inp, ranges[None].expand(batch.shape[0], -1), coords_batch)[:,0].clamp(0,1))
        mask = support[None].expand(x.shape[0], -1, -1)
        return TensorPair(torch.where(mask, torch.cat(plain), 0.), torch.where(mask, torch.cat(filtered), 0.), mask)


def scan_convert(frames: Tensor, geometry: SectorGeometry, *, output_size=None, grid=None,
                 sampling='endpoints', display_head=None, outside=0.) -> Tensor:
    """Scan-convert (H,W) or (B,H,W) tensors onto a specified Cartesian grid.

    Geometry is in metres/radians; output_size is (height,width). uint8 input
    is scaled by 255, while floating inputs retain their values. Provide the
    loaded filter's display_head for its complete trained Cartesian output.
    outside defaults to zero; NaN is useful for transparent plotting.
    """
    if not isinstance(frames, Tensor) or frames.ndim not in (2,3):
        raise ValueError('Expected a (H,W) or (B,H,W) tensor')
    single = frames.ndim == 2
    x = frames[None] if single else frames
    if x.shape[0] < 1 or min(x.shape[1:]) < 2:
        raise ValueError('Expected nonempty frames with both dimensions >= 2')
    if x.dtype == torch.uint8:
        x = x.float() / 255
    elif x.dtype != torch.float32:
        raise ValueError('Use uint8 or float32 frames')
    if not bool(torch.isfinite(x).all()):
        raise ValueError('Frame values must be finite')
    coords, support = _sampling_grid(x.shape[1:], geometry, grid, output_size, sampling, x.device)
    if display_head is not None:
        parameter = next(display_head.parameters())
        if parameter.device != x.device or parameter.dtype != x.dtype or any(m.training for m in display_head.modules()):
            raise ValueError('Display head must be in evaluation mode on the same device/dtype')
    output = []
    for batch in x.split(1):
        cart = F.grid_sample(batch[:,None], coords.to(x.dtype)[None], mode='bilinear',
                             padding_mode='border', align_corners=True)
        if display_head is not None:
            cart = display_head(cart).clamp(0,1)
        output.append(cart[:,0])
    result = torch.where(support[None], torch.cat(output), outside)
    return result[0] if single else result


def load_filter(checkpoint: str | Path = 'weights/appearance.pt', *, device: str = 'cpu', chunk_size: int = 1) -> AppearanceFilter:
    """Load the bundled filter without downloads, file-format adapters or GPU selection."""
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if not isinstance(saved, dict) or not {'spec', 'state_dict'} <= saved.keys():
        raise ValueError('Expected a checkpoint with spec and state_dict')
    model = BeamEnhancer(**saved['spec'])
    model.load_state_dict(saved['state_dict'], strict=True)
    return AppearanceFilter(model.to(device=device, dtype=torch.float32), chunk_size).eval().requires_grad_(False)


def load_example_recording(directory: str | Path = 'examples/recordings') -> Recording:
    """Load the bundled 32-frame native tensor and geometry/timestamps using weights_only=True."""
    directory = Path(directory)
    frames = torch.load(directory / 'frames.pt', map_location='cpu', weights_only=True)
    metadata = torch.load(directory / 'metadata.pt', map_location='cpu', weights_only=True)
    return Recording(frames, SectorGeometry(**metadata['geometry']), metadata['timestamps_s'],
                     metadata['sampling'], tuple(metadata['output_size']))
