"""Acquisition geometry and Cartesian pixel-edge grid, in metres/radians."""
from dataclasses import dataclass


@dataclass(frozen=True)
class SectorGeometry:
    depth_start_m: float
    depth_end_m: float
    tilt_rad: float
    width_rad: float
    apex_x_m: float = 0.0  # lateral apex position in the Cartesian frame
    apex_z_m: float = 0.0  # axial apex position (negative = apex above grid top)
    depth_offset_m: float = 0.0


@dataclass(frozen=True)
class Grid:
    height: int
    width: int
    x0_m: float  # x of the left edge of column 0
    z0_m: float  # z of the top edge of row 0
    dx_m: float
    dz_m: float
