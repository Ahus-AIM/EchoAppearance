"""Data-free identity check of the tensor interface."""
import torch
from image_enhancement import AppearanceFilter, scan_convert
from image_enhancement.beam import BeamEnhancer
from image_enhancement.scan_conversion import SectorGeometry

if __name__ == '__main__':
    torch.set_num_threads(2)
    model = AppearanceFilter(BeamEnhancer(channels=4, blocks=1, head_channels=4, head_blocks=1)).eval()
    frames = torch.rand(2, 39, 23)
    filtered = model(frames)
    assert filtered.shape == frames.shape
    torch.testing.assert_close(filtered, frames, atol=0, rtol=0)
    geometry = SectorGeometry(0., .16, 0., 1.3)
    original = scan_convert(frames, geometry, output_size=(47, 53))
    cartesian = scan_convert(filtered, geometry, output_size=(47, 53), display_head=model.display_head)
    torch.testing.assert_close(cartesian, original, atol=0, rtol=0)
    print('Native tensor forward and separate scan conversion verified.')
