"""Tensor inference, geometry, batch and shipped-payload contracts."""
from pathlib import Path
import hashlib
import math

import pytest
import torch

from image_enhancement import AppearanceFilter, Recording, load_example_recording, load_filter, scan_convert
from image_enhancement.beam import BeamEnhancer
from image_enhancement.scan_conversion import Grid, SectorGeometry

ROOT = Path(__file__).resolve().parents[1]
torch.set_num_threads(2)


def identity():
    return AppearanceFilter(BeamEnhancer(channels=4, blocks=1, head_channels=4, head_blocks=1),chunk_size=1).eval()


def test_endpoint_geometry_matches_an_analytic_radial_angular_brightness_field():
    radius=torch.linspace(0,1,39)[:,None]
    angle=torch.linspace(0,1,23)[None,:]
    frames=(.2+.3*radius+.1*angle)[None].expand(2,-1,-1)
    geo=SectorGeometry(.008,.16,.07,1.3)
    grid=Grid(47,53,-.12,-.015,.0045,.004)
    pair=identity().compare(frames,geometry=geo,grid=grid)
    z,x=torch.meshgrid(grid.z0_m+(torch.arange(47,dtype=torch.float64)+.5)*grid.dz_m,
                       grid.x0_m+(torch.arange(53,dtype=torch.float64)+.5)*grid.dx_m,indexing='ij')
    distance=torch.hypot(x,z)
    theta=torch.atan2(x,z)-geo.tilt_rad
    expected=.2+.3*(distance-geo.depth_start_m)/(geo.depth_end_m-geo.depth_start_m)+.1*(theta/geo.width_rad+.5)
    mask=(distance>=geo.depth_start_m)&(distance<=geo.depth_end_m)&(theta.abs()<=geo.width_rad/2)
    torch.testing.assert_close(pair.support[0],mask)
    torch.testing.assert_close(pair.filtered[0][mask],expected[mask].float(),atol=2e-6,rtol=0)
    torch.testing.assert_close(pair.filtered,pair.unfiltered,atol=0,rtol=0)
    assert not pair.filtered[~pair.support].any()


def test_uint8_scaling_batch_chunking_and_recording_metadata_are_equivalent():
    frames=torch.randint(0,256,(3,39,23),dtype=torch.uint8)
    geo=SectorGeometry(0.,.16,0.,1.3)
    recording=Recording(frames,geo,torch.tensor([0.,.04,.08]),output_size=(47,53))
    model=load_filter(ROOT/'weights/appearance.pt')
    together=model(recording)
    separate=torch.cat([model(frame[None]) for frame in frames])
    torch.testing.assert_close(together,separate,atol=0,rtol=0)
    torch.testing.assert_close(together,model(frames.float()/255),atol=0,rtol=0)
    assert together.shape==(3,39,23)
    assert all(not p.requires_grad for p in model.parameters())


def test_input_gradients_remain_available_with_frozen_filter_weights():
    frames=torch.rand(2,21,17,requires_grad=True)
    model=identity().requires_grad_(False)
    result=model(frames)
    result.mean().backward()
    assert frames.grad is not None and torch.isfinite(frames.grad).all() and frames.grad.abs().sum()>0
    assert all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('change', ['missing_geometry','bad_shape','range','dtype','sampling','two_grids'])
def test_rejects_ambiguous_or_invalid_tensor_inputs(change):
    frames=torch.rand(2,21,17);kw={'geometry':SectorGeometry(0.,.16,0.,1.3)}
    if change=='missing_geometry':kw={}
    elif change=='bad_shape':frames=frames[:,None]
    elif change=='range':frames=frames+2
    elif change=='dtype':frames=frames.double()
    elif change=='sampling':kw['sampling']='unknown'
    elif change=='two_grids':kw.update(grid=Grid(9,9,-.1,0,.01,.01),output_size=(9,9))
    with pytest.raises(ValueError):identity().compare(frames,**kw)


def test_shipped_payloads_have_matching_checksums_and_no_private_metadata():
    for folder in (ROOT/'weights',ROOT/'examples/recordings'):
        for line in (folder/'SHA256SUMS').read_text().splitlines():
            expected,name=line.split()
            assert hashlib.sha256((folder/name).read_bytes()).hexdigest()==expected
    saved=torch.load(ROOT/'weights/appearance.pt',weights_only=True,map_location='cpu')
    assert set(saved)=={'spec','state_dict'}
    model=load_filter(ROOT/'weights/appearance.pt')
    assert sum(p.numel() for p in model.parameters())==116530
    recording=load_example_recording(ROOT/'examples/recordings')
    assert recording.shape==(32,732,242) and recording.frames.dtype==torch.uint8
    assert recording.timestamps.shape==(32,) and bool((torch.diff(recording.timestamps)>0).all())
    metadata=torch.load(ROOT/'examples/recordings/metadata.pt',weights_only=True,map_location='cpu')
    assert set(metadata)=={'source_dataset','source_recording_id','source_frame_indices','timestamps_s',
                           'geometry','sampling','output_size','source_units','licence'}
    assert metadata['licence']=='CC-BY-NC-SA-4.0'


def test_separate_native_forward_and_scan_conversion_preserve_the_full_trained_pipeline():
    recording=load_example_recording(ROOT/'examples/recordings')
    model=load_filter(ROOT/'weights/appearance.pt')
    frames=recording.frames[:1]
    native=model(frames)
    assert native.shape==frames.shape
    pair=model.compare(frames,geometry=recording.geometry,output_size=(64,71))
    cart=scan_convert(native,recording.geometry,output_size=(64,71),display_head=model.display_head)
    plain=scan_convert(frames,recording.geometry,output_size=(64,71))
    torch.testing.assert_close(cart,pair.filtered,atol=0,rtol=0)
    torch.testing.assert_close(plain,pair.unfiltered,atol=0,rtol=0)
    plotted=scan_convert(native[0],recording.geometry,output_size=(64,71),
                         display_head=model.display_head,outside=float('nan'))
    torch.testing.assert_close(plotted[pair.support[0]],cart[0][pair.support[0]],atol=0,rtol=0)
    assert bool(torch.isnan(plotted[~pair.support[0]]).all())
