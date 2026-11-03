# Appearance checkpoint

`appearance.pt` contains the paper's fixed beamspace CNN and Cartesian head:
116,530 parameters, 32 channels, six blocks, dilation 2; head 16 channels / one
block, no depth conditioning or normalisation. The weights use EchoXFlow's [CC BY-NC-SA 4.0 licence](../LICENSE),
as they were trained on EchoXFlow.

SHA-256: `f8b79718d48cc8b8286e032dc5672076508214a863f0513598faaf561b83a3e6`

From the repository root:

```bash
(cd weights && sha256sum --check SHA256SUMS)
```
