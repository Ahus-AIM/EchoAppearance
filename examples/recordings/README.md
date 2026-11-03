# Native example recording

One EchoXFlow pure 2D B-mode recording, first 32 consecutive native frames.
`frames.pt`: uint8 tensor `(32, 732, 242)` in (time, depth, beam) order.
`metadata.pt`: native geometry, original timestamps, frame indices, dataset
recording identity and the explicit endpoint sampling convention. No patient,
examination, institution-specific path or acquisition-date fields are included.

These example data use EchoXFlow's [CC BY-NC-SA 4.0 licence](../../LICENSE).
Please cite both articles as given in the repository README. No independent
per-frame intensity normalisation or geometric fitting is used at inference.
