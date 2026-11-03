"""Filter the bundled native EchoXFlow tensor; no EchoXFlow package is needed."""
from pathlib import Path
import torch
from image_enhancement import load_filter, load_example_recording

if __name__ == '__main__':
    torch.set_num_threads(2)
    recording = load_example_recording()
    model = load_filter()
    filtered = model(recording.frames)
    Path('outputs').mkdir(exist_ok=True)
    torch.save(filtered, 'outputs/filtered.pt')
    print(f'Native shape preserved: {tuple(recording.shape)} -> {tuple(filtered.shape)}; saved outputs/filtered.pt')
