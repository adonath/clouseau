"""Record the forward pass of a PyTorch model.

Run with::

    uv run python examples/pytorch_example.py
"""

from collections import OrderedDict

import torch
from torch import nn

from clouseau import inspector

PATH = ".clouseau"
FILENAME_PATTERN = "activations-{idx:03d}-torch.safetensors"

model = nn.Sequential(
    OrderedDict([
        ("dense1", nn.Linear(764, 100)),
        ("act1", nn.ReLU()),
        ("dense2", nn.Linear(100, 50)),
        ("act2", nn.ReLU()),
        ("output", nn.Linear(50, 10)),
        ("outact", nn.Sigmoid()),
    ])
)

x = torch.randn((764,))


if __name__ == "__main__":
    with inspector.tail(model, path=PATH, filename_pattern=FILENAME_PATTERN) as m:
        m(x)

    inspector.magnify(f"{PATH}/" + FILENAME_PATTERN.format(idx=0))
