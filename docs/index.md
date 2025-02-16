# clouseau

[![Release](https://img.shields.io/github/v/release/adonath/clouseau)](https://img.shields.io/github/v/release/adonath/clouseau)
[![Build status](https://img.shields.io/github/actions/workflow/status/adonath/clouseau/main.yml?branch=main)](https://github.com/adonath/clouseau/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/adonath/clouseau)](https://img.shields.io/github/commit-activity/m/adonath/clouseau)
[![License](https://img.shields.io/github/license/adonath/clouseau)](https://img.shields.io/github/license/adonath/clouseau)

Clouseau is a tool to record and inspect the forward pass
of neural networks. It is designed to be used with PyTorch and Jax (others libraries might come later...). It helps you to debug models, transition models from one framework to another, and inspect the inner workings of neural networks.

## Installation

```bash
pip install clouseau
```

## Usage

```python
import torch
from clouseau import inspector

class MyModel(torch.nn.Module):
    def forward(self, x):
        return x + 1

model = MyModel()

with inspector(model) as m:
    m(torch.randn(1))
```

This executes the forward pass of the model and record all `forward` operations. You can then inspect the model using:

```python

from clouseau import magnifier

magnifier(".clouseau/trace.safetensors")
```

### Filtering

Clouseau provides a generic filtering mechanism to filter the layers you are interested in. A filter has the following
signature:

```python
def filter_(path, node):
    return callable(node)
```

The path is a list of strings, while the node is the layer
object. In Pytorch, this is a subclass of `torch.nn.Module`, in Jax it can be any pytree node.

### Tipps & Tricks

With clever filtering you can pinpoint the exact operation you are interested in
