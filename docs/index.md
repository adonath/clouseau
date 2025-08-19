# clouseau

[![Release](https://img.shields.io/github/v/release/adonath/clouseau)](https://img.shields.io/github/v/release/adonath/clouseau)
[![Build status](https://img.shields.io/github/actions/workflow/status/adonath/clouseau/main.yml?branch=main)](https://github.com/adonath/clouseau/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/adonath/clouseau)](https://img.shields.io/github/commit-activity/m/adonath/clouseau)
[![License](https://img.shields.io/github/license/adonath/clouseau)](https://img.shields.io/github/license/adonath/clouseau)

![clouseau-banner](_static/clouseau-banner.jpg)

Clouseau is a little tool to basically do one thing: record and inspect the forward pass
of neural networks in a library independent way. It is designed to be used with [PyTorch](https://pytorch.org/)
and [Jax](https://docs.jax.dev/) (others libraries might come later...).
It helps you to debug models, transition models from one framework to another (!),
and inspect the inner workings of neural networks.

## Installation

Clouseau is available on PyPI and can be installed with:

```bash
python -m pip install clouseau
```

## Usage

Let"s start with a simple example using PyTorch:

### PyTorch Example

```python
import torch
from torch import nn
from clouseau import inspector
from collections import OrderedDict

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

with inspector.tail(model) as m:
    m(x)
```

This executes the forward pass of the model and records all `forward` operations. You can then inspect the recorded arrays using:

```python
inspector.magnify(".clouseau/trace.safetensors")
```

For PyTorch models the inspector registers a forward hook for each layer that matches the default filter, which is
`isinstance(node, torch.nn.Module)`. It adds all array to a global cache, and finally writes the cache
to a [safetensors]() file, on exiting the context manager. After writing the file, the cache is cleared.

### Jax Example

`clouseau` also works with Jax. It recognizes any valid [PyTree](https://docs.jax.dev/en/latest/pytrees.html) and wraps
its nodes into a custom wrapper.

Usage is exactly the same as in the example above:

```python
import jax
import equinox as eqx
from clouseau import inspector
import tempfile
from pathlib import Path

tmpdir = tempfile.TemporaryDirectory()
path = Path(tmpdir.name)

keys = jax.random.split(jax.random.PRNGKey(918832), 4)

model = eqx.nn.Sequential([
    eqx.nn.Linear(764, 100, key=keys[0]),
    eqx.nn.Lambda(jax.nn.relu),
    eqx.nn.Linear(100, 50, key= keys[1]),
    eqx.nn.Lambda(jax.nn.relu),
    eqx.nn.Linear(50, 10, key=keys[2]),
    eqx.nn.Lambda(jax.nn.sigmoid),
])
x = jax.random.normal(keys[3], (764,))

def is_leaf(path, node):
    return isinstance(node, jax.Array) or node in (jax.nn.relu, jax.nn.sigmoid)

with inspector.tail(model, path=path / "activations.safetensors", is_leaf=is_leaf) as m:
    m(x)
```

You can also provide a custom path to the `tail` function, which will be used to store the safetensors file.
As the wrapper is also a PyTree node itself it can be used in any PyTree context. Thus it should also be compatible
with libraries such as [Equinox](https://docs.kidger.site/equinox/).

### Filtering

Clouseau provides a generic filtering mechanism to filter the layers you are interested in. A filter function
has the following signature:

```python
def filter_(path, node):
    return ...
```

Now we can use the model above and e.g. only trace the output of the activation functions:

```python
def filter_(path, node):
    return node in (jax.nn.relu, jax.nn.sigmoid)

with inspector.tail(model, path=path / "trace-jax-filtered.safetensors", filter_=filter_, is_leaf=is_leaf) as m:
    m(x)
```

Alternatively you can also filter on the content of the path, like so:

```python
def filter_(path, node):
    return "act" in path

with inspector.tail(model, path=path / "trace-jax-filtered.safetensors", filter_=filter_, is_leaf=is_leaf) as m:
    m(x)

```

The path is a list of strings, while the node is the layer object. In Pytorch, this is a subclass
of `torch.nn.Module`, in Jax it can be any valid node of a PyTree.

`clouseau` provide a little helper function to read from the safetensors file. This is important because
safetensor files do not conserve the order of the tensors. However typically it is desired to inspect
the outputs of the layers in the order they were called. As a workaround `clouseau`stores the order
in the metadata and re-orders on read. For convenience there is a small wrapper the pre-serves the order
on read:

```python
from clouseau.io_utils import read_from_safetensors

arrays = read_from_safetensors(path / "activations.safetensors")
```
