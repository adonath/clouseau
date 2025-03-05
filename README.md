# Clouseau: the forward pass inspector

[![Release](https://img.shields.io/github/v/release/adonath/clouseau)](https://img.shields.io/github/v/release/adonath/clouseau)
[![Build status](https://img.shields.io/github/actions/workflow/status/adonath/clouseau/main.yml?branch=main)](https://github.com/adonath/clouseau/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/adonath/clouseau/branch/main/graph/badge.svg)](https://codecov.io/gh/adonath/clouseau)
[![Commit activity](https://img.shields.io/github/commit-activity/m/adonath/clouseau)](https://img.shields.io/github/commit-activity/m/adonath/clouseau)
[![License](https://img.shields.io/github/license/adonath/clouseau)](https://img.shields.io/github/license/adonath/clouseau)

<p align="center">
<img width="61.8%" src="https://raw.githubusercontent.com/adonath/clouseau/main/docs/_static/clouseau-banner.jpg" alt="Clouseau Banner"/>
</p>

A library independent forward pass inspector for neural nets. The tool is designed to be used with [PyTorch](https://pytorch.org/) and [Jax](https://docs.jax.dev/) (others libraries might come later...).
It allows you to register hooks for the forward pass of a model, and write the forward pass activations
to a file for later inspection. It is useful for debugging models or transitioning models from one framework to another and checking their equivalence at any stage.

## Installation

```bash
pip install clouseau
```

## Usage

### Jax / Equinox Example

You can use the inspector as a context manager to record the forward pass of a model. The following example shows how to use the inspector with a model from the [Equinox](https://docs.kidger.site/equinox/) library:

```python
import jax
import equinox as eqx
from clouseau import inspector

keys = jax.random.split(jax.random.PRNGKey(918832), 3)

model = eqx.nn.Sequential([
    eqx.nn.Linear(764, 100, keys[0]),
    jax.nn.relu,
    eqx.nn.Linear(100, 50, keys[0]),
    jax.nn.relu,
    eqx.nn.Linear(50, 10, keys[0]),
    jax.nn.sigmoid,
])
x = jax.random.normal(jax.random.PRNGKey(0), (764,))

with inspector.tail(model, path="activations.safetensors") as m:
    m(x)
```

Then in an interactive session inspect the recorded activations:

```python
from clouseau import inspector

inspector.magnify("activations.safetensors")
```

Which will open the file and generate a hierachical [treescope](https://treescope.readthedocs.io/en/stable/) view of the activations.

### PyTorch Example

```python
from torch import nn
from clouseau import inspector

model = nn.Sequential({
    "dense1": nn.Linear(764, 100),
    "act1": nn.ReLU(),
    "dense2": nn.Linear(100, 50),
    "act2": nn.ReLU(),
    "output": nn.Linear(50, 10),
    "outact": nn.Sigmoid(),
})

x = torch.randn((764,))

with inspector.tail(model) as m:
    m(x)
```

For more advanced usage including filtering layer types, please refer to the [documentation](https://adonath.github.io/clouseau/).
