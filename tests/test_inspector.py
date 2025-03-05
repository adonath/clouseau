from dataclasses import dataclass

import jax
import jax.numpy as jnp
import torch
from jax.tree_util import register_dataclass
from torch import nn as torch_nn

from clouseau import inspector


@register_dataclass
@dataclass
class Linear:
    weight: jax.Array
    bias: jax.Array

    def __call__(self, x):
        return jnp.dot(x, self.weight) + self.bias


@register_dataclass
@dataclass
class SubModel:
    linear: Linear

    def __call__(self, x):
        return self.linear(x)


@register_dataclass
@dataclass
class Model:
    sub_model: SubModel

    def __call__(self, x):
        return self.sub_model(x)


class TorchSubModel(torch_nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch_nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class TorchModel(torch_nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_model = TorchSubModel()

    def forward(self, x):
        return self.sub_model(x)


def test_jax(tmp_path):
    path = tmp_path / "trace.safetensors"
    m = Model(SubModel(Linear(jnp.ones((2, 2)), jnp.ones(2))))

    x = jnp.ones((2, 2))

    with inspector.tail(m, path, filter_=lambda p, _: isinstance(_, Linear)) as fm:
        fm(x)

    data = inspector.read_from_safetensors(path, framework="jax")
    assert "sub_model.linear.__call__" in data


def test_torch(tmp_path):
    path = tmp_path / "trace.safetensors"

    m = TorchModel()

    x = torch.ones((2, 2))

    with inspector.tail(m, path, filter_=lambda p, _: isinstance(_, torch_nn.Linear)) as fm:
        fm(x)

    data = inspector.read_from_safetensors(path, framework="torch")
    assert "sub_model.linear.forward" in data


def test_equinox(tmp_path):
    path = tmp_path / "trace.safetensors"
    m = Model(SubModel(Linear(jnp.ones((2, 2)), jnp.ones(2))))

    x = jnp.ones((2, 2))

    with inspector.tail(m, path, filter_=lambda p, _: isinstance(_, Linear)) as fm:
        fm(x)

    data = inspector.read_from_safetensors(path, framework="jax")
    assert "sub_model.linear.__call__" in data



