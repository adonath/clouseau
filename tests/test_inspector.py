from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import torch
from jax.tree_util import register_dataclass

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


class TorchSubModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)

    def forward(self, x):
        return self.linear(x)


class TorchModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sub_model = TorchSubModel()

    def forward(self, x):
        return self.sub_model(x)


class EqxSubModel(eqx.Module):
    linear: eqx.nn.Linear

    def __init__(self):
        super().__init__()
        self.linear = eqx.nn.Linear(2, 2, key=jax.random.PRNGKey(0))

    def __call__(self, x):
        return self.linear(x)


class EqxModel(eqx.Module):
    sub_model: EqxSubModel

    def __init__(self):
        super().__init__()
        self.sub_model = EqxSubModel()

    def __call__(self, x):
        return self.sub_model(x)


def test_jax(tmp_path):
    path = tmp_path / "trace.safetensors"
    m = Model(SubModel(Linear(jnp.ones((2, 2)), jnp.ones(2))))

    x = jnp.ones((2, 2))

    with inspector.tail(
        m, path, filter_=lambda p, _: isinstance(_, (Linear, SubModel))
    ) as fm:
        fm(x)

    data = inspector.read_from_safetensors(path, framework="jax")
    assert tuple(data.keys()) == ("sub_model.linear.__call__", "sub_model.__call__")


def test_torch(tmp_path):
    path = tmp_path / "trace.safetensors"

    m = TorchModel()

    x = torch.ones((2, 2))

    with inspector.tail(
        m, path, filter_=lambda p, _: isinstance(_, (torch.nn.Linear, TorchSubModel))
    ) as fm:
        fm(x)

    data = inspector.read_from_safetensors(path, framework="torch")
    assert tuple(data.keys()) == ("sub_model.linear.forward", "sub_model.forward")


def test_equinox(tmp_path):
    path = tmp_path / "trace.safetensors"
    m = EqxModel()

    x = jnp.ones((2, 2))

    with inspector.tail(
        m, path, filter_=lambda p, _: isinstance(_, (eqx.nn.Linear, EqxSubModel))
    ) as fm:
        fm(x)

    data = inspector.read_from_safetensors(path, framework="jax")
    assert tuple(data.keys()) == ("sub_model.linear.__call__", "sub_model.__call__")
