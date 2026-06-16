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
    m = Model(SubModel(Linear(jnp.ones((2, 2)), jnp.ones(2))))

    x = jnp.ones((2, 2))

    with inspector.tail(
        m, tmp_path, filter_=lambda p, _: isinstance(_, (Linear, SubModel))
    ) as fm:
        fm(x)

    data = inspector.read_from_safetensors(
        tmp_path / "activations-000.safetensors", framework="jax"
    )
    assert tuple(data.keys()) == ("sub_model.linear.__call__", "sub_model.__call__")


def test_jax_loop(tmp_path):
    m = Model(SubModel(Linear(jnp.ones((2, 2)), jnp.ones(2))))

    x = jnp.ones((2, 2))

    filter_ = lambda p, _: isinstance(_, (Linear, SubModel))

    with inspector.tail(m, tmp_path, filter_=filter_) as fm:
        for _ in range(5):
            fm(x)

    data = inspector.read_from_safetensors(
        tmp_path / "activations-000.safetensors", framework="jax"
    )
    assert tuple(data.keys()) == ("sub_model.linear.__call__", "sub_model.__call__")


def test_torch(tmp_path):
    m = TorchModel()

    x = torch.ones((2, 2))

    with inspector.tail(
        m,
        tmp_path,
        filter_=lambda p, _: isinstance(_, (torch.nn.Linear, TorchSubModel)),
    ) as fm:
        fm(x)

    data = inspector.read_from_safetensors(
        tmp_path / "activations-000.safetensors", framework="torch"
    )
    assert tuple(data.keys()) == ("sub_model.linear.__call__", "sub_model.__call__")


def test_equinox(tmp_path):
    m = EqxModel()

    x = jnp.ones((2, 2))

    with inspector.tail(
        m, tmp_path, filter_=lambda p, _: isinstance(_, (eqx.nn.Linear, EqxSubModel))
    ) as fm:
        fm(x)

    data = inspector.read_from_safetensors(
        tmp_path / "activations-000.safetensors", framework="jax"
    )
    assert tuple(data.keys()) == ("sub_model.linear.__call__", "sub_model.__call__")


def test_overlapping_recorders_are_isolated(tmp_path):
    # two `tail` contexts open at the same time must not clobber each other's
    # path/config or share recorded data (each recorder owns its own cache)
    m1, m2 = TorchModel(), TorchModel()
    x = torch.ones((2, 2))
    path1, path2 = tmp_path / "rec1", tmp_path / "rec2"

    with (
        inspector.tail(
            m1, path1, filter_=lambda p, _: isinstance(_, torch.nn.Linear)
        ) as fm1,
        inspector.tail(
            m2, path2, filter_=lambda p, _: isinstance(_, TorchSubModel)
        ) as fm2,
    ):
        fm1(x)
        fm2(x)

    data1 = inspector.read_from_safetensors(
        path1 / "activations-000.safetensors", framework="torch"
    )
    data2 = inspector.read_from_safetensors(
        path2 / "activations-000.safetensors", framework="torch"
    )

    # each file holds only its own recorder's filtered keys
    assert tuple(data1.keys()) == ("sub_model.linear.__call__",)
    assert tuple(data2.keys()) == ("sub_model.__call__",)
