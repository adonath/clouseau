"""Record the forward pass of a JAX / Equinox model.

Run with::

    uv run python examples/jax_equinox_example.py
"""

from typing import Any

import equinox as eqx
import jax

from clouseau import inspector

PATH = ".clouseau"

keys = jax.random.split(jax.random.PRNGKey(918832), 4)

model = eqx.nn.Sequential([
    eqx.nn.Linear(764, 100, key=keys[0]),
    eqx.nn.Lambda(jax.nn.relu),
    eqx.nn.Linear(100, 50, key=keys[1]),
    eqx.nn.Lambda(jax.nn.relu),
    eqx.nn.Linear(50, 10, key=keys[2]),
    eqx.nn.Lambda(jax.nn.sigmoid),
])

x = jax.random.normal(keys[3], (764,))


def is_leaf(path: tuple[Any, ...], node: Any) -> bool:
    return isinstance(node, jax.Array) or node in (jax.nn.relu, jax.nn.sigmoid)


if __name__ == "__main__":
    with inspector.tail(model, path=PATH, is_leaf=is_leaf) as m:
        # block, so the async io callbacks complete before the cache is flushed
        m(x).block_until_ready()

    inspector.magnify(f"{PATH}/activations-000.safetensors")
