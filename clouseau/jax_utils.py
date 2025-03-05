
from dataclasses import dataclass
from functools import partial
from typing import Any, Callable

import jax
from jax._src.tree_util import _registry_with_keypaths
from jax.tree_util import GetAttrKey, SequenceKey, register_dataclass

JaxKeys = GetAttrKey | SequenceKey
AnyArray = Any

JAX_CACHE = {}

# only works in latest jax
# join_path = partial(keystr, simple=True, separator=PATH_SEP)
def join_path(path: tuple[JaxKeys, ...]) -> str:
    """Join path to Pytree leave"""
    values = [getattr(_, "name", str(getattr(_, "idx", getattr(_, "key", None)))) for _ in path]
    return ".".join(values)


def get_node_types(treedef):
    """Get unique node types in a pytree"""
    node_types = set()

    def traverse(node):
        if node.node_data() is None:
            return

        node_types.add(node.node_data()[0])

        for child in node.children():
            traverse(child)

    traverse(treedef)
    return sorted(node_types, key=lambda x: x.__name__)


def add_to_cache_jax(x: AnyArray, key: str) -> Any:
    """Add a intermediate x to the global cache"""
    JAX_CACHE[key] = x
    return x


def wrap_model_jax(node, path: tuple[JaxKeys, ...] = (), filter_: Callable | None = None):
    """Recursively apply the clouseau wrapper class"""
    if filter_ is None:
        filter_ = lambda p, _: callable(_)

    serializer = _registry_with_keypaths.get(type(node), None)

    # if there is not pytree registration entry it is considered a leaf
    if serializer is None:
        return node

    children, aux = serializer.flatten_with_keys(node)
    children = [wrap_model_jax(_, (*path, p), filter_=filter_) for p, _ in children]
    node = serializer.unflatten_func(aux, children)

    if filter_(path, node):
        return _ClouseauJaxWrapper(node, path=join_path(path))

    return node


@partial(register_dataclass, data_fields=("model",), meta_fields=("path", "call_name"))
@dataclass
class _ClouseauJaxWrapper:
    """Jax module wrapper that applies a callback function after executing the module.

    Parameters
    ----------
    model : Callable
        The JAX model/function to wrap
    path : str
        Location of the wrapped module within the pytree.
    """

    model: Callable
    path: str
    call_name: str = "__call__"

    def __call__(self, *args, **kwargs):
        x = getattr(self.model, self.call_name)(*args, **kwargs)

        key = self.path + PATH_SEP + self.call_name
        callback = partial(add_to_cache_jax, key=key)

        jax.experimental.io_callback(callback, x, x)
        return x