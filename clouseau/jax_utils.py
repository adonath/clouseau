from dataclasses import dataclass
from functools import partial
from typing import Any, Callable, Union

import jax
from jax.tree_util import GetAttrKey, SequenceKey, register_dataclass

from .io_utils import PATH_SEP

JaxKeys = Union[GetAttrKey, SequenceKey]  # type: ignore[no-any-unimported]

AnyArray = Any

CACHE = {}


# only works in latest jax
# join_path = partial(keystr, simple=True, separator=PATH_SEP)
def join_path(path: tuple[JaxKeys, ...]) -> str:
    """Join path to Pytree leave"""
    values = [
        getattr(_, "name", str(getattr(_, "idx", getattr(_, "key", None))))
        for _ in path
    ]
    return ".".join(values)


def get_node_types(treedef: Any) -> list[type]:
    """Get unique node types in a pytree"""
    node_types = set()

    def traverse(node: Any) -> None:
        if node.node_data() is None:
            return

        node_types.add(node.node_data()[0])

        for child in node.children():
            traverse(child)

    traverse(treedef)
    return sorted(node_types, key=lambda x: x.__name__)


def add_to_cache_jax(x: AnyArray, key: str) -> Any:
    """Add a intermediate x to the global cache"""
    CACHE[key] = x
    return x


def wrap_model_helper(
    node: Any,
    filter_: Callable,
    is_leaf: Callable,
    path: tuple[JaxKeys, ...] = (),
) -> Any:
    """Recursively apply the clouseau wrapper class"""
    if is_leaf(path, node):
        return node

    children, treedef = jax.tree.flatten_with_path(
        node, is_leaf=lambda _: _ is not node
    )
    children = [
        wrap_model_helper(
            _,
            is_leaf=is_leaf,
            filter_=filter_,
            path=(*path, p[0]),
        )
        for p, _ in children
    ]
    node = treedef.unflatten(children)

    if filter_(path, node):
        return _ClouseauJaxWrapper(node, path=join_path(path))

    return node


def wrap_model(
    model: Any,
    filter_: Callable[[tuple[str, ...], Any], bool] | None = None,
    is_leaf: Callable | None = None,
) -> tuple[Any, None]:
    """Wrap model jax"""
    if filter_ is None:
        filter_ = lambda p, _: callable(_)

    if is_leaf is None:
        is_leaf = lambda p, _: isinstance(_, jax.Array)

    model = wrap_model_helper(model, filter_=filter_, is_leaf=is_leaf)
    return getattr(model, "_model", model), None


@partial(register_dataclass, data_fields=("_model",), meta_fields=("path", "call_name"))
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

    _model: Callable
    path: str
    call_name: str = "__call__"

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        x = getattr(self._model, self.call_name)(*args, **kwargs)

        key = self.path + PATH_SEP + self.call_name
        callback = partial(add_to_cache_jax, key=key)

        jax.experimental.io_callback(callback, x, x)
        return x
