"""Clouseau tracks the forward pass of a model and saves the intermediate arrays to a file.

Currently it supports Jax and Pytorch models. They way this is achieved 
is different for both frameworks.

- For Jax it uses a wrapper class that wraps each (callable) node in the pytree (
see also https://github.com/patrick-kidger/equinox/issues/864).
Saving arrays to file is a side effect in Jax. See e.g. https://docs.jax.dev/en/latest/external-callbacks.html
However the global cache seems acceptable in combination with jax.experimental.io_callback,
which is explicitly designed for this purpose.
- For Pytorch it uses a forward hook that is registered on each module.
See e.g. https://web.stanford.edu/~nanbhas/blog/forward-hooks-pytorch/
The forward hook is de-registered after the forward pass is done.


In both cases Jax / Pytorch it tracks the output of the layer. I might add tracking 
of the inputs as well later.
"""

import logging
from dataclasses import dataclass, is_dataclass
from enum import Enum
from functools import cached_property, partial
from pathlib import Path
from typing import Any, Callable

import jax
import treescope
from jax._src.tree_util import _registry_with_keypaths
from jax.tree_util import GetAttrKey, SequenceKey, register_dataclass
from safetensors.flax import save_file as save_file_jax
from safetensors.torch import save_file as save_file_torch

DEFAULT_PATH = Path.cwd() / ".clouseau" / "trace.safetensors"

__all__ = ["inspector", "magnifier"]


log = logging.getLogger(__name__)

GLOBAL_CACHE = {}
PATH_SEP = "."

# only works in latest jax
# join_path = partial(keystr, simple=True, separator=PATH_SEP)

JaxKeys = GetAttrKey | SequenceKey
AnyArray = Any
AnyModel = Any


def join_path(path: tuple[JaxKeys, ...]) -> str:
    """Join path to Pytree leave"""
    values = [getattr(_, "name", str(getattr(_, "idx", getattr(_, "key", None)))) for _ in path]
    return ".".join(values)


class FrameworkEnum(str, Enum):
    """Framework enum"""

    jax = "jax"
    torch = "torch"


def is_torch_model(model: AnyModel) -> bool:
    """Check if model is a torch model"""
    try:
        import torch

        return isinstance(model, torch.nn.Module)
    except ImportError:
        return False


def is_jax_model(model: AnyModel) -> bool:
    """Check if model is a jax model"""
    try:
        import jax

        jax.tree.flatten(model)
    except (ImportError, TypeError):
        return False
    else:
        return True


def unflatten_dict(d: dict[str, Any], sep: str = ".") -> dict[str, Any]:
    """Unflatten dictionary"

    Taken from https://stackoverflow.com/a/6037657/19802442
    """
    result = {}
    for key, value in d.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value
    return result


def save_to_safetensors_jax(x: dict[str, AnyArray], filename: str | Path) -> None:
    """Safetensors I/O for jax"""
    log.info(f"Writing {filename}")
    # safetensors does not support ordered dicts, see https://github.com/huggingface/safetensors/issues/357
    order = {str(idx): key for idx, key in enumerate(x.keys())}
    save_file_jax(x, filename, metadata=order)


def save_to_safetensors_torch(x: dict[str, AnyArray], filename: str | Path) -> None:
    """Safetensors I/O for torch"""
    log.info(f"Writing {filename}")
    # safetensors does not support ordered dicts, see https://github.com/huggingface/safetensors/issues/357
    order = {str(idx): key for idx, key in enumerate(x.keys())}
    save_file_torch(x, filename, metadata=order)


def read_from_safetensors(filename: str | Path, framework: str = "numpy", device=None) -> dict[str, Any]:
    """Read from safetensors"""
    from safetensors import safe_open

    with safe_open(filename, framework=framework, device=device) as f:
        # reorder according to metadata, which maps index to key / path
        keys = dict(sorted(f.metadata().items(), key=lambda _: int(_[0]))).values()
        data = {key: f.get_tensor(key) for key in keys}

    return data


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


def wrap_model_torch(model: AnyModel, filter_: Callable | None = None):
    """Wrap model torch"""
    from torch import nn

    hooks = {}

    if filter_ is None:
        filter_ = lambda p, _: isinstance(_, nn.Module)

    def traverse(path: tuple[str, ...], node):
        if node is None:
            return

        if filter_(path, node):
            name = PATH_SEP.join(path)
            hooks[name] = node.register_forward_hook(add_to_cache_torch(name))

        for p, child in node.named_children():
            traverse((*path, p), child)

    traverse(path=(), node=model)
    return hooks


def wrap_model_jax(node, path: tuple[JaxKeys, ...] = (), filter_: Callable | None = None):
    """Recursively apply the clouseau wrapper class"""
    if filter_ is None:
        filter_ = lambda p, _: is_dataclass(_)

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


def add_to_cache_jax(x: AnyArray, key: str) -> Any:
    """Add a intermediate x to the global cache"""
    GLOBAL_CACHE[key] = x
    return x


def add_to_cache_torch(key: str):
    """Add a intermediate x to the global cache"""

    def hook(module, input_, output):
        GLOBAL_CACHE[key + PATH_SEP + "forward"] = output.detach().clone()

    return hook


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


WRITE_REGISTRY = {
    "jax": save_to_safetensors_jax,
    "torch": save_to_safetensors_torch,
}


class _Inspector:
    """Inspector class that can be used as a context manager."""

    def __init__(self, model: AnyModel, path: str | Path = DEFAULT_PATH, filter_: Callable | None = None):
        self.model = model
        self.path = Path(path)
        self.filter_ = filter_
        self.hooks = None

    @cached_property
    def framework(self):
        """Determine framework"""

        if is_torch_model(self.model):
            return FrameworkEnum.torch
        elif is_jax_model(self.model):
            return FrameworkEnum.jax

        message = "The model does not seem to be a PyTorch or JAX model."
        raise ValueError(message)

    def __enter__(self):
        if self.framework == FrameworkEnum.jax:
            wrapped_model = wrap_model_jax(self.model, filter_=self.filter_)
            return getattr(wrapped_model, "model", wrapped_model)

        self.hooks = wrap_model_torch(model=self.model, filter_=self.filter_)
        return self.model

    def __exit__(self, exc_type, exc_value, traceback):
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not GLOBAL_CACHE:
            log.warning("No arrays were recorded. Check the filter function.")

        WRITE_REGISTRY[self.framework](GLOBAL_CACHE, self.path)
        GLOBAL_CACHE.clear()

        if self.hooks:
            for _, hook in self.hooks.items():
                hook.remove()


def inspector(model: AnyModel, path: str | Path = DEFAULT_PATH, filter_: Callable | None = None) -> _Inspector:
    """Inspect the forward pass of a model

    Parameters
    ----------
    model : object
        The model to inspect. Can be a PyTorch model or JAX/Equinox model.
    path : str or Path
        Path where to store the forward pass arrays.
    filter_ : callable
        Function that filters which tensors to inspect.
        Takes the pytree leaves, child modules as input and returns a boolean.

    Returns
    -------
    _Inspector
        Inspector instance that can be used as a context manager.

    Examples
    --------
    >>> import torch
    >>> from clouseau import inspector, magnifier
    >>> model = torch.nn.Linear(10, 5)
    >>> with inspector(model,  path=".clouseau/trace-torch.safetensors") as fmodel:
    ...     out = fmodel(torch.randn(3, 10))

    When working with a Jax / Equinox model, it is important to add `.block_until_ready()`
    >>> import jax
    >>> with inspector(model, path=".clouseau/trace-jax.safetensors") as fmodel:
    ...     fmodel(x, time).block_until_ready()

    """
    return _Inspector(model=model, path=path, filter_=filter_)


def magnifier(filename: str | Path, framework: str = "numpy", device=None) -> None:
    """Visualize nested arrays using treescope"""
    data = read_from_safetensors(filename, framework=framework, device=device)

    with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
        treescope.display(unflatten_dict(data))
