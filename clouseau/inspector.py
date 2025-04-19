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
from enum import Enum
from pathlib import Path
from typing import Any, Callable

import treescope  # type: ignore[import-untyped]

from .io_utils import (
    read_from_safetensors,
    save_to_safetensors_jax,
    save_to_safetensors_torch,
    unflatten_dict,
)

DEFAULT_PATH = Path.cwd() / ".clouseau" / "trace.safetensors"

__all__ = ["magnify", "tail"]


log = logging.getLogger(__name__)

PATH_SEP = "."

AnyArray = Any
AnyModel = Any


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


WRITE_REGISTRY = {
    "jax": save_to_safetensors_jax,
    "torch": save_to_safetensors_torch,
}


class _Recorder:
    """Recorder class that can be used as a context manager."""

    def __init__(
        self,
        model: AnyModel,
        path: str | Path = DEFAULT_PATH,
        filter_: Callable[[tuple[str, ...], Any], bool] | None = None,
        is_leaf: Callable[[tuple[str, ...], Any], bool] | None = None,
    ):
        self.model = model
        self.path = Path(path)
        self.filter_ = filter_
        self.is_leaf = is_leaf
        self.hooks = None
        self.cache: dict[str, Any] = {}

    @property
    def framework(self) -> FrameworkEnum:
        """Determine framework"""

        if is_torch_model(self.model):
            return FrameworkEnum.torch
        elif is_jax_model(self.model):
            return FrameworkEnum.jax

        message = "The model does not seem to be a PyTorch or JAX model."
        raise ValueError(message)

    def __enter__(self) -> AnyModel:
        if self.framework == FrameworkEnum.jax:
            from . import jax_utils as utils
        elif self.framework == FrameworkEnum.torch:
            from . import torch_utils as utils  # type: ignore[no-redef]

        self.cache = utils.CACHE
        wrapped_model, self.hooks = utils.wrap_model(
            model=self.model, filter_=self.filter_, is_leaf=self.is_leaf
        )
        return wrapped_model

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)

        if not self.cache:
            log.warning("No arrays were recorded. Check the filter function.")

        WRITE_REGISTRY[self.framework](self.cache, self.path)
        self.cache.clear()

        if self.hooks:
            for _, hook in self.hooks.items():
                hook.remove()


def tail(
    model: AnyModel,
    path: str | Path = DEFAULT_PATH,
    filter_: Callable[[Any, Any], bool] | None = None,
    is_leaf: Callable[[Any, Any], bool] | None = None,
) -> _Recorder:
    """Tail and record the forward pass of a model

    Parameters
    ----------
    model : object
        The model to inspect. Can be a PyTorch model or JAX/Equinox model.
    path : str or Path
        Path where to store the forward pass arrays.
    filter_ : callable
        Function that filters which tensors to inspect.
        Takes the pytree leaves, child modules as input and returns a boolean.
    is_leaf : callable, optional
        Function that determines whether a node in the model tree should be treated as a leaf.
        Takes a node as input and returns a boolean. If True, the node will not be traversed further.
        This is particularly useful for JAX/Equinox models to control the granularity of inspection.

    Returns
    -------
    _Inspector
        Inspector instance that can be used as a context manager.

    Examples
    --------
    >>> import torch
    >>> from clouseau import inspector, magnifier
    >>> model = torch.nn.Linear(10, 5)
    >>> with inspector.tail(model,  path=".clouseau/trace-torch.safetensors") as fmodel:
    ...     out = fmodel(torch.randn(3, 10))

    When working with a JAX/Equinox model, it is important to add `.block_until_ready()`
    >>> import jax
    >>> with inspector.tail(model, path=".clouseau/trace-jax.safetensors") as fmodel:
    ...     fmodel(x, time).block_until_ready()

    """
    return _Recorder(model=model, path=path, filter_=filter_, is_leaf=is_leaf)


def magnify(
    filename: str | Path = DEFAULT_PATH, framework: str = "numpy", device: Any = None
) -> None:
    """Visualize nested arrays using treescope"""
    data = read_from_safetensors(filename, framework=framework, device=device)

    with treescope.active_autovisualizer.set_scoped(treescope.ArrayAutovisualizer()):
        treescope.display(unflatten_dict(data))
