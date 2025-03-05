
from collections.abc import Callable

from torch import nn

from .io_utils import PATH_SEP

CACHE = {}

def add_to_cache_torch(key: str):
    """Add a intermediate x to the global cache"""

    def hook(module, input_, output):
        CACHE[key + PATH_SEP + "forward"] = output.detach().clone()

    return hook


def wrap_model(model: nn.Module, filter_: Callable | None = None):
    """Wrap model torch"""
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
    return model, hooks

