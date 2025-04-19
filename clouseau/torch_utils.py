from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .io_utils import PATH_SEP

CACHE: dict[str, torch.Tensor] = {}


def add_to_cache_torch(key: str) -> Callable:
    """Add a intermediate x to the global cache"""

    def hook(module: nn.Module, input_: Any, output: torch.Tensor) -> None:
        CACHE[key + PATH_SEP + "forward"] = output.detach().clone()

    return hook


def wrap_model(
    model: nn.Module,
    filter_: Callable[[tuple[str, ...], Any], bool] | None = None,
    is_leaf: Callable | None = None,
) -> tuple[nn.Module, dict[str, torch.utils.hooks.RemovableHandle]]:
    """Wrap model torch"""
    hooks: dict[str, torch.utils.hooks.RemovableHandle] = {}

    if filter_ is None:
        filter_ = lambda p, _: isinstance(_, nn.Module)

    if is_leaf is None:
        is_leaf = lambda p, _: False

    def traverse(path: tuple[str, ...], node: Any) -> None:
        if node is None or is_leaf(path, node):
            return

        if filter_(path, node):
            name = PATH_SEP.join(path)
            hooks[name] = node.register_forward_hook(add_to_cache_torch(name))

        for p, child in node.named_children():
            traverse((*path, p), child)

    traverse(path=(), node=model)
    return model, hooks
