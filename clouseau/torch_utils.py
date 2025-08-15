import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .io_utils import PATH_SEP

CACHE: dict[str, list[torch.Tensor]] = {}

log = logging.getLogger(__file__)


def add_to_cache_torch(key: str) -> Callable:
    """Add a intermediate x to the global cache"""

    def hook(module: nn.Module, input_: Any, output: torch.Tensor) -> None:
        key_full = key + PATH_SEP + "__call__"
        CACHE.setdefault(key_full, [])
        if isinstance(output, tuple):
            log.warning(
                f"Output for `{key_full}` is a tuple, choosing first entry only."
            )
            output = output[0]

        CACHE[key_full].append(output.detach().clone())

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
        is_leaf = lambda p, _: _ is None

    def traverse(path: tuple[str, ...], node: Any) -> None:
        if is_leaf(path, node):
            return

        if filter_(path, node):
            name = PATH_SEP.join(path)
            hooks[name] = node.register_forward_hook(add_to_cache_torch(name))

        for p, child in node.named_children():
            traverse((*path, p), child)

    traverse(path=(), node=model)
    return model, hooks
