import logging
from collections.abc import Callable
from typing import Any

import torch
from torch import nn

from .io_utils import PATH_SEP, ArrayCache, flatten_output

log = logging.getLogger(__file__)


def add_to_cache_torch(key: str, cache: ArrayCache) -> Callable:
    """Add a intermediate x to the given cache"""

    def hook(module: nn.Module, input_: Any, output: Any) -> None:
        key_full = key + PATH_SEP + "__call__"

        for sub_key, value in flatten_output(output, key_full):
            if not isinstance(value, torch.Tensor):
                log.debug(f"Skipping non tensor output for `{sub_key}`")
                continue

            # move to host so accumulated activations do not pile up in device memory;
            # copy=True guarantees an independent buffer even when already on CPU
            # atleast_1d gives scalar outputs an axis to be concatenated along
            value = torch.atleast_1d(value.detach())
            cache.add(sub_key, value.to("cpu", copy=True))

    return hook


def wrap_model(
    model: nn.Module,
    cache: ArrayCache,
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
        if is_leaf(path, node):  # type: ignore[call-non-callable]
            return

        if filter_(path, node):  # type: ignore[call-non-callable]
            name = PATH_SEP.join(path)
            hooks[name] = node.register_forward_hook(add_to_cache_torch(name, cache))

        for p, child in node.named_children():
            traverse((*path, p), child)

    traverse(path=(), node=model)
    return model, hooks
