import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

__all__ = [
    "read_from_safetensors",
    "save_to_safetensors_jax",
    "save_to_safetensors_torch",
    "unflatten_dict",
]

AnyArray = Any

PATH_SEP = "."


class FrameworkEnum(str, Enum):
    """Framework enum"""

    jax = "jax"
    torch = "torch"


def unflatten_dict(d: dict[str, Any], sep: str = PATH_SEP) -> dict[str, Any]:
    """Unflatten dictionary"

    Taken from https://stackoverflow.com/a/6037657/19802442
    """
    result: dict[str, Any] = {}

    for key, value in d.items():
        parts = key.split(sep)
        d = result
        for part in parts[:-1]:
            if part not in d:
                d[part] = {}
            d = d[part]
        d[parts[-1]] = value

    return result


def save_to_safetensors_jax(x: dict[str, list[AnyArray]], filename: str | Path) -> None:
    """Safetensors I/O for jax"""
    from jax import numpy as jnp
    from safetensors.flax import save_file as save_file_jax

    log.info(f"Writing {filename}")
    # safetensors does not support ordered dicts, see https://github.com/huggingface/safetensors/issues/357
    order = {str(idx): key for idx, key in enumerate(x.keys())}

    x_concat: dict[str, AnyArray] = {
        key: jnp.concatenate(value) for key, value in x.items()
    }

    save_file_jax(x_concat, filename, metadata={"order": json.dumps(order)})


def save_to_safetensors_torch(
    x: dict[str, list[AnyArray]], filename: str | Path
) -> None:
    """Safetensors I/O for torch"""
    import torch
    from safetensors.torch import save_file as save_file_torch

    log.info(f"Writing {filename}")
    # safetensors does not support ordered dicts, see https://github.com/huggingface/safetensors/issues/357
    order = {str(idx): key for idx, key in enumerate(x.keys())}

    x_concat: dict[str, torch.Tensor] = {
        key: torch.cat(value, dim=0) for key, value in x.items()
    }

    save_file_torch(x_concat, filename, metadata={"order": json.dumps(order)})


def read_from_safetensors(
    filename: str | Path,
    framework: str = "numpy",
    device: Any = None,
    key_pattern: str = ".*",
) -> dict[str, Any]:
    """Read from safetensors"""
    from safetensors import safe_open

    with safe_open(filename, framework=framework, device=device) as f:
        # reorder according to metadata, which maps index to key / path
        # note this triggers the lazy loading and immediately loads all data into memory
        order = f.metadata().get("order")

        if order is not None:
            order = json.loads(order)
            keys = list(dict(sorted(order.items(), key=lambda _: int(_[0]))).values())
        else:
            keys = f.keys()

        data = {key: f.get_tensor(key) for key in keys if re.search(key_pattern, key)}

    return data


@dataclass
class ArrayCache:
    """Simple cache for dict of lists of arrays with automatic safetensors saving."""

    framework: FrameworkEnum
    max_size_bytes: int = 1024**3
    path: str = ".clouseau"
    filename_pattern: str = "activations-{idx:03d}.safetensors"
    _current_size: int = 0
    _data: dict[str, list[AnyArray]] = field(default_factory=lambda: defaultdict(list))
    _file_counter = 0

    def __post_init__(self):
        self.path = Path(self.path)

    @property
    def data(self):
        """Data"""
        return self._data

    @property
    def current_size(self):
        """Current size"""
        return self._current_size

    @property
    def file_counter(self):
        """Current size"""
        return self._file_counter

    @property
    def current_size_mb(self) -> float:
        """Current cache size in MB."""
        return self.current_size / (1024 * 1024)

    def add(self, key: str, array) -> None:
        """Add array to cache. Returns True if cache was flushed."""
        array_size = array.nbytes

        # Flush if adding would exceed limit
        if self.current_size + array_size > self.max_size_bytes:
            self._flush()

        # Add to cache
        self._data[key].append(array)
        self._current_size += array_size

    def _flush(self):
        """Save cache to safetensors and clear."""
        if not self.data:
            return

        filename = self.path / self.filename_pattern.format(idx=self.file_counter)

        WRITE_REGISTRY[self.framework](self._data, filename)

        self._data.clear()
        self._current_size = 0
        self._file_counter += 1

    def flush(self):
        """Force save current cache."""
        self._flush()

    def flush_final(self):
        """Force save current cache."""
        self._flush()
        self._file_counter = 0


WRITE_REGISTRY = {
    FrameworkEnum.jax: save_to_safetensors_jax,
    FrameworkEnum.torch: save_to_safetensors_torch,
}
