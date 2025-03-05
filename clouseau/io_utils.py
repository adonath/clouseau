import logging
from pathlib import Path
from typing import Any

from safetensors.flax import save_file as save_file_jax
from safetensors.torch import save_file as save_file_torch

log = logging.getLogger(__name__)

__all__ = [
    "read_from_safetensors",
    "save_to_safetensors_jax",
    "save_to_safetensors_torch",
    "unflatten_dict",
]

AnyArray = Any

PATH_SEP = "."


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


def read_from_safetensors(
    filename: str | Path, framework: str = "numpy", device: Any = None
) -> dict[str, Any]:
    """Read from safetensors"""
    from safetensors import safe_open

    with safe_open(filename, framework=framework, device=device) as f:
        # reorder according to metadata, which maps index to key / path
        keys = list(
            dict(sorted(f.metadata().items(), key=lambda _: int(_[0]))).values()
        )
        data = {key: f.get_tensor(key) for key in keys}

    return data
