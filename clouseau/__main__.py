#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import tyro
from rich import print_json
from safetensors import safe_open

from clouseau.io_utils import read_from_safetensors, unflatten_dict
from clouseau.visualize import (
    FORMATTER_REGISTRY,
    ArrayDiffFormatter,
    ArrayShapeFormatter,
    ArrayStatsFormatter,
    ArrayValuesFormatter,
    print_tree,
)


@dataclass
class Show:
    """Show contents of a single safetensors file"""

    filename: str
    key_pattern: str = ".*"  # regex pattern to match and select a path
    show_meta: bool = False  # show metadata and exit
    fmt_stats: ArrayStatsFormatter = field(default_factory=ArrayStatsFormatter)
    fmt_values: ArrayValuesFormatter = field(default_factory=ArrayValuesFormatter)


@dataclass
class Diff:
    """Compare two files and show differences"""

    filename: str
    filename_ref: str
    key_pattern: str = ".*"  # regex pattern to match and select a path
    fmt_diff: ArrayDiffFormatter = field(default_factory=ArrayDiffFormatter)


Commands = Show | Diff


def show(args: Show) -> None:
    """Show contents of a single file"""
    path = Path(args.filename)

    if args.show_meta:
        with safe_open(args.filename, framework="numpy") as f:
            print_json(data=f.metadata())
            return

    data = unflatten_dict(read_from_safetensors(path, key_pattern=args.key_pattern))

    FORMATTER_REGISTRY[np.ndarray] = (
        lambda _: ArrayShapeFormatter()(_)  # type: ignore[assignment]
        + "\n"
        + args.fmt_stats(_)
        + "\n"
        + args.fmt_values(_)
    )

    print_tree(data, label=f"File: [orchid]{path.name}[/orchid]")


def diff(args: Diff) -> None:
    """Compare two files and show differences"""
    path, path_ref = Path(args.filename), Path(args.filename_ref)

    data = read_from_safetensors(path, key_pattern=args.key_pattern)
    data_ref = read_from_safetensors(path_ref, key_pattern=args.key_pattern)

    if len(data) != len(data_ref):
        message = (
            f"Size of the sub-trees must match, got {len(data)} and {len(data_ref)}"
        )
        raise ValueError(message)

    FORMATTER_REGISTRY[tuple] = lambda _: args.fmt_diff(_)

    merged = {}

    for (key, value), value_ref in zip(data.items(), data_ref.values()):
        merged[key] = (value, value_ref)

    print_tree(merged, label=f"File: [orchid]{path.name}[/orchid]")


def main() -> None:
    args = tyro.cli(  # type: ignore[call-overload]
        Commands, description="Show and diff content of safetensors files"
    )

    if isinstance(args, Show):
        show(args)
    elif isinstance(args, Diff):
        diff(args)


if __name__ == "__main__":
    main()
