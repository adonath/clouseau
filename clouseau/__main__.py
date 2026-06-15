#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
    """Compare two files and show differences, aligning leaves by key/path"""
    path, path_ref = Path(args.filename), Path(args.filename_ref)

    data = read_from_safetensors(path, key_pattern=args.key_pattern)
    data_ref = read_from_safetensors(path_ref, key_pattern=args.key_pattern)

    FORMATTER_REGISTRY[tuple] = lambda _: args.fmt_diff(_)

    # Align leaves by key/path rather than by position. data_ref is the
    # reference, so iterate it first (it drives the ordering) and flag any leaf
    # that is missing from the file under test or only present in it.
    merged: dict[str, Any] = {}

    for key, value_ref in data_ref.items():
        if key in data:
            merged[key] = (data[key], value_ref)
        else:
            merged[key] = f"[yellow]missing in {path.name}[/yellow]"

    for key in data:
        if key not in data_ref:
            merged[key] = f"[yellow]only in {path.name}, not in reference[/yellow]"

    label = (
        f"File: [orchid]{path.name}[/orchid] "
        f"vs reference [orchid]{path_ref.name}[/orchid]"
    )
    print_tree(merged, label=label)


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
