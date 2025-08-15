#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tyro
from rich import print_json
from safetensors import safe_open

from clouseau.io_utils import read_from_safetensors, unflatten_dict
from clouseau.visualize import print_tree


@dataclass
class Show:
    """Show contents of a single safetensors file"""

    filename: str
    key_pattern: str = ".*"
    show_meta: bool = False


@dataclass
class Diff:
    """Compare two files and show differences"""

    filename: str
    filename_other: str


Commands = Show | Diff


def show(args: Show) -> None:
    """Show contents of a single file"""
    path = Path(args.filename)
    data = unflatten_dict(read_from_safetensors(path, key_pattern=args.key_pattern))

    if args.show_meta:
        with safe_open(args.filename, framework="numpy") as f:
            print_json(data=f.metadata())
            return

    print_tree(data, label=f"File: [orchid]{path.name}[/orchid]")


def diff(args: Diff) -> None:
    """Compare two files and show differences"""
    # TODO: Implement diff functionality
    print(f"Diffing files: {args.filename} and {args.filename_other}")


def main() -> None:
    args = tyro.cli(Commands, description="Show and diff content of safetensors files")

    if isinstance(args, Show):
        show(args)
    elif isinstance(args, Diff):
        diff(args)


if __name__ == "__main__":
    main()
