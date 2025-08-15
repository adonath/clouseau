#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import Any

from rich import print_json
from safetensors import safe_open

from clouseau.io_utils import read_from_safetensors, unflatten_dict
from clouseau.visualize import print_tree


def show(args: Any) -> None:
    """Show contents of a single file"""
    path = Path(args.filename)
    data = unflatten_dict(read_from_safetensors(path, key_pattern=args.key_pattern))

    if args.show_meta:
        with safe_open(args.filename, framework="numpy") as f:
            print_json(data=f.metadata())
            return

    print_tree(data, label=f"File: [orchid]{path.name}[/orchid]")


def diff(args: Any) -> None:
    """Compare two files and show differences"""
    # TODO: Implement diff functionality
    print(f"Diffing files: {args.file1} and {args.file2}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show and diff content of safetensors files"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Show subcommand
    show_parser = subparsers.add_parser("show", help="Show contents of a file")
    show_parser.add_argument("filename", help="File to display")
    show_parser.add_argument(
        "--key-pattern", help="Regex to select the key paths to show", default=".*"
    )
    show_parser.add_argument(
        "--show-meta", action="store_true", help="Show metadata of the file"
    )
    show_parser.set_defaults(func=show)

    # Diff subcommand
    diff_parser = subparsers.add_parser("diff", help="Compare two files")
    diff_parser.add_argument("filename", help="First file to compare")
    diff_parser.add_argument("filename_other", help="Second file to compare")
    diff_parser.set_defaults(func=diff)

    args = parser.parse_args()

    # Call the appropriate function
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
