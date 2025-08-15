from typing import Any

from rich.console import Console
from rich.tree import Tree

FORMATTER_REGISTRY = {}


class LeaveFormatter:
    """"""

    pass


def array_values(value, color="cornsilk1"):
    """Print array stats"""
    import numpy as np

    with np.printoptions(precision=2, edgeitems=2, threshold=100):
        str_value = str(value)

    return f"\n[{color}]{str_value}[/{color}]"


def array_stats(value, color="pale_turquoise1", color_label="turquoise2"):
    """Print array stats"""
    import numpy as np

    str_value = f"[{color_label}]mean[/{color_label}]={np.mean(value):.2e}, [{color_label}]std[/{color_label}]={np.std(value):.2e}, "
    str_value += f"[{color_label}]min[/{color_label}]={np.min(value):.2e}, [{color_label}]max[/{color_label}]={np.max(value):.2e}"
    return f"[{color}]{str_value}[/{color}]" + array_values(value)


def print_tree(tree):
    """Print a PyTree to console"""
    tree = dict_to_tree(tree)
    console = Console()
    console.print(tree)


def dict_to_tree(data: dict[str, Any], root_label: str = "File") -> Tree:
    """Convert a dictionary to a rich.tree.Tree object using recursion."""
    tree = Tree(root_label)

    _add_dict_to_tree(tree, data)
    return tree


def _add_dict_to_tree(parent_node: Tree, data: dict[str, Any], format_leave) -> None:
    """Recursively add dictionary items to a tree node."""
    for key, value in data.items():
        if isinstance(value, dict):
            branch = parent_node.add(f"[bright_white]{key}[/bright_white]")
            _add_dict_to_tree(branch, value, format_leave)
        else:
            # Add leaf node for primitive values
            formatted_value = format_leave(value)
            parent_node.add(
                f"[dark_turquoise]{key} {value.dtype.str}({value.shape})[/dark_turquoise]: {formatted_value}"
            )
