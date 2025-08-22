from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

import numpy as np
from rich.console import Console
from rich.tree import Tree


@dataclass(frozen=True)
class ArrayDiffFormatter:
    """Array values formatter"""

    color_good: str = "green"
    color_bad: str = "bright_red"
    rtol: float = 1e-7
    atol: float = 0
    equal_nan: bool = True
    verbose: bool = False

    def __call__(self, value: np.ndarray, value_ref: np.ndarray) -> str:
        try:
            np.testing.assert_allclose(
                value,
                value_ref,
                atol=self.atol,
                rtol=self.rtol,
                equal_nan=self.equal_nan,
                verbose=self.verbose,
            )
        except AssertionError as e:
            message = f"[{self.color_bad}]{e}[/{self.color_bad}]"
        else:
            str_value = f"Equal to tolerance rtol={self.rtol}, atol={self.atol}"
            message = f"[{self.color_good}]{str_value}[/{self.color_good}]"

        return message


@dataclass(frozen=True)
class ArrayShapeFormatter:
    """Array values formatter"""

    color: str = "dark_turquoise"

    def __call__(self, value: np.ndarray) -> str:
        str_value = f"{value.dtype} {value.shape}"

        return f"[{self.color}]{str_value}[/{self.color}]"


@dataclass(frozen=True)
class ArrayValuesFormatter:
    """Array values formatter"""

    precision: int = 2
    edgeitems: int = 3
    threshold: int = 100
    color: str = "cornsilk1"

    def __call__(self, value: np.ndarray) -> str:
        with np.printoptions(
            precision=self.precision,
            edgeitems=self.edgeitems,
            threshold=self.threshold,
        ):
            str_value = str(value)

        return f"[{self.color}]{str_value}[/{self.color}]"


class StatsFuncEnum(StrEnum):
    """Stats func enum"""

    mean = "mean"
    std = "std"
    min = "min"
    max = "max"
    non_zero = "non_zero"
    is_nan = "is_nan"


STATS_FUNCS: dict[StatsFuncEnum, Callable[[np.ndarray], float]] = {
    StatsFuncEnum.mean: np.mean,
    StatsFuncEnum.std: np.std,
    StatsFuncEnum.min: np.min,
    StatsFuncEnum.max: np.max,
    StatsFuncEnum.non_zero: np.count_nonzero,
    StatsFuncEnum.is_nan: lambda x: np.sum(np.isnan(x)),
}


@dataclass(frozen=True)
class ArrayStatsFormatter:
    """Array stats formatter"""

    color: str = "pale_turquoise1"
    color_label: str = "turquoise2"
    stats: tuple[StatsFuncEnum, ...] = tuple(StatsFuncEnum)
    fmt: str = "{:.2e}"

    def __call__(self, value: np.ndarray) -> str:
        str_value = ""

        for stat in self.stats:
            func = STATS_FUNCS[stat]
            str_value += f"[{self.color_label}]{stat.name}[/{self.color_label}]={self.fmt.format(func(value))}, "

        return f"[{self.color}]{str_value}[/{self.color}]"


def format_np_array(value: np.ndarray) -> str:
    return ArrayStatsFormatter()(value) + "\n" + ArrayValuesFormatter()(value)


FORMATTER_REGISTRY = {np.ndarray: format_np_array}


def print_tree(tree_dict: dict[str, Any], label: str = "Tree") -> None:
    """Print a PyTree to console"""
    tree = dict_to_tree(tree_dict, label=label)
    console = Console()
    console.print(tree)


def dict_to_tree(data: dict[str, Any], label: str) -> Tree:
    """Convert a dictionary to a rich.tree.Tree object using recursion."""
    tree = Tree(label)

    _add_dict_to_tree(tree, data)
    return tree


def _add_dict_to_tree(parent_node: Tree, data: dict[str, Any]) -> None:
    """Recursively add dictionary items to a tree node."""
    for key, value in data.items():
        if isinstance(value, dict):
            branch = parent_node.add(f"[bright_white]{key}[/bright_white]")
            _add_dict_to_tree(branch, value)
        else:
            formatter = FORMATTER_REGISTRY.get(type(value))
            formatted_value = formatter(value) if formatter else str(value)
            parent_node.add(
                f"[dark_turquoise]{key}[/dark_turquoise]: {formatted_value}"
            )
