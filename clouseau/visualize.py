from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable

import numpy as np
from rich.console import Console
from rich.tree import Tree


@dataclass(frozen=True)
class ArrayDiffFormatter:
    """Quantitative diff between an array and a reference array.

    Reports the *magnitude* of the divergence (max absolute and max relative
    difference) and how many elements exceed the tolerance, instead of a bare
    pass/fail. This is far more useful for cross-framework comparisons, where
    float32 results routinely differ by a few ulp due to differing accumulation
    order and an exact match is the exception rather than the rule. The relative
    difference is taken against the reference (the second entry of the tuple).

    The defaults match the float32 tolerances used by ``torch.testing.assert_close``.
    """

    color_good: str = "green"
    color_bad: str = "bright_red"
    rtol: float = 1.3e-6
    atol: float = 1e-5
    equal_nan: bool = True
    fmt: str = "{:.2e}"

    def __call__(self, value_tuple: tuple[np.ndarray, np.ndarray]) -> str:
        value, value_ref = (np.asarray(_) for _ in value_tuple)

        if value.shape != value_ref.shape:
            message = f"shape mismatch: {value.shape} vs reference {value_ref.shape}"
            return f"[{self.color_bad}]{message}[/{self.color_bad}]"

        abs_diff = np.abs(value - value_ref)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_diff = abs_diff / np.abs(value_ref)

        finite_abs = abs_diff[np.isfinite(abs_diff)]
        finite_rel = rel_diff[np.isfinite(rel_diff)]
        max_abs = float(finite_abs.max()) if finite_abs.size else 0.0
        max_rel = float(finite_rel.max()) if finite_rel.size else 0.0

        close = np.isclose(
            value, value_ref, rtol=self.rtol, atol=self.atol, equal_nan=self.equal_nan
        )
        n_exceed = int((~close).sum())

        color = self.color_good if n_exceed == 0 else self.color_bad
        label = "close" if n_exceed == 0 else "differs"

        message = (
            f"{label} (max_abs={self.fmt.format(max_abs)}, "
            f"max_rel={self.fmt.format(max_rel)}, "
            f"{n_exceed}/{value.size} exceed rtol={self.rtol}, atol={self.atol})"
        )
        return f"[{color}]{message}[/{color}]"


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


FORMATTER_REGISTRY: dict[type, Callable[[Any], str]] = {np.ndarray: format_np_array}


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
