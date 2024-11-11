from __future__ import annotations
from typing import TYPE_CHECKING

import polars as pl
from pathlib import Path

from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars_mcu_toolkit.typing import IntoExprColumn

LIB = Path(__file__).parent


def neg_sample(expr: IntoExprColumn, sample_from: list[int]) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="neg_sample",
        is_elementwise=True,
        kwargs={"sample_from": sample_from},
    )


def non_val_indices(expr: IntoExprColumn, val: int) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="non_val_indices",
        is_elementwise=True,
        kwargs={"val": val}
    )
