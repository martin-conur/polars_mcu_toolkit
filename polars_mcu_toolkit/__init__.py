from __future__ import annotations
from typing import TYPE_CHECKING

import polars as pl
from pathlib import Path

from polars.plugins import register_plugin_function

if TYPE_CHECKING:
    from polars_mcu_toolkit.typing import IntoExprColumn

LIB = Path(__file__).parent


def neg_sample(expr: IntoExprColumn,
               sample_from: list[int],
               neg_ratio: int = 1) -> pl.Expr:
    """
    Given a Series of type List(Int64), sample the negative cases (values that 
    not appears on the List Series) from a list that holds all the values.

    Args:
        sample_from: List with all the values that the series could contain.
        neg_ratio: negative samples ratio with respect to the positive cases. 
        By default 1, which means that returns the same number of positive 
        samples. 2 returns twice as many positive samples, and so on.
    
    Returns:
        A List(i64) series with negative samples.ß
    """
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="neg_sample",
        is_elementwise=True,
        kwargs={"sample_from": sample_from, "neg_ratio": neg_ratio},
    )


def non_val_indices(expr: IntoExprColumn, val: int) -> pl.Expr:
    return register_plugin_function(
        args=[expr],
        plugin_path=LIB,
        function_name="non_val_indices",
        is_elementwise=True,
        kwargs={"val": val}
    )
