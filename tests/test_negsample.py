import polars as pl
from polars_mcu_toolkit import neg_sample


def test_negsample():

    superset = [0, 1, 2, 3]
    df = pl.DataFrame({'dense': [[0, 1], [2, 3]]})
    neg_sampled_df = df.with_columns(
        dense=neg_sample("dense", superset).list.sort()
    )

    expected_df = pl.DataFrame(
        {
            "dense": [[2, 3], [0, 1]]
        }
    )

    neg_sampled_df.equals(expected_df)
    assert neg_sampled_df.equals(expected_df)
