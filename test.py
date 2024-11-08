import polars as pl
import polars_negsample as mp

pl.Config().set_fmt_table_cell_list_len(10)

df = pl.DataFrame({'dense': [[0, 9], [8, 6, 0, 9], None, [3, 3]]})
print(df)
print(df.with_columns(indices=mp.non_val_indices('dense', val=0)))
print(df.with_columns(indices=mp.non_val_indices('dense', val=9)))