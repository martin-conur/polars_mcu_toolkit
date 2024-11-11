# This file is just for local testing

import polars as pl
import polars_mcu_toolkit as mcu
import random
import time


pl.Config().set_fmt_table_cell_list_len(10)

start_time = time.perf_counter()

N_ITEMS = 10_000
N_ROWS = 100_000
maestra = list(range(1, N_ITEMS))

random_lists = [[random.randint(0, N_ITEMS) for _ in range(random.randint(1, 5))] for _ in range(N_ROWS)]

df = pl.DataFrame({'dense': random_lists})

mid_time = time.perf_counter()

elapsed_mid = mid_time - start_time

print(f"df generation time: {elapsed_mid}")


print(df)
print(df.with_columns(indices=mcu.neg_sample('dense', sample_from=maestra)))

end_time = time.perf_counter()

elapsed_rust = end_time - mid_time

print(f"rust calculation time: {elapsed_rust}")
print(f"total time: {elapsed_mid + elapsed_rust}")