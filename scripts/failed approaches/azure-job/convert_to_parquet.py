import numpy as np
import pandas as pd

df = pd.read_csv(
    "behaviour.csv",
    usecols=["user_id","book_id"],
    dtype={
        "user_id":    str,
        "book_id":  str,
    },
    low_memory=False,
)
df.to_parquet("behaviour.parquet")
