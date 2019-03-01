import pandas as pd
import numpy as np
import os

path = "pre_vote/final_output.csv"

top_1 = pd.read_csv(path)["1"].tolist()

new = top_1.count("new_whale")
print(new)
print(new/7960.0)