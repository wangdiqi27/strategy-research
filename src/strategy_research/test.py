import numpy as np
import pandas as pd

df = pd.DataFrame(
    [{"can" : True},
     {"can" : False},
{"can" : np.nan},
     ]
)

arr = df.values

print(df)

if arr[1]:
    print("True")