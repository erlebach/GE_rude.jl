import pandas as pd

fn = "dicts.yml"
df = pd.read_json(fn)
print(df)
