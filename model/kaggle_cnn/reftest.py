import pandas as pd



ref = pd.read_csv("input/Data_path.csv")
ref.head()

for re in ref.labels:
    print(re)