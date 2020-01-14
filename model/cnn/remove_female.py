import pandas as pd


ref = pd.read_csv("input/arff_data_path_enhanced_m.csv")
ref.head()
colnames = ['labels', 'path']
ref2 = pd.read_csv("input/arff_data_m.csv", names=colnames)
"""
print(ref.head())


to_drop = []
counter = 0
for line in ref.labels:
    if "female" in line:
        to_drop.append(counter)
    counter += 1

ref.drop(to_drop, inplace=True)
print(ref.head())

ref.to_csv("input/Data_path_male.csv")
"""

colnames = ['labels', 'path']
ref = pd.read_csv("input/arff_data_m.csv")

with open("input/arff_data_m.csv", "r") as reff:
    for line in reff.readlines():
        print(line)
reff.close()