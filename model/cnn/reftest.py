import pandas as pd
from keras.utils import np_utils, to_categorical


ref = pd.read_csv("input/Data_path.csv")
ref.head()
print(ref)