import numpy as np
import pandas as pd

def labelling(df):
   df["temp"] =  (df["Adj Close"].shift(-1) / df["Adj Close"] ) - 1

   df["label"] = 0
   df.loc[df["temp"] > 0.55/100, "label"] = 1
   df.loc[df["temp"] < -0.50/100, "label"] = -1
    
   df.drop(df[df["label"] == 0].index, inplace= True)
   del df["temp"]

def generate_sequences(df, features_columns, T):
    X_stock_array = np.array(df[features_columns])
    y_stock_array = np.array(df["label"])
    sequences_indexes = [np.arange(i, T + i, 1) for i in range(len(df) - T)]
    _X = X_stock_array[sequences_indexes]
    _y = y_stock_array[sequences_indexes][:, -1]
    return _X, _y

def shuffled_X_y(X, y):
    assert len(X) == len(y)
    p = np.random.permutation(len(X))
    return X[p], y[p]