import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import glob
import datetime
from sklearn.preprocessing import robust_scale



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

def main():
    T = 15
    date_limit_train_validation = datetime.datetime(year = 2015, month=8, day=3)#2015-08-03
    date_limit_validation_test= datetime.datetime(year=2015, month=10, day=1 )#2015-10-01
    features = {
        "preprocessed_open": lambda df : df["Open"] / df["Close"] - 1,
        "preprocessed_high": lambda df : df["High"] / df["Close"] - 1,
        "preprocessed_low": lambda df : df["Low"] / df["Close"] - 1,
        "preprocessed_close": lambda df : df["Close"].pct_change(),
        "preprocessed_adj_close": lambda df : df["Adj Close"].pct_change(),
        "preprocessed_5-day": lambda df : (df["Adj Close"].rolling(5).mean() / df["Adj Close"]) -1,
        "preprocessed_10-day": lambda df : (df["Adj Close"].rolling(10).mean() / df["Adj Close"]) -1,
        "preprocessed_15-day": lambda df : (df["Adj Close"].rolling(15).mean() / df["Adj Close"]) -1,
        "preprocessed_20-day": lambda df : (df["Adj Close"].rolling(20).mean() / df["Adj Close"]) -1,
        "preprocessed_25-day": lambda df : (df["Adj Close"].rolling(25).mean() / df["Adj Close"]) -1,
        "preprocessed_30-day": lambda df : (df["Adj Close"].rolling(30).mean() / df["Adj Close"]) -1,
        "preprocessed_volume": lambda df : df["Volume"]
    }

    raw_data_path = "data/stocknet-dataset/price/raw/*.csv"
    raw_data_pathes = glob.glob(raw_data_path)

    X_train, y_train = None, None
    X_validation, y_validation = None, None
    X_test, y_test = None, None
    #Iterate through each stock RAW data
    for path in raw_data_pathes: 
        stock_df = pd.read_csv(path, parse_dates=["Date"],index_col="Date")
        
        ## Generate labels
        stock_df["temp"] =  (stock_df["Adj Close"].shift(-1) / stock_df["Adj Close"] ) - 1

        stock_df["label"] = -1
        stock_df.loc[stock_df["temp"] > 0.55/100, "label"] = 1
        stock_df.loc[stock_df["temp"] < -0.50/100, "label"] = 0

        stock_df.drop(stock_df[stock_df["label"] == -1].index, inplace= True)
        del stock_df["temp"]

        ## Preprocessing
        stock_df.sort_index(inplace= True)
        stock_df.dropna(inplace = True)

        ### Apply features functions
        for feature_key in features.keys(): 
            stock_df[feature_key] = robust_scale(features[feature_key](stock_df))
        stock_df.dropna(inplace = True)

        train_stock_df = stock_df[stock_df.index < date_limit_train_validation]
        validation_stock_df = stock_df[(stock_df.index >= date_limit_train_validation) & (stock_df.index < date_limit_validation_test) ]
        test_stock_df = stock_df[(stock_df.index >= date_limit_validation_test) ]

        ## Generate sequences
        X_stock_train, y_stock_train = generate_sequences(df = train_stock_df, features_columns= features.keys(), T = T )
        X_stock_validation, y_stock_validation = generate_sequences(df = validation_stock_df, features_columns= features.keys(), T = T)
        X_stock_test, y_stock_test = generate_sequences(df = test_stock_df, features_columns= features.keys(), T = T)

        # Adding X_stock and y_stock to the main X and y
        if X_train is None: X_train = X_stock_train
        else : X_train = np.concatenate([X_train, X_stock_train], axis = 0)
        if X_validation is None: X_validation = X_stock_validation
        else : X_validation = np.concatenate([X_validation, X_stock_validation], axis = 0)
        if X_test is None: X_test = X_stock_test
        else : X_test = np.concatenate([X_test, X_stock_test], axis = 0)
        if y_train is None: y_train = y_stock_train
        else : y_train = np.concatenate([y_train, y_stock_train], axis = 0)
        if y_validation is None: y_validation = y_stock_validation
        else : y_validation = np.concatenate([y_validation, y_stock_validation], axis = 0)
        if y_test is None: y_test = y_stock_test
        else : y_test = np.concatenate([y_test, y_stock_test], axis = 0)


    # Shuffle X and y
    X_train, y_train = shuffled_X_y(X_train, y_train)
    X_validation, y_validation = shuffled_X_y(X_validation, y_validation)
    X_test, y_test = shuffled_X_y(X_test, y_test)

    with open('preprocessed_data.npz', 'wb') as f:
        np.save(f, X_train, allow_pickle=True)
        np.save(f, y_train, allow_pickle=True)
        np.save(f, X_validation, allow_pickle=True)
        np.save(f, y_validation, allow_pickle=True)
        np.save(f, X_test, allow_pickle=True)
        np.save(f, y_test, allow_pickle=True)

if __name__ == "__main__":
    main()