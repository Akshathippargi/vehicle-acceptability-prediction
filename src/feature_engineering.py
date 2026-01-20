import pandas as pd

def encode_features(df):
    return pd.get_dummies(df, drop_first=True)
