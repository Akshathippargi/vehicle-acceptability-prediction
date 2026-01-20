import pandas as pd

def load_data(path):
    columns = ["buying","maint","doors","persons","lug_boot","safety","class"]
    return pd.read_csv(path, names=columns)
