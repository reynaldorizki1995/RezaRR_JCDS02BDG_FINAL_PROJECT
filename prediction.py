import pickle
from pandas import DataFrame, get_dummies
import pandas as pd

model = pickle.load(open('booster.sav','rb'))
one_hot_columns = pickle.load(open('features_dummies_colomn.sav','rb'))

def prediction(data):
    df = DataFrame(data, index=[0])
    df = get_dummies(df)
    df = df.reindex(columns = one_hot_columns, fill_value=0)
    hasil = model.predict(df)
    return round(hasil[0]) 

def data_list():
    df = pd.read_csv('clean_listing.csv')
    return df