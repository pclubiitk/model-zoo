import pandas as pd

# Downloading the dataset
#!kaggle datasets download -d abhinavwalia95/entity-annotated-corpus
# unzip the dataset and keep it in a folder named entity-annotaed-corpus

def load_data():
    data_train = pd.read_csv("./entity-annotated-corpus/ner.csv",engine="python")
    data_train = data_train.fillna(method="ffill")
    
    return data_train