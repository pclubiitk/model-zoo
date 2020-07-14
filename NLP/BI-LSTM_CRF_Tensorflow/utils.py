from tensorflow.keras.preprocessing.sequence import pad_sequences 
import tensorflow as tf
import json,os

class modifying(object):
    
    def __init__(self, data):
        self.n_sent = 1
        self.data = data
        self.empty = False

        agg_func = lambda s: [(w, p, t) for w, p, t in zip(s["Word"].values.tolist(),
                                                           s["POS"].values.tolist(),
                                                           s["Tag"].values.tolist())]
        self.grouped = self.data.groupby("Sentence #").apply(agg_func)
        self.sentences = [s for s in self.grouped]
        
        self.tags = list(set(self.data['Tag'].values))
        self.words = list(set(self.data['Word'].values))
        
       

    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

    def indexing(self):
       
       word2idx = {w: i + 1 for i, w in enumerate(self.words)}
       word2idx['PAD'] =0

       tag2idx = {t: i+1 for i, t in enumerate(self.tags)}
       tag2idx['PAD'] = 0

       return  tag2idx,len(self.tags)+1,word2idx,len(self.words)+1

    def padding(self,max_len,word2idx,tag2idx):
   
       X = [[word2idx[w[0]] for w in s] for s in self.sentences]
       X = pad_sequences(maxlen=max_len,sequences=X,padding='post',value=word2idx['PAD'])

       Y = [[tag2idx[w[2]] for w in s] for s in self.sentences]
       Y = pad_sequences(maxlen=max_len, sequences=Y, padding="post", value=tag2idx["PAD"])      
       return X,Y



