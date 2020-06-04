import torch
import torch.nn as nn
from random import randint, shuffle
from random import random as rand
from pytorch_pretrained_bert.tokenization import BertTokenizer
import random
import math
import os
import argparse
import model_pretrain
import pandas as pd
from utils import load

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
# model config
parser.add_argument('--dim', type=int, default=768)
parser.add_argument('--max_len', type=int, default=512)
parser.add_argument('--heads', type=int, default=12)
parser.add_argument('--n_segs', type=int, default=2)

parser.add_argument('--pretrain_file', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True)       #COLA dataset in csv format
parser.add_argument('--epochs', type=int, default=4)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.00002)
parser.add_argument('--beta1', type=float, default=0.9)
parser.add_argument('--beta2', type=float, default=0.999)
parser.add_argument('--decay', type=float, default=0.01)

args = parser.parse_args()

df = pd.read_csv(args.dataset, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
sentences = df.sentence.values
labels = df.label.values

train_sent=sentences[0:6000]
train_label=labels[0:6000]
test_sent=sentences[6000:]
test_label=labels[6000:]

class PreprocessCola():
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_len=512):
        super().__init__()
        
        self.indexer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len

    def __call__(self,data):
        token,label=data
        #truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + token + ['[SEP]'] 
        segment_ids = [0]*(len(token)+2)
        input_mask = [1]*len(tokens)

        # Token Indexing
        input_ids = self.indexer.convert_tokens_to_ids(tokens)
       

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*int(n_pad))
        segment_ids.extend([0]*int(n_pad))
        input_mask.extend([0]*int(n_pad))

        # Zero Padding for masked target
        

        return (input_ids, segment_ids, input_mask,label)
        
class DataLoaderCola():
    """ Load sentence pair from corpus """
    def __init__(self, sent,label, batch_size, max_len, short_sampling_prob=0.1):
        super().__init__()
        self.sent=sent
        self.label=label
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len 
        self.short_sampling_prob = short_sampling_prob
        self.batch_size = batch_size
        self.preproc= PreprocessCola(max_len)


    def __iter__(self): # iterator to load data
        k=0
        while True:
            batch = []
            for i in range(self.batch_size):
             
                len_tokens = randint(1, int(self.max_len / 2)) \
                    if rand() < self.short_sampling_prob \
                    else int(self.max_len / 2)


                tokens =self.tokenizer.tokenize( self.sent[k])
                label=self.label[k]
                k=k+1
                data = (tokens,label)
                data=self.preproc(data)
                if k>len(sentences):
                  return
                
                batch.append(data)

            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors

data_train=DataLoaderCola(train_sent,train_label,args.batch_size,args.max_len)
data_test=DataLoaderCola(test_sent,test_label,args.batch_size,args.max_len)

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

class ColaClassifier(nn.Module):
  def __init__(self,dim,heads,max_len,n_seg):
    super().__init__()
    self.allenc=model_pretrain.AllEncode(dim,heads,max_len,n_seg)
    self.fc1=nn.Linear(dim,dim)
    self.tanh=nn.Tanh()
    self.fc2=nn.Linear(dim,2)

  def forward(self,batch):
    input_ids, segment_ids, input_mask,label=batch
    out=self.allenc(input_ids,input_mask,segment_ids)

    out1=self.fc1(out[:,0])
    out1=self.tanh(out1)
    out1=self.fc2(out1)
    return out1

modelcls=ColaClassifier(args.dim,args.heads,args.max_len,args.n_segs).to(device)

criterion=nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.AdamW(modelcls.parameters(), lr=args.lr, betas=(args.beta1,args.beta2), weight_decay=0.01) 

load(args.pretrain_file,modelcls.allenc)

def loss_func(model,batch):
  input_ids, segment_ids, input_mask,label=batch
  clsf=model(batch)
  lossclf=criterion(clsf,label)
  return lossclf
  
for epoch in range(args.epochs):
  train_loss=0
  for i,batch in enumerate(data_train):
    batch = [t.to(device) for t in batch]
    optimizer.zero_grad()
    loss=loss_func(modelcls,batch)
    train_loss += loss.item()
    loss.backward()
    optimizer.step()
    loss_list.append
   
  avg_train_loss = train_loss / len(data_train) 
  print("  Average training loss: {0:.2f}".format(avg_train_loss))
  
  modelcls.eval()
  total_eval_accuracy = 0
  
  for batch in data_test:
    batch = [t.to(device) for t in batch]
    input_ids, segment_ids, input_mask,label=batch
    with torch.no_grad():     
      clsf=modelcls(batch)
  
    total_eval_accuracy += flat_accuracy(clsf, label)
    
  avg_val_accuracy = total_eval_accuracy / len(dat_test)
  print("  Accuracy: {0:.2f}".format(avg_val_accuracy))  
  
  
    
  
  
    
    
    
