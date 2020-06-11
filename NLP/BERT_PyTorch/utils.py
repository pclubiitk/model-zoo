from random import randint, shuffle
from random import random as rand
from pytorch_pretrained_bert.tokenization import BertTokenizer
import torch
import torch.nn as nn
import random
import math
import os

def seek_random_offset(f, back_margin=2000):
    """ seek random offset of file pointer """
    f.seek(0, 2)
    max_offset = f.tell() - back_margin
    f.seek(randint(0, max_offset), 0)
    f.readline() 
    
def truncate_tokens_pair(tokens_a, tokens_b, max_len):
    while True:
        if len(tokens_a) + len(tokens_b) <= max_len:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def get_random_word(vocab_words):
    i = random.randint(0,30000)
    return list(vocab_words)[i]
    
def gelu(x):
    "Implementation of the gelu activation function by Hugging Face"
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class LayerNorm(nn.Module):
    def __init__(self, dim, variance_epsilon=1e-12):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta  = nn.Parameter(torch.zeros(dim))
        self.variance_epsilon = variance_epsilon

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.gamma * x + self.beta
        
def save_model(i,model,dir):
        """ save current model """
        torch.save(model.allenc.state_dict(),os.path.join(dir, 'model_epoch_'+str(i)+'.pt'))

def load(model_file,model):
        if model_file:
            model.load_state_dict(torch.load(model_file))
