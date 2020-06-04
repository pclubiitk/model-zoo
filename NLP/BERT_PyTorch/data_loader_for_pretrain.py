import os
from pytorch_pretrained_bert.tokenization import BertTokenizer
import preprocess_pretraining 
import torch
from utils import seek_random_offset
from random import random as rand
from random import randint, shuffle


class DataLoader():
    """ Load sentence pair from corpus """
    def __init__(self, file, batch_size, max_len, short_sampling_prob=0.1):
        super().__init__()
        self.f_pos = open(file, "r", encoding='utf-8', errors='ignore')
        self.f_neg = open(file, "r", encoding='utf-8', errors='ignore') 
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len 
        self.short_sampling_prob = short_sampling_prob
        self.batch_size = batch_size
        self.preproc= preprocess_pretraining.PreProcess(max_len*0.15,0.15,max_len)

    def read_tokens(self, f, length, discard_last_and_restart=True):
        """ Read tokens from file pointer with limited length """
        tokens = []
        while len(tokens) < length:
            line = f.readline()
            if not line: # end of file
                return None
            if not line.strip(): 
                if discard_last_and_restart:
                    continue
                else:
                    return tokens 
            tokens.extend(self.tokenizer.tokenize(line.strip()))
            
        return tokens

    def __iter__(self): # iterator to load data
        while True:
            batch = []
            for i in range(self.batch_size):
             
                len_tokens = randint(1, int(self.max_len / 2)) \
                    if rand() < self.short_sampling_prob \
                    else int(self.max_len / 2)

                is_next = rand() < 0.5 # whether token_b is next to token_a or not

                tokens_a = self.read_tokens(self.f_pos, len_tokens, True)
                seek_random_offset(self.f_neg)
                f_next = self.f_pos if is_next else self.f_neg
                tokens_b = self.read_tokens(f_next, len_tokens, False)

                if tokens_a is None or tokens_b is None: 
                    self.f_pos.seek(0, 0)
                    return

                data = (is_next, tokens_a, tokens_b)
                data=self.preproc(data)
                
                batch.append(data)

            batch_tensors = [torch.tensor(x, dtype=torch.long) for x in zip(*batch)]
            yield batch_tensors

