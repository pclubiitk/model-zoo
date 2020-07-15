import random
import re
import numpy as np
from utils import get_segments, get_ids, create_padding_mask

def preprocess(file, BATCH_SIZE, max_length, tokenizer):
  train_dataset = []
  input_vocab_size = len(tokenizer.vocab)
  f = open(file, 'r')
    
  words = f.read()
    
  words = words.replace('\n\n', '.')
  words = words.replace('\n', ' ')
  words = re.split('[;:.!?]', words)
  
  i = 0
  for _ in range(len(words)//BATCH_SIZE + 1):
    if i + 1 >= len(words):
        break
    input_ids_list = []
    segment_list = []
    is_masked_list = []
    is_next_list = []
  
    for j in range(BATCH_SIZE):
      if i + 1 >= len(words):
        break
    
      now = int(random.random() > 0.5) # decide if the 2nd sentence has to be next sentence or not
        
      if now == 1:
        res = ["[CLS]"] + tokenizer.tokenize(words[i]) + ["[SEP]"] + tokenizer.tokenize(words[i+1]) + ["[SEP]"]
      else:
        res = ["[CLS]"] + tokenizer.tokenize(words[i]) + ["[SEP]"] + tokenizer.tokenize(words[random.randint(0, len(words) - 1)]) + ["[SEP]"]
          
      input_ids = get_ids(res,tokenizer, max_length)
      segment_list.append(get_segments(res, max_length))
      is_next_list.append(now)
      is_masked = [0]*max_length
        
      for ind in range(max_length):
        if input_ids[ind] == 0: # is padding token appears, then break
          break
        if input_ids[ind] == 101 or input_ids[ind] == 102: # don't mask [CLS] and [SEP] tokens
          continue
        if random.random() < 0.15: # mask 15% of tokens
          is_masked[ind] = input_ids[ind]
          if random.random() < 0.8: # out of 15%, mask 80%
            input_ids[ind] = 103
          elif random.random() < 0.5: # replace 10% with random token
            input_ids[ind] = random.randint(1000, input_vocab_size)
            #in the remaining tokens, keep the same token
      input_ids_list.append(input_ids)
      is_masked_list.append(is_masked)
      if now == 1:
        i += 2
      else:
        i += 1
      
    input_ids_list = np.array(input_ids_list)
    is_masked_list = np.array(is_masked_list)
    masks = create_padding_mask(input_ids_list)
    segment_list = np.array(segment_list)
    is_next_list = np.array(is_next_list)
    is_next_list = np.reshape(is_next_list, (len(is_next_list), 1))
    train_dataset.append([input_ids_list, segment_list, masks, is_next_list, is_masked_list])
    
  return train_dataset
      
      

