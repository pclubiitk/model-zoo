import numpy as np
from keras.utils import to_categorical

def OneHotOfAllInVocab(sentences_list):
    
  vocab = set()
  for line in sentences_list:
    for word in line:
      vocab.add(word)
  vocabSize = len(vocab)
  print("there are total {} number of words in whole  vocab".format(vocabSize))

  word_to_id = {token: idx for idx, token in enumerate(vocab)}
  id_to_word = {idx: token for idx, token in enumerate(vocab)}

  vec = [word_to_id[word] for word in vocab]
  vec = to_categorical(vec)

  print(vec[0])
  data = {
      "vec":vec,
      "word_to_id":word_to_id,
      "id_to_word":id_to_word,
      "vocabSize":vocabSize,
      "vocab":vocab
  }
  return data

def constructBagOfWordsInWindowSize(corpus):
    context_tuple_list = []
    w = 4

    for line in corpus:
        for i, word in enumerate(line):
            first_context_word_index = max(0, i-w)
            last_context_word_index = min(i+w, len(line))
            for j in range(first_context_word_index, last_context_word_index):
                if(i!=j):
                    context_tuple_list.append((word, line[j]))
    return context_tuple_list
    print("there are {} pairs of target and context words".format(len(context_tuple_list)))

def contextPairToOneHot(context_tuple_list, sentences_list):
    data = OneHotOfAllInVocab(sentences_list)
    vec = data["vec"] 
    word_to_id = data["word_to_id"]
    oneHotList = []
    for word, context in context_tuple_list:
        oneHotList.append(np.array([vec[word_to_id[word]], vec[word_to_id[context]]]))
  
  #convert to numpy array
    oneHotNumpy=np.array(oneHotList)
    return oneHotNumpy, data