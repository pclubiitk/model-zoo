import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def subsampling(freq, threshold, vocab, vocab_size, word2idx):
  p_freqs = { key:val/vocab_size for key,val in freq.items() }
  prob = { key:(val-threshold)/val - np.sqrt(threshold/val) for key,val in p_freqs.items() }
  subsampled_words = [ word for word in vocab if (1-prob[word]) > random.random() ]
  subsampled_words = [word2idx[word] for word in subsampled_words]
  return subsampled_words


def negativeSampling(freq):
  vocab_size = sum([freq[k] for k in freq])
  unigram = list(freq.values())
  unigram = np.asarray(unigram)/vocab_size
  neg_sample = torch.from_numpy(unigram)
  return neg_sample


def getSimilarWords(embedding, word, top, idx2word, word2idx):

  magnitudes = embedding.pow(2).sum(dim=1).sqrt().unsqueeze(0)
  valid_vectors = embedding[word2idx[word],:].unsqueeze(1)
  similar_words = torch.mm(valid_vectors.t(), embedding.t())/magnitudes
  score, closest = similar_words.topk(top)
  closest = closest.to('cpu')
  closest_words = []
  for idx in closest[0]:
    closest_words.append(idx2word[int(idx)])
  return closest_words


def getScore(embedding, word1, word2, idx2word, word2idx):

  magnitudes = embedding.pow(2).sum(dim=1).sqrt().unsqueeze(0)*10
  word1_vectors = embedding[word2idx[word1],:].unsqueeze(1)
  word2_vectors = embedding[word2idx[word2],:].unsqueeze(1)
  similar_words = torch.mm(word1_vectors.t(), word2_vectors)/magnitudes
  score, closest = similar_words.topk(2)

  return (np.array(score[0][0]))


def getAnalogy(embedding, word1, word2, word3, top, idx2word, word2idx):
  word1=word2idx[word1]
  word2=word2idx[word2]
  word3=word2idx[word3]
  analogy = embedding[word1,:].unsqueeze(1) - embedding[word2,:].unsqueeze(1) + embedding[word3,:].unsqueeze(1)
  magnitudes = embedding.pow(2).sum(dim=1).sqrt().unsqueeze(0)*10
  similarities = torch.mm(analogy.t(), embedding.t())/magnitudes
  score, closest = similarities.topk(top)
  analogies = []
  i=0
  for idx in closest[0]:
    ans ={}
    ans[idx2word[int(idx)]] = float(score[0][i])
    analogies.append(ans)
    i=i+1
  return analogies


def plotData(embedding, idx2word):

  words = 100
  tsne = TSNE()
  embed_tsne = tsne.fit_transform(embedding[:words, :]) 
  fig, ax = plt.subplots(figsize=(15, 15))
  for idx in range(words):
    plt.scatter(*embed_tsne[idx, :], color='green')
    plt.annotate(idx2word[idx], (embed_tsne[idx, 0], embed_tsne[idx, 1]), alpha=0.7)

