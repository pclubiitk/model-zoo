import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn import preprocessing
import argparse

def getSimilarity(word1, word2, data, emb):
    word_to_id = data["word_to_id"]
    word1_emb = emb[word_to_id[word1],:]
    word2_emb = emb[word_to_id[word2],:]

    similarity = np.dot(word1_emb,word2_emb.T)/(np.abs(np.dot(word1_emb,word1_emb.T))*np.abs(np.dot(word2_emb,word2_emb.T)))
    return similarity

def getSimilarityByEmbedding(emb1, emb2):
    similarity = np.dot(emb1,emb2.T)/(np.abs(np.dot(emb1,emb1.T))*np.abs(np.dot(emb2,emb2.T)))
    return similarity

def getTenClosestWords(search, vocab, data, emb):
    topTen = list()
    for word in vocab:
        topTen.append([word, getSimilarity(search, word, data, emb)])
    topTen.sort(key = lambda x: x[1],reverse=True)
    return topTen[:10]   
  
def analogy(word1, word2, word3, data, vocab, emb):
    word_to_id = data["word_to_id"]
    word4_emb = emb[word_to_id[word1],:] - emb[word_to_id[word2],:] + emb[word_to_id[word3],:]

    topTen = list()
    for word in vocab:
        topTen.append([word, getSimilarityByEmbedding(word4_emb,emb[word_to_id[word]])])
    topTen.sort(key = lambda x: x[1],reverse=True)
    return topTen[:10] 

def plotEmbeddingsIn2D(emb, data):
    plt.figure(figsize=(10,20))
    word_to_id = data["word_to_id"]
    vocab = list(data["vocab"])[:100]
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    vectors = model.fit_transform(emb)
    normalizer = preprocessing.Normalizer()
    vectors =  normalizer.fit_transform(vectors, 'l2')
    fig, ax = plt.subplots()
    for word in vocab:
        print(word, vectors[word_to_id[word]][1])
        ax.annotate(word, (vectors[word_to_id[word]][0],vectors[word_to_id[word]][1] ))
    plt.show()
