import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataloader import loadData, prepareData, loadBatches
from model import SkipGram
from utils import subsampling, negativeSampling, getSimilarWords, getAnalogy, plotData
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type = int, default = 50, help = "No of EPOCHS: default 5 ")
parser.add_argument('--sampling_threshold', type = int, default = 1e-5, help = "Subsampling Threshold, default 1e-5 ")
parser.add_argument('--min_freq', type = int, default = 5, help = "Minimum Frequency of words, default 5 ")
parser.add_argument('--batch_size', type = int, default = 512, help = "Batch size, default 512")
parser.add_argument('--window_size', type = int, default = 5, help = "Window size, default 5")
parser.add_argument('--lr', type = int, default = 0.003, help = "Learning rate for generator optimizer,default 0.003 ")
parser.add_argument('--embed_size', type = int, default = 300, help = "Embed size, default 300")
parser.add_argument('--mode', type = str, default = "train", help = "Mode Type, default train")
parser.add_argument('--N_words', type = int, default = 10, help = "Top N similar words, default 10")
parser.add_argument('--word', type = str, default = "woman", help = "word, default word is woman")
parser.add_argument('--word1', type = str, default = "king", help = "1st word, default king")
parser.add_argument('--word2', type = str, default = "male", help = "2nd word, default male")
parser.add_argument('--word3', type = str, default = "female", help = "3rd word, default female")
args = parser.parse_args()


with open('text8') as f:
    text = f.read().split()

tokenize_data = loadData("text8")
int_text, word2idx, idx2word, freq, vocab = prepareData(tokenize_data, args.min_freq)


if args.mode == "train":
	vocab_size = sum([freq[k] for k in freq])
	subsampled_words = subsampling(freq, args.sampling_threshold, vocab, vocab_size, word2idx)
	neg_sample = negativeSampling(freq)
	#print(neg_sample.shape)

	device='cpu'

	model = SkipGram(len(word2idx), args.embed_size, neg_sample).to(device)
	optimizer = optim.Adam(model.parameters(), args.lr)
	epoch = args.epochs
	steps = 0

	for i in range(epoch):

	  for input_words, target_words in loadBatches(subsampled_words, args.batch_size, args.window_size):
	    steps = steps + 1
	  
	    inputs = torch.LongTensor(input_words)
	    targets = torch.LongTensor(target_words)
	    
	    #inputs, targets = inputs.to(device), targets.to(device)
	    loss = model.forward(inputs, targets, inputs.shape[0], 2)

	    optimizer.zero_grad()
	    loss.backward()
	    optimizer.step()

	    if steps % 2000 == 0:
	      print("Epoch: {}/{}".format(i+1, epoch))
	      print("Loss: ", loss.item())
	      print("\n")

	path = 'model.pth'
	#torch.save(model.state_dict(), path)

elif args.mode == "topNsimilar":
	path = 'model.pth'
	state_dict = torch.load(path)
	embedding = state_dict['input_embedding.weight'].to('cpu')
	Words = getSimilarWords(embedding, args.word, args.N_words, idx2word, word2idx)
	print(", ".join(words))

elif args.mode == "Analogy":
	path = 'model.pth'
	state_dict = torch.load(path)
	embedding = state_dict['input_embedding.weight'].to('cpu')
	words = getAnalogy(embedding, args.word1, args.word2, args.word3, 5, idx2word, word2idx)
	print(words[1])

elif args.mode == "GetScore":
	path = 'model.pth'
	state_dict = torch.load(path)
	embedding = state_dict['input_embedding.weight'].to('cpu')
	print(getScore(embedding, args.word1, args.word2, idx2word, word2idx))
		
elif args.mode == "plot":
	path = 'model.pth'
	state_dict = torch.load(path)
	embedding = state_dict['input_embedding.weight'].to('cpu')
	plotData(embedding,idx2word)
