import numpy as np
import torch
import torch.nn as nn

class SkipGram(nn.Module):

  def __init__(self, vocab_size, embedding_layer_size, uniform_dist):

    super().__init__()

    self.vocab_size = vocab_size
    self.embedding_layer_size = embedding_layer_size
    self.uniform_dist = uniform_dist

    self.input_embedding = nn.Embedding(vocab_size, embedding_layer_size)
    self.output_embedding = nn.Embedding(vocab_size, embedding_layer_size)

    self.input_embedding.weight.data.uniform_(-1/embedding_layer_size,1/embedding_layer_size)
    self.output_embedding.weight.data.uniform_(-1/embedding_layer_size,1/embedding_layer_size)

  def forward(self, input_batch, output_batch, input_size, num_samples):

    i_embed = self.input_embedding(input_batch)
    o_embed = self.output_embedding(output_batch)
    
    #negative_mask = torch.multinomial(self.uniform_dist, input_size*num_samples, replacement=True).to(device)
    negative_mask = torch.multinomial(self.uniform_dist, input_size*num_samples, replacement=True)
    n_embed = self.output_embedding(negative_mask)
    
    n_embed = n_embed.view(input_size, num_samples, self.embedding_layer_size )
    pos_loss = torch.bmm(o_embed.view(input_size, 1, self.embedding_layer_size), i_embed.view(input_size, self.embedding_layer_size, 1)).sigmoid().log()
    neg_loss = torch.bmm(n_embed.neg(), i_embed.view(input_size, self.embedding_layer_size, 1)).sigmoid().log()
    pos_loss = pos_loss.squeeze()
    neg_loss = neg_loss.squeeze().sum(1)
   
    return -(pos_loss + neg_loss).mean()
