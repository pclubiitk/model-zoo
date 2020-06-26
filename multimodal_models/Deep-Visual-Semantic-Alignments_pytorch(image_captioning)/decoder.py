import torch
import torch.nn as nn

class RNN(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        super(RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, vocab_size)
        self.init_weights()

    def init_weights(self):
        self.word_embeddings.weight.data.uniform_(-0.1,0.1)
        self.linear.weight.data.uniform_(-0.1,0.1)
        self.linear.bias.data.fill_(0)

    def forward(self, features, caption):
        embeds = self.word_embeddings(caption)
        embeds = torch.cat((features, embeds), 0)
        lstm_out, _ = self.lstm(embeds.unsqueeze(1))
        output = self.linear(lstm_out.view(len(caption)+1, -1))
        return output

    def greedy(self, cnn_out, seq_len = 20):
        inp = cnn_out
        hidden = None
        ids_list = []
        for t in range(seq_len):
            lstm_out, hidden = self.lstm(inp.unsqueeze(1), hidden)
            linear_out = self.linear(lstm_out.squeeze(1))
            word_caption = linear_out.max(dim=1)[1]
            ids_list.append(word_caption)
            inp = self.word_embeddings(word_caption)
        return ids_list
