from collections import Counter, defaultdict
import numpy as np
import torch

class GloveDataset:

    def __init__(self, text, n_words=200000, window_size=5):
        self._window_size = window_size
        self._tokens = text.split(" "[:n_words])
        word_counter = Counter()
        word_counter.update(self._tokens)
        self._word2id = {w:i for i,(w,_) in enumerate(word_counter.most_common())}
        self._id2word = {i:w for w,i in self._word2id.items()}
        self._vocab_len = len(self._word2id)
        self._id_tokens = [self._word2id[w] for w in self._tokens]

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.create_cooccurence_matrix(device)

        print('No. of words: {}'.format(len(self._tokens)))
        print('Vocabulary length : {}'.format(self._vocab_len))


    def create_cooccurence_matrix(self, device):
        cooc_mat = defaultdict(Counter)
        for i,w in enumerate(self._id_tokens):
            start = max(i-self._window_size, 0)
            end = min(i+self._window_size + 1, len(self._id_tokens))

            for j in range(start, end):
                if i!=j :
                    c = self._id_tokens[j]
                    cooc_mat[w][c] += 1/abs(j-i)

        self._i_idx = list()
        self._j_idx = list()
        self._xij = list()

        #Create indexes and x values tensors
        for w, cnt in cooc_mat.items():
            for c, v in cnt.items():
                self._i_idx.append(w)
                self._j_idx.append(c)
                self._xij.append(v)

        self._i_idx = torch.LongTensor(self._i_idx).to(device)
        self._j_idx = torch.LongTensor(self._j_idx).to(device)
        self._xij = torch.FloatTensor(self._xij).to(device)

    def get_batches(self, batch_size):
        rand_ids = torch.LongTensor(np.random.choice(len(self._xij), len(self._xij), replace=False))

        for p in range(0, len(rand_ids), batch_size):
            batch_ids = rand_ids[p:p+batch_size]
            yield self._xij[batch_ids], self._i_idx[batch_ids], self._j_idx[batch_ids]
