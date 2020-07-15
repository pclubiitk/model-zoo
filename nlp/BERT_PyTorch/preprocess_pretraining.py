from utils import truncate_tokens_pair,get_random_word
from pytorch_pretrained_bert.tokenization import BertTokenizer
from random import random as rand
from random import randint,shuffle
import random

class PreProcess():
    """ Pre-processing steps for pretraining transformer """
    def __init__(self, max_pred, mask_prob, max_len):
        super().__init__()
        self.max_pred = max_pred 
        self.mask_prob = mask_prob 
        self.indexer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_len = max_len

    def __call__(self,data):
        is_next, tokens_a, tokens_b = data
        truncate_tokens_pair(tokens_a, tokens_b, self.max_len - 3)

        # Add Special Tokens
        tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']
        segment_ids = [0]*(len(tokens_a)+2) + [1]*(len(tokens_b)+1)
        input_mask = [1]*len(tokens)

        # For masked Language Models
        masked_tokens, masked_pos = [], []
        n_pred = min(self.max_pred, max(1, int(round(len(tokens)*self.mask_prob))))
        cand_pos = [i for i, token in enumerate(tokens)
                    if token != '[CLS]' and token != '[SEP]']
        shuffle(cand_pos)
        for pos in cand_pos[:int(n_pred)]:
            masked_tokens.append(tokens[pos])
            masked_pos.append(pos)
            if rand() < 0.8: # 80%
                tokens[pos] = '[MASK]'
            elif rand() < 0.5: # 10%
                tokens[pos] = get_random_word(self.indexer.vocab)
        masked_weights = [1]*len(masked_tokens)

        # Token Indexing
        input_ids = self.indexer.convert_tokens_to_ids(tokens)
        masked_ids = self.indexer.convert_tokens_to_ids(masked_tokens)

        # Zero Padding
        n_pad = self.max_len - len(input_ids)
        input_ids.extend([0]*int(n_pad))
        segment_ids.extend([0]*int(n_pad))
        input_mask.extend([0]*int(n_pad))

        # Zero Padding for masked target
        if self.max_pred > n_pred:
            n_pad = self.max_pred - n_pred
            masked_ids.extend([0]*int(n_pad))
            masked_pos.extend([0]*int(n_pad))
            masked_weights.extend([0]*int(n_pad))

        return (input_ids, segment_ids, input_mask, masked_ids, masked_pos, masked_weights, is_next)
