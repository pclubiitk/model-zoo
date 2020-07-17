import os
import nltk
import json
from collections import Counter
from preprocess import load_captions

class Vocabulary():

    def __init__(self, captions_dict, threshold):
        self.word2id = {}
        self.id2word = {}
        self.index = 0
        self.build(captions_dict, threshold)

    def add_word(self, word):
        if word not in self.word2id:
            self.word2id[word] = self.index
            self.id2word[self.index] = word
            self.index += 1

    def get_id(self, word):
        if word in self.word2id:
            return self.word2id[word]
        return self.word2id['<unk>']

    def get_word(self, index):
        return self.id2word[index]

    def build(self, captions_dict, threshold):
        counter = Counter()
        tokens = []
        for i, captions in captions_dict.items():
            for caption in captions:
                tokens.extend(nltk.tokenize.word_tokenize(caption.lower()))

            counter.update(tokens)
            words = [word for word, count in counter.items() if count >= threshold]
            # threshold to prevent useless things
            self.add_word('<unk>')
            self.add_word('<start>')
            self.add_word('<end>')
            self.add_word('<pad>')

            for word in words:
                self.add_word(word)

    def get_sentence(self, ids_list):
        sentence = ''
        for id in ids_list:
            curr_word = self.id2word[id.item()]
            sentence += ' ' + curr_word
            if curr_word == '<end>':
                break

        return sentence

if __name__ == '__main__':
    captions_dict = load_captions('train')
    vocab = Vocabulary(captions_dict, 5) # threshold = 5
    print(vocab.index)
