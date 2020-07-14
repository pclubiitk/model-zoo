import tensorflow as tf

class Load_Data:
    def __init__(self,MAX_LENGTH,tokenizer_en,tokenizer_pt):
        self.MAX_LENGTH = MAX_LENGTH
        self.tokenizer_pt = tokenizer_pt
        self.tokenizer_en = tokenizer_en
    
    def encode(self,lang1, lang2):
        lang1 = [self.tokenizer_pt.vocab_size] + self.tokenizer_pt.encode(
            lang1.numpy()) + [self.tokenizer_pt.vocab_size+1]

        lang2 = [self.tokenizer_en.vocab_size] + self.tokenizer_en.encode(
            lang2.numpy()) + [self.tokenizer_en.vocab_size+1]
        
        return lang1, lang2

    def tf_encode(self,pt, en):
        result_pt, result_en = tf.py_function(self.encode, [pt, en], [tf.int64, tf.int64])
        result_pt.set_shape([None])
        result_en.set_shape([None])

        return result_pt, result_en    

    def filter_max_length(self,x, y):
        return tf.logical_and(tf.size(x) <= self.MAX_LENGTH,
                                tf.size(y) <= self.MAX_LENGTH)    
        