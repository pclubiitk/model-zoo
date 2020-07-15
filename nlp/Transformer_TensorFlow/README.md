# TensorFlow Implementation of Transformers

### Dataset used

TED talks dataset for translation from Portuguese to English : https://github.com/neulab/word-embeddings-for-nmt

### Usage
```bash
$ python3 main.py 
```
NOTE: on Colab Notebook use following command:
```bash
!git clone link-to-repo
%run main.py 
```

### Help Log
```

usage: main.py [-h] [--EPOCHS EPOCHS] [--num_layers NUM_LAYERS]
               [--d_model D_MODEL] [--dff DFF] [--num_heads NUM_HEADS]
               [--BUFFER_SIZE BUFFER_SIZE] [--BATCH_SIZE BATCH_SIZE]
               [--MAX_LENGTH MAX_LENGTH] [--dropout_rate DROPOUT_RATE]
               [--beta_1 BETA_1] [--beta_2 BETA_2] [--input INPUT]
               [--real_translation REAL_TRANSLATION] [--outdir OUTDIR]
               [--plot PLOT]

optional arguments:
  -h, --help            show this help message and exit
  --EPOCHS EPOCHS       No of training epochs
  --num_layers NUM_LAYERS
                        No of layers of encoder and decoder
  --d_model D_MODEL     dimension
  --dff DFF             dimension
  --num_heads NUM_HEADS
                        No of attention heads
  --BUFFER_SIZE BUFFER_SIZE
                        Buffer size
  --BATCH_SIZE BATCH_SIZE
                        Batch size
  --MAX_LENGTH MAX_LENGTH
                        Maximum allowable length of input and output sentences
  --dropout_rate DROPOUT_RATE
                        Dropout rate
  --beta_1 BETA_1       Exponential decay rate for 1st moment
  --beta_2 BETA_2       Exponential decay rate for 2nd moment
  --input INPUT         Input sentence in portuguese
  --real_translation REAL_TRANSLATION
                        Real translation of input sentence in English
  --outdir OUTDIR       Directory in which to store data
  --plot PLOT           Decoder layer and block whose attention weights are to
                        be plotted
                        
```

### Contributed by:
* [Ashish Murali](https://github.com/ashishmurali)

## References :

* **Title**: Attention Is All You Need
* **Authors**: Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin
* **Link**: https://arxiv.org/abs/1706.03762
* **Tags**: Neural Network, NLP, transformers, supervised learning
* **Year**: 2017

## Summary

* **What are Transformers**:

  The paper ‘Attention Is All You Need’ introduces a novel architecture called Transformer which uses attention mechanism. Transformer is an architecture for 
  transforming one sequence into another one with the help of two parts (Encoder and Decoder), but it differs from the existing sequence-to-sequence 
  models because it does not imply any Recurrent Networks (GRU, LSTM, etc.). The core idea behind the Transformer model is self-attention—the ability to attend 
  to different positions of the input sequence to compute a representation of that sequence. 
  
  A transformer model handles variable-sized input using stacks of self-attention layers instead of RNNs or CNNs. This general architecture has a number of         advantages:

  * It make no assumptions about the temporal/spatial relationships across the data. This is ideal for processing a set of objects (for example, StarCraft units).
  * Layer outputs can be calculated in parallel, instead of a series like an RNN.
  * Distant items can affect each other's output without passing through many RNN-steps, or convolution layers (see Scene Memory Transformer for example).
  * It can learn long-range dependencies. This is a challenge in many sequence tasks.

* **Model Architecture**:

  Transformer consists of the encoder, decoder and a final linear layer. The output of the decoder is the input to the linear layer and its output is returned.

<p align="center">
  <img src="https://github.com/ashishmurali/model-zoo/blob/master/NLP/Transformer_Tensorflow/assets/transformer.png">
</p>   
  
* **Encoder**:

  The input is put through an embedding which is summed with the positional encoding. The output of this summation is the input to the encoder layers. 
  Each encoder layer consists of sublayers:
  * Multi-head attention (with padding mask)
  * Point wise feed forward networks.
  Each of these sublayers has a residual connection around it followed by a layer normalization. Residual connections help in avoiding the vanishing gradient       problem in deep networks.The output of the encoder is the input to the decoder.

* **Decoder**:

  The target is put through an embedding which is summed with the positional encoding. The output of this summation is the input to the decoder layers. 
  The output   of the decoder is the input to the final linear layer.
  Each decoder layer consists of sublayers:
  * Masked multi-head attention (with look ahead mask and padding mask)
  * Multi-head attention (with padding mask). V (value) and K (key) receive the encoder output as inputs. Q (query) receives the output from the masked 
    multi-head attention sublayer.
  * Point wise feed forward networks

* **Scaled Dot Product Attention**

  <p align="center">
  <img src="https://github.com/ashishmurali/model-zoo/blob/master/NLP/Transformer_Tensorflow/assets/scaled_attention.png">
  </p>   
   
  The attention function used by the transformer takes three inputs: Q (query), K (key), V (value). The equation used to calculate the attention weights is:
  
  ![attention](https://github.com/ashishmurali/model-zoo/blob/master/NLP/Transformer_Tensorflow/assets/attention.png)
  
* **Multi-head attention**

  <p align="center">
  <img src="https://github.com/ashishmurali/model-zoo/blob/master/NLP/Transformer_Tensorflow/assets/multi_head_attention.png">
  </p>  
  
  Each multi-head attention block gets three inputs; Q (query), K (key), V (value). These are put through linear (Dense) layers and split up into multiple heads.
  The scaled_dot_product_attention defined above is applied to each head (broadcasted for efficiency). An appropriate mask must be used in the attention step. The   attention output for each head is then concatenated (using tf.transpose, and tf.reshape) and put through a final Dense layer.
  
-------------------------

## Our implementation :

* We have implemented a transformer model to translate Portuguese to English.
* The default hyperparameters used in the model are similar to those given in the 'Attention is All You Need' paper.
* We used [TFDS](https://www.tensorflow.org/datasets) to load the [Portugese-English translation dataset](https://github.com/neulab/word-embeddings-for-nmt)
  from the [TED Talks Open Translation Project](https://www.ted.com/participate/translate).
  This dataset contains approximately 50000 training examples, 1100 validation examples, and 2000 test examples.
* We used the Adam optimizer with a custom learning rate scheduler according to the formula in the paper. 
* By default we are training the model for 20 epochs. After training we can translate the input Portuguese sentence to English and plot the attention weight of     any decoder layer.

## Results of our implementation :

* After training we translated the following Portuguese sentence to English and plotted the attention weight of all heads in the 2nd attention block of 4th decoder layer.

* **INPUT PORTUGUESE SENTENCE** : este é o primeiro livro que eu fiz.
* **REAL ENGLISH TRANSLATION** : this is the first book i've ever done.
* **PREDICTED ENGLISH TRANSLATION** : this is the first book that i did .

![plot_attention](https://github.com/ashishmurali/model-zoo/blob/master/NLP/Transformer_Tensorflow/assets/plot_attention.png)

* The results after training for 20 epochs :

1. Train Accuracy 

![train_accuracy](https://github.com/ashishmurali/model-zoo/blob/master/NLP/Transformer_Tensorflow/assets/train_accuracy.png)

2. Test Accuracy

![test_accuracy](https://github.com/ashishmurali/model-zoo/blob/master/NLP/Transformer_Tensorflow/assets/test_accuracy.png)

3. Train Loss

![train_loss](https://github.com/ashishmurali/model-zoo/blob/master/NLP/Transformer_Tensorflow/assets/train_loss.png)

4. Test Loss

![test_loss](https://github.com/ashishmurali/model-zoo/blob/master/NLP/Transformer_Tensorflow/assets/test_loss.png)


> **_NOTE:_** The above graphs were plotted using tensorboard

## Sources:
* [Transformer model for language understanding](https://www.tensorflow.org/tutorials/text/transformer)
