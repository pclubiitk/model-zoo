from model import Word2Vec, ScoringLayer, EmbeddingLayer
from utils import constructBagOfWordsInWindowSize, contextPairToOneHot, OneHotOfAllInVocab
from keras.callbacks import TensorBoard 
from dataloader import tokenizeData, performTokenization
import argparse
import datetime
from numpy import save, load
from evaluation import getSimilarity, getSimilarityByEmbedding, getTenClosestWords, analogy, plotEmbeddingsIn2D
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser()

    #optim config
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2000)
    parser.add_argument('--optimizer', type=str, default="adam")
    #Model config
    parser.add_argument('--dim_embedding', type=int, default=100)
    #evaluation
    parser.add_argument('--mode', default="train", type=str)
    #getSimilarity 
    parser.add_argument('--word1', type=str, default="window")
    parser.add_argument('--word2', type=str, default="hoouse")
    #getTenClosestWords
    parser.add_argument('--word', type=str, default="window")
    #analogy
    parser.add_argument('--word1_', type=str, default="window")
    parser.add_argument('--word2_', type=str, default="hoouse")
    parser.add_argument('--word3_', type=str, default="door")
    #wordIsInVocab
    parser.add_argument('--word_', type=str)

    args = parser.parse_args()

    optim_config = OrderedDict([
        ('epochs', args.epochs),
        ('batch_size', args.batch_size),
        ('optimizer', args.optimizer)
    ])

    model_config = OrderedDict([
        ('dim_embedding', args.dim_embedding)
    ])
    
    evaluation_config = OrderedDict([
        ('word1', args.word1),
        ('word2', args.word2),
        ('word', args.word),
        ('word1_', args.word1_),
        ('word2_', args.word2_),
        ('word3_', args.word3_),
        ('word_', args.word_),
    ])

    
    config = OrderedDict([
        ('optim_config', optim_config),
        ('evaluation_config', evaluation_config),
        ('model_config', model_config),
        ('mode', args.mode),
    ])

    return config

config = parse_args()

model_config = config['model_config']
optim_config = config['optim_config']
evaluation_config = config['evaluation_config']
mode = config['mode']
# log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)


tokenized_data = performTokenization()
context_tuple_list = constructBagOfWordsInWindowSize(tokenized_data)
oneHotNumpy, data = contextPairToOneHot(context_tuple_list, tokenized_data)
print("The total number of words in corpus size are: ",data["vocabSize"])


if(mode == "train"):
    def train():

        dimensionality_of_embeddings = model_config['dim_embedding']
        optimizer = optim_config['optimizer']
        epochs = optim_config['epochs']
        batch_size = optim_config['batch_size']

        model = Word2Vec(input_dim=data['vocabSize'], units = int(dimensionality_of_embeddings))
        model.compile(loss='categorical_crossentropy',
                    optimizer= optimizer,
                    metrics= ['accuracy'])
        model.fit(oneHotNumpy[:,0,:],oneHotNumpy[:,1,:],
                epochs = epochs,
                batch_size = batch_size)


        emb = model.get_weights()[0]
        save("word2vec_embeddings.npy",emb)

elif(mode == "help"):
    print("$ python3 main.py --epochs 100 --optimizer \"adam\" --batch_size 2000 --dim_embedding 100\n")
    print("$ python3 main.py --mode \"getSimilarity\" --word1 \"window\" --word2 \"house\"\n")
    print("$ python3 main.py --mode \"getTenClosestWords\" --word \"window\"\n")
    print("$ python3 main.py --mode \"analogy\" --word1_ \"window\" --word2_ \"house\" --word3_ \"door\"\n")
    print("$ python3 main.py --mode \"plot\"")
    print("$ python3 main.py --mode \"help\"")
    print("$ python3 main.py --mode \"wordIsInVocab\" --word_ \"window\"")

else:
    emb = load("embeddings.npy")

    if(mode == "getSimilarity"):
        word1 = evaluation_config['word1']
        word2 = evaluation_config['word2']

        print(getSimilarity(word1, word2, data, emb))

    if(mode == "getTenClosestWords"):
        word = evaluation_config['word']

        print(getTenClosestWords(word, data['vocab'], data, emb))

    if(mode == "analogy"):
        word1_ = evaluation_config['word1_']
        word2_ = evaluation_config['word2_']
        word3_ = evaluation_config['word3_']

        print(analogy(word1_, word2_, word3_, data, data['vocab'], emb))
    
    if(mode == "wordIsInVocab"):
        word_ = evaluation_config['word_']
        vocabList = data['vocab'].tolist()

        if word_ in vocabList:
            print("YES")
        else:
            print("NO")

    if(mode == "plot"):
        plotEmbeddingsIn2D(emb, data)
