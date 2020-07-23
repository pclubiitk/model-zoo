from PIL import Image
from utils import preprocess_image
from model import vgg_model, inception_model, rnn_cnn_model
import argparse


parser = argparse.ArgumentParser()

parser.add_argument('--model', type = str, default = "inception", help = "Model (adam/inception) , default inception")
parser.add_argument('--path', type = str, default = "", help = "path of the image with name")
parser.add_argument('--optimizer', type = str, default = "RMSprop", help = "Optimizer for the mrnn model (adam/RMSprop), default RMSprop")
args = parser.parse_args()



def word_2_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
 

# generate a description for an image

def caption(mod, tokenizer, feature, max_length):
    in_text = 'sos'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = mod.predict([feature,sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = word_2_id(yhat, tokenizer)
        if word is None:
            break
        in_text += ' ' + word
        if word == 'eos':
            break
    return in_text





if args.model == "inception":
	eval_image = preprocess_image_inception(args.path)
	model = inception_model()
	shape = 2048
else:
	eval_image = preprocess_image_vgg(args.path)
	model = vgg_model()
	shape = 4096


	
feature = model.predict(eval_image)
tokenizer = pickle.load(open("flickr_tokenizer.pkl", 'rb'))
vocab_size = len(tokenizer.word_index) + 1

mrnn_model = rnn_cnn_model(shape, 34, vocab_size, args.optimizer)
mrnn_model.load_weights('rc_model_weights.h5')
description = caption(mrnn_model, tokenizer, feature, max_len)
print(description)
