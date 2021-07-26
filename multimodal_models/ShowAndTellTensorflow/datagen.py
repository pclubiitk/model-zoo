from tensorflow.keras.preprocessing.sequence import pad_sequences
from data_process import *
from encoder import *
num_img = len(train_image_names)


# Function to load data in the form of which it can be fed in the model
def batch_generator(batchsize=128):
    while True:
        temp = np.random.randint(num_img, size=batchsize)

        tokens = one_of_five_caps(temp)
        len_tok = [len(x) for x in tokens]
        len_tok = max(len_tok)
        padded = pad_sequences(tokens, maxlen=len_tok, padding='post')

        transf = train_transfer[temp]

        decoder_input_data = padded[:, 0:-1]
        decoder_output_data = padded[:, 1:]
        x_data = {
            'transfer_values_input': transf,
            'decoder_input': decoder_input_data
        }
        y_data = {
            'decoder_output': decoder_output_data
        }

        yield (x_data, y_data)


generator = batch_generator()
