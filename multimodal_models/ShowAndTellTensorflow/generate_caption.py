import matplotlib.pyplot as plt
from model import *


# Function to print the caption predicted by the model for a given image
def generate_caption(image_path, max_tokens=30):
    imgp = image.load_img(image_path, target_size=(299, 299))
    imgp = image.img_to_array(imgp)
    img1 = imgp
    imgp = np.expand_dims(imgp, axis=0)
    imgp = preprocess_input(imgp)

    transfer_values = cnn_encoder.predict(imgp)

    shape = (1, max_tokens)
    decoder_input_data = np.zeros(shape=shape, dtype=np.int)

    token_int = word_index['startseq']
    token_end = word_index['endseq']
    output_text = ''

    count_tokens = 0

    while token_int != token_end and count_tokens < max_tokens:

        decoder_input_data[0, count_tokens] = token_int

        x_data = \
            {
                'transfer_values_input': transfer_values,
                'decoder_input': decoder_input_data
            }

        decoder_output2 = decoder_model.predict(x_data)

        token_onehot = decoder_output2[0, count_tokens, :]

        token_int = np.argmax(token_onehot)

        sampled_word = index_word[token_int]
        if sampled_word != 'endseq':
            output_text += " " + sampled_word

        count_tokens += 1

    output_tokens = decoder_input_data[0]

    plt.imshow(img1)
    plt.show()

    print("Predicted caption:")
    print(output_text)


# Enter the path of image for which caption is to be generated
img_pred_path = "../input/flickr8k/Flickr_Data/Flickr_Data/Images/543363241_74d8246fab.jpg"
generate_caption(img_pred_path)
