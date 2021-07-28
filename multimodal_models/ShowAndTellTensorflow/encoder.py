from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from data_process import *

# Loading the Pre-trained CNN model for image processing
model1 = InceptionV3()
cnn_encoder = Model(inputs=model1.input, outputs=model1.layers[-2].output)


# Function to get cnn_encoder output of the images ( in batches )
def transfer_vals(img_dir, img_names, batch=32):
    transfers = np.zeros((len(img_names), 2048), dtype=np.float16)

    start = 0
    total = len(img_names)

    while start < total:
        img_batch = np.zeros((batch, 299, 299, 3), dtype=np.float16)
        end = start + batch

        if end > total:
            end = total

        curr = end - start

        for i, j in enumerate(img_names[start:end]):
            imgt = image.load_img(img_dir + j, target_size=(299, 299))
            imgt = image.img_to_array(imgt)
            imgt = np.expand_dims(imgt, axis=0)
            imgt = preprocess_input(imgt)
            img_batch[i] = imgt

        temp = cnn_encoder.predict(img_batch[0:curr])

        transfers[start:end] = temp[0:curr]

        start = end

    return transfers


# Enter the path of directory containing all the images in img_dir variable
img_dir = "../input/flickr8k/Flickr_Data/Flickr_Data/Images/"
train_transfer = transfer_vals(img_dir, train_image_names)
