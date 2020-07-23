from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D, Dropout, ZeroPadding2D, LSTM, TimeDistributed, Embedding, Bidirectional, Activation, RepeatVector, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import InceptionV3



def VGG_16(weights_path=None):
 
    model = Sequential()
    model.add(ZeroPadding2D((1,1), input_shape=(224, 224, 3), data_format='channels_last'))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64, kernel_size=(3, 3), strides=1, activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2), data_format='channels_last'))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)
        
    return model


def vgg_model():

	vgg_url = "https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5"
	r = requests.get(vgg_url, allow_redirects=True)
	open('vgg16_weights_tf_dim_ordering_tf_kernels.h5', 'wb').write(r.content)

	vmodel = VGG_16('vgg16_weights_tf_dim_ordering_tf_kernels.h5') 
	vmodel = Model(inputs=vmodel.inputs, outputs=vmodel.layers[-2].output)
	return vmodel


def inception_model():

	inc_model = InceptionV3(include_top=True, weights='imagenet')
	inc_model = Model(inputs = inc_model.inputs, outputs = inc_model.layers[-2].output)
	return inc_model


def rnn_cnn_model( ishape, max_len, vocab_size, optimizer_name ):

	inputs1 = Input(shape=(ishape,))
	image_model = Dense(256, activation='relu')(inputs1)
	image_model = RepeatVector(max_len)(image_model)

	inputs2 = Input(shape=(max_len,))
	language_model = Embedding(input_dim=vocab_size, output_dim=256, mask_zero=True)(inputs2)
	language_model = LSTM(256, return_sequences=True)(language_model)
	language_model = TimeDistributed(Dense(256))(language_model)

	concate = Concatenate()([image_model, language_model])
	x = LSTM(256, return_sequences=False)(concate)
	out = Dense(vocab_size, activation='softmax')(x)

	model = Model( outputs = out, inputs=[inputs1, inputs2])
	model.compile(loss='categorical_crossentropy', optimizer=optimizer_name, metrics=['accuracy'])

	return model
	
