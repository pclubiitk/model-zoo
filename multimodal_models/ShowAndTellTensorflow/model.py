from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding
from tensorflow.keras.optimizers import RMSprop
from datagen import *

# Defining the layers to be used in the model
transfer_values_input = Input(shape=(2048,), name='transfer_values_input')
decoder_transfer_map = Dense(256, activation='tanh')
decoder_input = Input(shape=(None, ), name='decoder_input')
decoder_embedding = Embedding(input_dim=5000, output_dim=128, name='decoder_embedding')
decoderlstm = LSTM(256, return_sequences=True)
decoder_dense = Dense(5000, activation='softmax', name='decoder_output')


# Function to get the output of the decoder, given output of encoder
def connect_decoder(transfer_values):
    state = decoder_transfer_map(transfer_values)
    initial_state = [state, state]
    # Start the decoder-network with its input-layer.
    net = decoder_input

    net = decoder_embedding(net)

    net = decoderlstm(net, initial_state=initial_state)

    decoder_output1 = decoder_dense(net)

    return decoder_output1


decoder_output = connect_decoder(transfer_values=transfer_values_input)

# Defining, compiling, training, saving the model
decoder_model = Model(inputs=[transfer_values_input, decoder_input], outputs=[decoder_output])

decoder_model.compile(optimizer=RMSprop(lr=1e-3), loss='sparse_categorical_crossentropy')

decoder_model.fit(generator, steps_per_epoch=1700, epochs=25)

# Enter the path of output directory where model_weights can be saved
output_dir = './'
decoder_model.save_weights(output_dir)
