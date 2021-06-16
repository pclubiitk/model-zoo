import torch
import torch.nn as nn
import torchvision.models as model

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN,self).__init__()

        # pretrained GoogLeNet model from PyTorch , aux_logits = False because we do not need to train
        # the whole model again and auxillary outputs is useful only while training.
        self.model = torch.hub.load('pytorch/vision:v0.9.0', 'googlenet', pretrained=True)

        # freezing all the layers of the GoogLeNet model
        for param in self.model.parameters():
            param.requires_grad = False

        # replacing the last layer of the model with a linear layer with output size as the embed_size
        self.model.fc = nn.Linear(self.model.fc.in_features,embed_size)

        # dropout layer
        self.dropout = nn.Dropout(p=0.5)

        # activation layer
        self.relu = nn.ReLU()

    def forward(self, images):
        # getting the latent representation of our image
        output = self.model(images)

        # applying relu activation and dropout layer
        return self.dropout(self.relu(output))

class DecoderLSTM(nn.Module):
    def __init__(self,embed_size,hidden_size,vocab_size,num_layers):
        super(DecoderLSTM,self).__init__()

        # defining our class properties
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # LSTM Cell
        self.lstm = nn.LSTM(input_size=self.embed_size,hidden_size=self.hidden_size,num_layers=self.num_layers)

        # Output Layer
        self.linear = nn.Linear(in_features=self.hidden_size,out_features=self.vocab_size)

        # Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=self.vocab_size,embedding_dim=self.embed_size)

        # Dropout Layer
        self.dropout = nn.Dropout(p = 0.5)

    def forward(self, features, captions):
        # Getting the word embeddings for the captions
        word_embeddings = self.dropout(self.embedding(captions))

        # our input vector will be the feature vector from the images + target captions during each time step
        # during training
        word_embeddings = torch.cat((features.unsqueeze(0), word_embeddings), dim = 0)

        # Getting the cell state and the hidden state from the LSTM
        hidden_state, cell_state = self.lstm(word_embeddings)

        # Applying the linear layer to get a probability distribution output
        return self.linear(hidden_state)

class ImageToCaption(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(ImageToCaption,self).__init__()

        # Initializing the Encoder CNN
        self.encoder = EncoderCNN(embed_size)

        # Initializing the Decoder LSTM
        self.decoder = DecoderLSTM(embed_size,hidden_size,vocab_size,num_layers)

    def forward(self,images,captions):
        # Getting the latent representation of the image from the encoder
        features = self.encoder(images)

        # Returning the captions from the decoder LSTM during training time.
        return self.decoder(features,captions)

    def generate_caption(self,image,vocabulary,max_length = 50):
        # Initializing an empty list
        generated_caption = []

        # Inference time so no grad is required
        with torch.no_grad():

            # Defining the initial Input and the cell state
            x = self.encoder(image).unsqueeze(0)
            state = None

            for _ in range(max_length):
                # finding the hidden and cell states
                hidden, state = self.decoder.lstm(x,state)

                # applying the linear layer on the hidden state to get the output distribution
                output = self.decoder.linear(hidden.squeeze(0))

                # find out the word with the highest probability
                predicted = output.argmax(1)

                # appending the index of the word in our generated_caption list
                generated_caption.append(predicted.item())

                # setting the input for the next iteration
                x = self.decoder.embedding(predicted).unsqueeze(0)

                # if our model predicts End of Sequence then we just stop
                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        # we convert the indices to the words
        return [vocabulary.itos[idx] for idx in generated_caption]
