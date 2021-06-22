import torch
import torch.nn as nn
import torchvision.models as model
from tqdm import tqdm
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image  # Load img
from utils import save_checkpoint, load_checkpoint
from data_processing import get_loader, Flickr8k_Testing, Flickr8k_Training, Flickr8k, Vocabulary, MyCollate
from model import ImageToCaption
from train import train


def print_caption(model, device, dataset):
    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    model.eval()
    test_img = transform(Image.open("child.jpg").convert("RGB")).unsqueeze(0)
    print(generate_caption(model, test_img.to(device), dataset.vocab))


def generate_caption(model, image, vocabulary, max_length=50):
    model.eval()
    # Initializing an empty list
    generated_caption = []

    # Inference time so no grad is required
    with torch.no_grad():

        # Defining the initial Input and the cell state
        x = model.encoder(image).unsqueeze(0)
        state = None

        for _ in range(max_length):
            # finding the hidden and cell states
            hidden, state = model.decoder.lstm(x, state)

            # applying the linear layer on the hidden state to get the output distribution
            output = model.decoder.linear(hidden.squeeze(0))

            # find out the word with the highest probability
            predicted = output.argmax(1)

            # appending the index of the word in our generated_caption list
            generated_caption.append(predicted.cpu().detach().numpy().tolist())

            # setting the input for the next iteration
            x = model.decoder.embedding(predicted).unsqueeze(0)

            # if our model predicts End of Sequence then we just stop
            if vocabulary.itos[predicted.item()] == "<EOS>":
                break

    # return generated_caption
    # we convert the indices to the words
    caption = []
    for i in range(len(generated_caption)):
        idx = int(generated_caption[i][0])
        caption.append(vocabulary.itos[idx])
    return caption


if __name__ == "__main__":
    train_dataloader = torch.load("train_dataloader.pt")
    test_dataloader = torch.load("test_dataloader.pt")
    train_dataset = torch.load("train_dataset.pt")
    test_dataset = torch.load("test_dataset.pt")

    # Hyper-Parameters
    embed_size = 256
    hidden_size = 256
    vocab_size = len(train_dataset.vocab)
    num_layers = 1
    learning_rate = 3e-4
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    # initialize model, loss etc
    model = ImageToCaption(embed_size, hidden_size, vocab_size, num_layers).to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    step = load_checkpoint(torch.load("saved_checkpoint.pt"), model, optimizer)

    print_caption(model, device, train_dataset)
