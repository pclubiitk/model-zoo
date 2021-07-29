import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split

# download spacy first with pip install spacy
# This is the version for the english language
spacy_eng = spacy.load("en_core_web_sm")

class Vocabulary :
    def __init__(self, freq_threshold):
        # itos will contain the mapping of indices to the corresponding words
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}

        # stoi is just the inverse mapping of itos
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}

        # the number of times a word has to occur in the text for it to be eligible to be
        # part of our vocabulary
        self.freq_threshold = freq_threshold

    # will return us the length of our vocabulary
    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    # building our vocabulary from the set of sentences given to us
    def build_vocabulary(self, sentence_list):
        # to keep track of the frequency of a word appearing in our sentences
        frequencies = {}

        # starting from 4 as the first 4 are already taken up by the key words
        idx = 4

        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]

class Flickr8k(Dataset):
    def __init__(self,images_path,captions_file,transform = None,freq_threshold=4):
        self.images_path = images_path
        self.captions_file = captions_file
        self.dataframe = pd.read_csv(captions_file)
        self.transform = transform

        # Get img, caption columns
        self.imgs = self.dataframe["image"]
        self.captions = self.dataframe["caption"]

        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions.tolist())

        # Splitting it into train and test datasets
        self.train_imgs , self.test_imgs , self.train_captions , self.test_captions = train_test_split(self.imgs,self.captions,test_size=0.2,train_size=0.8,random_state=1,shuffle=True)

        self.train_imgs.reset_index(drop=True, inplace=True)
        self.train_captions.reset_index(drop=True, inplace=True)
        self.test_imgs.reset_index(drop=True, inplace=True)
        self.test_captions.reset_index(drop=True, inplace=True)

    def get_training_data(self):
        return self.train_imgs, self.train_captions

    def get_testing_data(self):
        return self.test_imgs, self.test_captions


class Flickr8k_Training(Dataset):
    def __init__(self, images_path , train_imgs, train_captions, vocab, transform=None):
        self.images_path = images_path
        self.transform = transform
        self.imgs = train_imgs
        self.captions = train_captions

        # Initialize vocabulary and build vocab
        self.vocab = vocab

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.images_path, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)


class Flickr8k_Testing(Dataset):
    def __init__(self, images_path, test_imgs, test_captions, vocab, transform=None):
        self.images_path = images_path
        self.transform = transform
        self.imgs = test_imgs
        self.captions = test_captions

        # Initialize vocabulary and build vocab
        self.vocab = vocab

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.imgs[index]
        img = Image.open(os.path.join(self.images_path, img_id)).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return img, torch.tensor(numericalized_caption)

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        imgs = [item[0].unsqueeze(0) for item in batch]
        imgs = torch.cat(imgs, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return imgs, targets

def get_loader(root_folder,
    annotation_file,
    transform,
    batch_size=32,
    num_workers=2,
    shuffle=True,
    pin_memory=True,
):
    dataset = Flickr8k(root_folder,annotation_file,transform)
    train_imgs , train_captions = dataset.get_training_data()
    test_imgs , test_captions = dataset.get_testing_data()

    train_dataset = Flickr8k_Training(root_folder,train_imgs,train_captions,dataset.vocab,transform)
    test_dataset = Flickr8k_Testing(root_folder,test_imgs,test_captions,dataset.vocab,transform)

    pad_idx_train = train_dataset.vocab.stoi["<PAD>"]
    pad_idx_test = test_dataset.vocab.stoi["<PAD>"]

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx_train),
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx_test),
    )

    return train_loader, test_loader, train_dataset, test_dataset


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), ]
    )
    train_loader, test_loader, train_dataset, test_dataset = get_loader(
        "flickr8k/Images/", "flickr8k/captions.txt", transform=transform
    )
    for idx, (data, image) in enumerate(test_loader):
        print(idx)
