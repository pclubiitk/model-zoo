import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

# Downloading the dataset
# !kaggle datasets download -d harmanpreet93/hotelreviews
# unzip the dataset and keep it in a folder named hotelreviews




def tokenizeData(indv_lines):
  review_data_list = list()
  for line in indv_lines:
    tokenizer = RegexpTokenizer('\w+')
    tokens = tokenizer.tokenize(line)

    words = [word.lower() for word in tokens]

    stop_word_list = set(stopwords.words('english'))
    words = [w for w in words if not w in stop_word_list]

    review_data_list.append(words)

    return review_data_list

def performTokenization():
    hotel_data = pd.read_csv('./hotelreviews/hotel-reviews.csv')
    hotel_data = hotel_data['Description'].tolist()
    hotel_data = hotel_data[0:100]#you can increase the upper limit depending on your ram size

    indv_lines = hotel_data

    return tokenizeData(indv_lines)
